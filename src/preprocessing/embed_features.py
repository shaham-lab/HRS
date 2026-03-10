"""
embed_features.py – BERT sentence embeddings for all text features.

Loads text parquets from FEATURES_DIR, embeds each text column using
mean pooling over all non-padding content tokens from the configured BERT
model, and saves embedding parquets to EMBEDDINGS_DIR.

Also reads labs_features.parquet (long format) and builds 13 per-lab-group
embedding parquets, one per group defined in lab_panel_config.yaml.

Device selection:
  - Uses BERT_DEVICE from config.
  - Falls back to CPU with a warning if CUDA is requested but unavailable.

Empty / null text → zero vector of the same embedding dimension.

Expected config keys:
    FEATURES_DIR      – directory containing raw text feature parquets
    EMBEDDINGS_DIR    – output directory for embedding parquets
    CLASSIFICATIONS_DIR – directory containing lab_panel_config.yaml
    PREPROCESSING_DIR – directory containing data_splits.parquet
    BERT_MODEL_NAME   – HuggingFace model identifier
    BERT_MAX_LENGTH   – maximum token length
    BERT_BATCH_SIZE   – batch size for embedding inference
    BERT_DEVICE       – 'cuda' or 'cpu'
"""

import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Mapping: (input parquet filename, text column) -> output filename, embedding column
_TEXT_FEATURES = [
    (
        "diag_history_features.parquet",
        "diag_history_text",
        "diag_history_embeddings.parquet",
        "diag_history_embedding",
    ),
    (
        "discharge_history_features.parquet",
        "discharge_history_text",
        "discharge_history_embeddings.parquet",
        "discharge_history_embedding",
    ),
    (
        "triage_features.parquet",
        "triage_text",
        "triage_embeddings.parquet",
        "triage_embedding",
    ),
    (
        "chief_complaint_features.parquet",
        "chief_complaint_text",
        "chief_complaint_embeddings.parquet",
        "chief_complaint_embedding",
    ),
    (
        "radiology_features.parquet",
        "radiology_text",
        "radiology_embeddings.parquet",
        "radiology_embedding",
    ),
]

# Per-feature max token length caps. Applied as min(BERT_MAX_LENGTH, cap) so
# short features are not padded to the global 8192-token ceiling.
_MAX_LENGTH_CAP: dict[str, int] = {
    "diag_history_text":      512,   # ICD label list — short
    "chief_complaint_text":   64,    # single phrase
    "triage_text":            256,   # structured vitals template
    "discharge_history_text": 4096,  # full clinical note — keep long
    "radiology_text":         1024,  # radiology report — medium
    # lab groups: capped at _LAB_MAX_LENGTH, applied in _worker
}
_LAB_MAX_LENGTH = 2048
_REFERENCE_LENGTH = 512   # batch size is calibrated for this sequence length


def _get_device(requested: str):
    """Return a torch device, falling back to CPU if CUDA is unavailable."""
    import torch  # type: ignore

    slurm_job = os.environ.get("SLURM_JOB_ID", "")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    if slurm_job:
        logger.info("SLURM job %s — CUDA_VISIBLE_DEVICES=%s", slurm_job, cuda_visible)

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def _get_available_gpus(max_gpus: int | None = None) -> list[str]:
    """Return list of available CUDA device strings e.g. ['cuda:0', 'cuda:1']."""
    import torch  # type: ignore

    if not torch.cuda.is_available():
        logger.warning("No CUDA GPUs available — falling back to CPU.")
        return ["cpu"]

    n = torch.cuda.device_count()
    if max_gpus is not None:
        n = min(n, max_gpus)

    devices = [f"cuda:{i}" for i in range(n)]
    for i, d in enumerate(devices):
        logger.info(
            "  GPU %d: %s — %s (capability sm_%d%d)",
            i, d,
            torch.cuda.get_device_name(i),
            *torch.cuda.get_device_capability(i),
        )
    return devices


def _effective_batch_size(base_batch_size: int, effective_max_length: int) -> int:
    """
    Scale batch size to maintain a roughly constant token budget per GPU step.
    base_batch_size is calibrated for _REFERENCE_LENGTH tokens.
    """
    scale = _REFERENCE_LENGTH / max(effective_max_length, 1)
    scaled = int(base_batch_size * scale)
    return max(1, min(scaled, base_batch_size * 8))  # cap at 8× base


def _output_is_valid(path: str, expected_rows: int, embedding_col: str) -> bool:
    """
    Return True if a completed embedding parquet exists at `path` and is usable.

    Checks:
    - File exists
    - Can be read as a parquet
    - Has the expected number of rows (matches the input feature file)
    - Contains the expected embedding column
    - No null values in the embedding column (a partial write would leave nulls)
    """
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_parquet(path)
    except Exception:
        return False
    if len(df) != expected_rows:
        return False
    if embedding_col not in df.columns:
        return False
    if df[embedding_col].isnull().any():
        return False
    return True


def _embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Return a (N, hidden_size) float32 array using mean pooling."""
    import torch  # type: ignore

    all_embeddings: list[torch.Tensor] = []   # keep on GPU until the end
    empty_indices: list[int] = []
    global_idx = 0

    n_batches = (len(texts) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(texts), batch_size), total=n_batches,
                      desc="Embedding batches", unit="batch", leave=False):
        batch_texts = texts[start: start + batch_size]
        batch_texts_safe = [t if isinstance(t, str) and t.strip() else " "
                            for t in batch_texts]
        empty_flags = [not (isinstance(t, str) and t.strip()) for t in batch_texts]

        for i, flag in enumerate(empty_flags):
            if flag:
                empty_indices.append(global_idx + i)
        global_idx += len(batch_texts)

        encoded = tokenizer(
            batch_texts_safe,
            padding="longest",       # pad to longest sequence in THIS batch only
            truncation=True,
            max_length=max_length,   # still enforce the hard ceiling
            return_tensors="pt",
        )
        encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        attention_mask = encoded["attention_mask"]
        token_embeddings = outputs.last_hidden_state          # (B, L, H)
        mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .to(token_embeddings.dtype)
        )
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)  # (B, H)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)           # (B, H)
        mean_embeddings = sum_embeddings / sum_mask                           # (B, H)

        all_embeddings.append(mean_embeddings)   # stays on GPU

    # Single GPU → CPU transfer for the entire feature
    result = torch.cat(all_embeddings, dim=0).cpu().numpy()   # (N, H)

    # Zero-out originally empty texts
    if empty_indices:
        result[empty_indices, :] = 0.0

    return result


def _build_feature_tasks(
    config: dict,
    lab_panel_config: dict,
    labs_df,      # pd.DataFrame | None
    splits_df,    # pd.DataFrame
) -> list[dict]:
    """
    Build the full list of feature task dicts (up to 18).
    Loads all input text on the main process once.
    Each dict contains everything a worker needs to embed one feature.
    """
    import yaml  # type: ignore

    features_dir = str(config["FEATURES_DIR"])
    embeddings_dir = str(config["EMBEDDINGS_DIR"])
    tasks = []

    # --- 5 text features ---
    for (input_filename, text_col, output_filename, embedding_col) in _TEXT_FEATURES:
        input_path = os.path.join(features_dir, input_filename)
        if not os.path.exists(input_path):
            logger.warning("Input not found, skipping: %s", input_path)
            continue
        df = pd.read_parquet(input_path)
        if text_col not in df.columns:
            logger.warning("Column '%s' not found in %s, skipping.", text_col, input_path)
            continue
        tasks.append({
            "kind":          "text",
            "text_col":      text_col,
            "output_path":   os.path.join(embeddings_dir, output_filename),
            "embedding_col": embedding_col,
            "texts":         [str(t) for t in df[text_col].tolist()],
            "subject_hadm":  df[["subject_id", "hadm_id"]].copy(),
        })
        logger.info("  Loaded %d texts for '%s'", len(df), text_col)

    # --- 13 lab group features ---
    if labs_df is not None and lab_panel_config:
        logger.info("Pre-grouping labs by panel (one scan for all %d groups)…",
                    len(lab_panel_config))
        itemid_to_group = {
            itemid: gname
            for gname, itemids in lab_panel_config.items()
            for itemid in itemids
        }
        labs_copy = labs_df.copy()
        labs_copy["_group"] = labs_copy["itemid"].map(itemid_to_group)
        labs_by_group = {
            gname: grp.drop(columns=["_group"])
            for gname, grp in labs_copy.groupby("_group")
            if gname in lab_panel_config
        }
        logger.info(
            "  Pre-grouped %d lab rows into %d groups",
            len(labs_df), len(labs_by_group),
        )

        for group_name in lab_panel_config:
            embedding_col = f"lab_{group_name}_embedding"
            output_path = os.path.join(
                embeddings_dir, f"lab_{group_name}_embeddings.parquet"
            )
            group_df = labs_by_group.get(group_name, pd.DataFrame())

            if not group_df.empty and "charttime" in group_df.columns:
                group_text = (
                    group_df.sort_values("charttime")
                    .groupby(["subject_id", "hadm_id"])["lab_text_line"]
                    .apply(lambda lines: "\n".join(lines))
                    .reset_index()
                    .rename(columns={"lab_text_line": "text"})
                )
            else:
                group_text = pd.DataFrame(columns=["subject_id", "hadm_id", "text"])

            group_text = splits_df.merge(
                group_text, on=["subject_id", "hadm_id"], how="left"
            )
            group_text["text"] = group_text["text"].fillna("")

            n_with_events = int((group_text["text"] != "").sum())
            logger.info(
                "  lab_%s: %d admissions (%d with events, %d zero vectors)",
                group_name, len(group_text),
                n_with_events, len(group_text) - n_with_events,
            )
            tasks.append({
                "kind":          "lab",
                "text_col":      "",
                "output_path":   output_path,
                "embedding_col": embedding_col,
                "texts":         group_text["text"].tolist(),
                "subject_hadm":  group_text[["subject_id", "hadm_id"]].copy(),
            })

    return tasks


def _worker(
    rank: int,
    device_str: str,
    feature_tasks: list[dict],
    config: dict,
    result_queue,
) -> None:
    """
    Worker process: loads BERT on `device_str`, embeds its assigned features,
    writes output parquets. One worker per GPU, spawned by torch.multiprocessing.
    """
    import torch  # type: ignore
    from transformers import AutoTokenizer, BertModel  # type: ignore
    from transformers import logging as hf_logging     # type: ignore

    hf_logging.set_verbosity_error()

    worker_logger = logging.getLogger(f"embed_features.worker{rank}")

    model_name    = str(config["BERT_MODEL_NAME"])
    max_length    = int(config["BERT_MAX_LENGTH"])
    batch_size    = int(config["BERT_BATCH_SIZE"])
    force_reembed = bool(config.get("BERT_FORCE_REEMBED", False))

    device = torch.device(device_str)
    worker_logger.info(
        "[GPU %d | %s] Loading model '%s'…", rank, device_str, model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, add_pooling_layer=False)
    model.to(device)
    model.eval()
    worker_logger.info(
        "[GPU %d | %s] Model ready. Processing %d feature(s).",
        rank, device_str, len(feature_tasks),
    )

    completed: list[str] = []
    failed: list[tuple[str, str]] = []

    for task in feature_tasks:
        output_path   = task["output_path"]
        embedding_col = task["embedding_col"]
        texts         = task["texts"]
        subject_hadm  = task["subject_hadm"]

        # --- Resume check ---
        if not force_reembed and _output_is_valid(
            output_path, expected_rows=len(texts), embedding_col=embedding_col
        ):
            worker_logger.info(
                "[GPU %d] [SKIP] %s already complete (%d rows).",
                rank, os.path.basename(output_path), len(texts),
            )
            completed.append(output_path)
            continue

        # --- Effective max_length and batch size ---
        if task["kind"] == "lab":
            effective_max_length = min(max_length, _LAB_MAX_LENGTH)
        else:
            effective_max_length = min(
                max_length,
                _MAX_LENGTH_CAP.get(task["text_col"], max_length),
            )
        effective_batch_size = _effective_batch_size(batch_size, effective_max_length)

        n_with_text = sum(1 for t in texts if isinstance(t, str) and t.strip())
        worker_logger.info(
            "[GPU %d] Embedding %s: %d texts (%d non-empty) "
            "max_length=%d batch_size=%d",
            rank, embedding_col, len(texts), n_with_text,
            effective_max_length, effective_batch_size,
        )

        try:
            embeddings = _embed_texts(
                texts, tokenizer, model, device,
                effective_max_length, effective_batch_size,
            )

            out_df = subject_hadm.copy()
            out_df[embedding_col] = list(embeddings)

            # Atomic write via temp file — safe against mid-write kill signals
            tmp_path = output_path + ".tmp"
            out_df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, output_path)

            worker_logger.info(
                "[GPU %d] Saved %s (%d rows, dim=%d)",
                rank, os.path.basename(output_path),
                len(out_df), embeddings.shape[1],
            )
            completed.append(output_path)

        except Exception as exc:  # noqa: BLE001
            worker_logger.error(
                "[GPU %d] FAILED — %s: %s", rank, embedding_col, exc
            )
            failed.append((output_path, str(exc)))

    result_queue.put({"rank": rank, "completed": completed, "failed": failed})


def run(config: dict) -> None:
    """Embed all text features using all available GPUs in parallel."""
    import torch  # type: ignore
    import yaml   # type: ignore

    required_keys = [
        "FEATURES_DIR", "EMBEDDINGS_DIR", "CLASSIFICATIONS_DIR",
        "PREPROCESSING_DIR", "BERT_MODEL_NAME", "BERT_MAX_LENGTH",
        "BERT_BATCH_SIZE", "BERT_DEVICE",
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")

    features_dir        = str(config["FEATURES_DIR"])
    embeddings_dir      = str(config["EMBEDDINGS_DIR"])
    classifications_dir = str(config["CLASSIFICATIONS_DIR"])
    preprocessing_dir   = str(config["PREPROCESSING_DIR"])
    device_str          = str(config["BERT_DEVICE"])
    max_gpus            = config.get("BERT_MAX_GPUS", None)
    if max_gpus is not None:
        max_gpus = int(max_gpus)

    os.makedirs(embeddings_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Detect GPUs
    # ------------------------------------------------------------------ #
    slurm_job = os.environ.get("SLURM_JOB_ID", "")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    if slurm_job:
        logger.info(
            "SLURM job %s — CUDA_VISIBLE_DEVICES=%s", slurm_job, cuda_visible
        )

    if device_str == "cuda" and torch.cuda.is_available():
        devices = _get_available_gpus(max_gpus)
        logger.info("Using %d GPU(s) for parallel embedding.", len(devices))
    else:
        devices = ["cpu"]
        logger.info("CPU mode.")

    n_gpus = len(devices)

    # ------------------------------------------------------------------ #
    # Load input data on main process (once)
    # ------------------------------------------------------------------ #
    lab_panel_config: dict = {}
    labs_df = None

    lab_panel_config_path = os.path.join(classifications_dir, "lab_panel_config.yaml")
    if os.path.exists(lab_panel_config_path):
        with open(lab_panel_config_path, encoding="utf-8") as fh:
            lab_panel_config = yaml.safe_load(fh)
        labs_path = os.path.join(features_dir, "labs_features.parquet")
        if os.path.exists(labs_path):
            logger.info("Loading labs_features.parquet…")
            labs_df = pd.read_parquet(labs_path)
            logger.info("  Loaded %d lab rows", len(labs_df))
        else:
            logger.warning("labs_features.parquet not found — lab embeddings skipped.")
    else:
        logger.warning("lab_panel_config.yaml not found — lab embeddings skipped.")

    splits_path = os.path.join(preprocessing_dir, "data_splits.parquet")
    splits_df = pd.read_parquet(splits_path)[["subject_id", "hadm_id"]].drop_duplicates()

    logger.info("Building feature task list…")
    all_tasks = _build_feature_tasks(config, lab_panel_config, labs_df, splits_df)
    logger.info("Total features to embed: %d across %d GPU(s)", len(all_tasks), n_gpus)

    # ------------------------------------------------------------------ #
    # Partition tasks round-robin across GPUs
    # ------------------------------------------------------------------ #
    partitions: list[list[dict]] = [[] for _ in range(n_gpus)]
    for i, task in enumerate(all_tasks):
        partitions[i % n_gpus].append(task)

    logger.info("Feature assignment:")
    for i, (device, partition) in enumerate(zip(devices, partitions)):
        logger.info(
            "  GPU %d (%s): %d features — %s",
            i, device, len(partition),
            ", ".join(t["embedding_col"] for t in partition),
        )

    # ------------------------------------------------------------------ #
    # Run workers
    # ------------------------------------------------------------------ #
    if n_gpus == 1:
        # Single device — run in current process, no spawn overhead
        import queue
        result_queue: queue.SimpleQueue = queue.SimpleQueue()
        _worker(0, devices[0], partitions[0], config, result_queue)
        results = [result_queue.get()]
    else:
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")   # required for CUDA
        result_queue_mp = ctx.Queue()

        processes = []
        for rank, (device, partition) in enumerate(zip(devices, partitions)):
            p = ctx.Process(
                target=_worker,
                args=(rank, device, partition, config, result_queue_mp),
                name=f"embed-worker-{rank}",
            )
            p.start()
            logger.info(
                "Spawned worker %d on %s (pid=%d)", rank, device, p.pid
            )
            processes.append(p)

        for p in processes:
            p.join()

        results = [result_queue_mp.get() for _ in processes]

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    total_completed = sum(len(r["completed"]) for r in results)
    total_failed    = sum(len(r["failed"])    for r in results)

    logger.info("")
    logger.info(
        "Embedding complete: %d succeeded, %d failed",
        total_completed, total_failed,
    )
    for r in sorted(results, key=lambda x: x["rank"]):
        rank = r["rank"]
        for path in r["completed"]:
            logger.info("  [GPU %d] OK   %s", rank, os.path.basename(path))
        for path, err in r["failed"]:
            logger.error("  [GPU %d] FAIL %s — %s", rank, os.path.basename(path), err)

    if total_failed > 0:
        raise RuntimeError(
            f"{total_failed} embedding task(s) failed. "
            "Check the log above. Re-run embed_job.sh — "
            "completed features will be skipped automatically."
        )


def main() -> None:
    """CLI entry point: python embed_features.py --config config/preprocessing.yaml"""
    import argparse
    import yaml  # type: ignore

    parser = argparse.ArgumentParser(description="Embed CDSS text features using BERT.")
    parser.add_argument(
        "--config",
        default="config/preprocessing.yaml",
        help="Path to preprocessing.yaml (default: config/preprocessing.yaml)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run(config)


if __name__ == "__main__":
    main()
