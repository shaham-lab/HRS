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
from typing import cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing_utils import _check_required_keys, _load_config

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
    Never exceeds 4× base to avoid OOM on short-capped features.
    Uses float division so caps larger than _REFERENCE_LENGTH get fractional scale.
    """
    scaled = int(base_batch_size * _REFERENCE_LENGTH / max(effective_max_length, 1))
    return max(1, min(scaled, base_batch_size * 4))  # cap at 4× base


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
    splits_df,    # pd.DataFrame — already filtered to the current slice
) -> list[dict]:
    """
    Build the full list of feature task dicts (up to 18).
    Loads all input text on the main process once, filtered to splits_df rows.
    Each dict contains everything a worker needs to embed one feature.
    """
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
        # Filter to current slice's hadm_ids
        df = splits_df.merge(df[["subject_id", "hadm_id", text_col]],
                             on=["subject_id", "hadm_id"], how="left")
        df[text_col] = df[text_col].fillna("")
        tasks.append({
            "kind":          "text",
            "text_col":      text_col,
            "output_path":   os.path.join(embeddings_dir, output_filename),
            "embedding_col": embedding_col,
            "texts":         [str(t) for t in df[text_col].tolist()],
            "subject_hadm":  df[["subject_id", "hadm_id"]].copy().reset_index(drop=True),
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
                "subject_hadm":  group_text[["subject_id", "hadm_id"]].copy().reset_index(drop=True),
            })

    return tasks


def _worker(
    rank: int,
    device_str: str,
    feature_tasks: list[dict],
    config: dict,
    result_queue,
    slice_index: int = 0,
    n_workers: int = 1,
) -> None:
    """
    Worker process: loads BERT on `device_str`, embeds its assigned features,
    writes output parquets with 3-level resume and fastparquet checkpointing.
    One worker per GPU, spawned by torch.multiprocessing.

    When n_workers > 1 (multi-GPU parallel run) each worker writes to a
    per-worker temporary parquet: ``output_path + f".worker{rank}"``.
    The main process merges the per-worker files after all workers complete.
    When n_workers == 1 (single-GPU / CPU) the worker writes directly to
    ``output_path`` — no temporary files are created.

    Resume levels:
      1. Feature-level: skip if all slice hadm_ids already present in output.
      2. Record-level: filter to only rows whose hadm_id is missing from output.
      3. Checkpoint: append every BERT_CHECKPOINT_INTERVAL rows via fastparquet.
    """
    import torch  # type: ignore
    import fastparquet as fp  # type: ignore
    from transformers import AutoTokenizer, BertModel  # type: ignore
    from transformers import logging as hf_logging     # type: ignore

    hf_logging.set_verbosity_error()

    worker_logger = logging.getLogger(f"embed_features.worker{rank}")

    model_name          = str(config["BERT_MODEL_NAME"])
    max_length          = int(config["BERT_MAX_LENGTH"])
    batch_size          = int(config["BERT_BATCH_SIZE"])
    force_reembed       = bool(config.get("BERT_FORCE_REEMBED", False))
    checkpoint_interval = int(config.get("BERT_CHECKPOINT_INTERVAL", 10000))

    device = torch.device(device_str)
    worker_logger.info(
        "[GPU %d | %s] Loading model '%s'…", rank, device_str, model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)              # type: ignore[return-value]
    model: BertModel = BertModel.from_pretrained(model_name, add_pooling_layer=False)  # type: ignore[assignment]
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
        texts         = task["texts"]          # list[str], aligned with subject_hadm
        subject_hadm  = task["subject_hadm"]   # pd.DataFrame with subject_id, hadm_id

        # Multi-GPU: write to per-worker temp; single-GPU: write directly.
        write_path = output_path + f".worker{rank}" if n_workers > 1 else output_path

        slice_hadm_ids = set(subject_hadm["hadm_id"].tolist())

        # Fix 6 — cleanup stale .tmp on force-reembed
        if force_reembed:
            for stale in (output_path + ".tmp", write_path + ".tmp", write_path):
                if stale != output_path and os.path.exists(stale):
                    os.remove(stale)
                    worker_logger.info(
                        "[GPU %d] FORCE_REEMBED: removed stale file %s",
                        rank, os.path.basename(stale),
                    )

        # --- Effective max_length and batch size ---
        if task["kind"] == "lab":
            effective_max_length = min(max_length, _LAB_MAX_LENGTH)
        else:
            effective_max_length = min(
                max_length,
                _MAX_LENGTH_CAP.get(task["text_col"], max_length),
            )
        effective_batch_size = _effective_batch_size(batch_size, effective_max_length)

        # ------------------------------------------------------------------ #
        # Level 1 — Feature-level resume: all slice rows already written?    #
        # ------------------------------------------------------------------ #
        already_done: set = set()
        if os.path.exists(output_path) and not force_reembed:
            try:
                existing = pd.read_parquet(output_path, columns=["hadm_id"])
                already_done = set(existing["hadm_id"].tolist()) & slice_hadm_ids
            except Exception:  # noqa: BLE001  # noinspection PyBroadException
                already_done = set()

        pending_hadm_ids = slice_hadm_ids - already_done

        if not pending_hadm_ids:
            worker_logger.info(
                "[GPU %d] [SKIP slice=%d feature=%s] all %d rows already present.",
                rank, slice_index, embedding_col, len(slice_hadm_ids),
            )
            # Clean up any stale worker temp left from a killed merge step.
            if n_workers > 1 and os.path.exists(write_path):
                try:
                    os.remove(write_path)
                except Exception:  # noqa: BLE001  # noinspection PyBroadException
                    pass
            completed.append(output_path)
            continue

        if already_done:
            worker_logger.info(
                "[GPU %d] Resuming %s: %d/%d rows pending.",
                rank, embedding_col, len(pending_hadm_ids), len(slice_hadm_ids),
            )

        # ------------------------------------------------------------------ #
        # Per-worker temp: handle stale file from a previously killed run    #
        # ------------------------------------------------------------------ #
        # Spec: "If a job is killed mid-merge, the per-worker temps are still #
        # present on restart and the merge is re-run."                        #
        # Spec: "If a job is killed mid-embedding, the incomplete per-worker  #
        # temps are detected (row count < expected) and that worker re-embeds #
        # from scratch for that feature."                                     #
        if n_workers > 1 and os.path.exists(write_path) and not force_reembed:
            try:
                wp_df = pd.read_parquet(write_path, columns=["hadm_id"])
                wp_done = set(wp_df["hadm_id"].tolist())
                if slice_hadm_ids.issubset(wp_done):
                    # worker temp is complete — merge was killed; skip re-embedding.
                    worker_logger.info(
                        "[GPU %d] Worker temp complete for %s — "
                        "skipping re-embed (merge will combine it)",
                        rank, embedding_col,
                    )
                    completed.append(output_path)
                    continue
                else:
                    # worker temp is incomplete from a killed embedding run.
                    os.remove(write_path)
                    worker_logger.info(
                        "[GPU %d] Removed incomplete worker temp for %s "
                        "(%d/%d expected rows) — re-embedding from scratch",
                        rank, embedding_col,
                        len(wp_done & slice_hadm_ids), len(slice_hadm_ids),
                    )
            except Exception:  # noqa: BLE001  # noinspection PyBroadException
                # Unreadable worker temp — delete and re-embed from scratch.
                if os.path.exists(write_path):
                    os.remove(write_path)
                    worker_logger.info(
                        "[GPU %d] Removed unreadable worker temp for %s "
                        "— re-embedding from scratch",
                        rank, embedding_col,
                    )

        # ------------------------------------------------------------------ #
        # Level 2 — Record-level resume: filter to only pending rows          #
        # ------------------------------------------------------------------ #
        pending_mask = subject_hadm["hadm_id"].isin(pending_hadm_ids).values
        subject_hadm_pending = subject_hadm[pending_mask].reset_index(drop=True)
        texts_pending = [t for t, m in zip(texts, pending_mask) if m]

        n_with_text = sum(1 for t in texts_pending if isinstance(t, str) and t.strip())
        worker_logger.info(
            "[GPU %d] Embedding %s: %d texts (%d non-empty) "
            "max_length=%d batch_size=%d",
            rank, embedding_col, len(texts_pending), n_with_text,
            effective_max_length, effective_batch_size,
        )

        try:
            # ------------------------------------------------------------------ #
            # Level 3 — Checkpoint: embed in intervals, append via fastparquet   #
            # ------------------------------------------------------------------ #
            n_total = len(texts_pending)
            is_first_write = not os.path.exists(write_path)
            rows_written = 0

            for start in range(0, n_total, checkpoint_interval):
                end = min(start + checkpoint_interval, n_total)
                batch_texts = texts_pending[start:end]
                batch_sh = subject_hadm_pending.iloc[start:end].copy()

                embeddings = _embed_texts(
                    batch_texts, tokenizer, model, device,
                    effective_max_length, effective_batch_size,
                )

                batch_sh[embedding_col] = [
                    np.asarray(emb, dtype=np.float32)
                    for emb in embeddings
                ]

                # Atomic first write; subsequent writes use fastparquet append.
                if is_first_write:
                    tmp_path = write_path + ".tmp"
                    fp.write(tmp_path, batch_sh, compression="snappy")
                    os.replace(tmp_path, write_path)
                    is_first_write = False
                else:
                    fp.write(write_path, batch_sh, compression="snappy", append=True)

                rows_written += len(batch_sh)
                worker_logger.info(
                    "[GPU %d] Checkpoint: wrote rows %d–%d for %s (%d total so far)",
                    rank, start, end - 1, embedding_col, rows_written,
                )

            worker_logger.info(
                "[GPU %d] Saved %s (%d new rows, dim=%d)",
                rank, os.path.basename(output_path),
                rows_written, 768,
            )
            completed.append(output_path)

        except Exception as exc:  # noqa: BLE001
            worker_logger.error(
                "[GPU %d] FAILED — %s: %s", rank, embedding_col, exc
            )
            failed.append((output_path, str(exc)))

    result_queue.put({"rank": rank, "completed": completed, "failed": failed})


def run(config: dict, slice_index: int | None = None) -> None:
    """Embed text features for one admission slice using multi-GPU parallel processing.

    Args:
        config: Preprocessing configuration dict.
        slice_index: 0-based index of the slice to process. If None, reads
            ``config["BERT_SLICE_INDEX"]`` (default 0).
    """
    import math
    import torch  # type: ignore
    import yaml   # type: ignore

    _check_required_keys(config, [
        "FEATURES_DIR", "EMBEDDINGS_DIR", "CLASSIFICATIONS_DIR",
        "PREPROCESSING_DIR", "BERT_MODEL_NAME", "BERT_MAX_LENGTH",
        "BERT_BATCH_SIZE", "BERT_DEVICE",
    ])

    features_dir        = str(config["FEATURES_DIR"])
    embeddings_dir      = str(config["EMBEDDINGS_DIR"])
    classifications_dir = str(config["CLASSIFICATIONS_DIR"])
    preprocessing_dir   = str(config["PREPROCESSING_DIR"])
    device_str          = str(config["BERT_DEVICE"])
    max_gpus_cfg        = config.get("BERT_MAX_GPUS")
    max_gpus: int | None = int(cast(int, max_gpus_cfg)) if max_gpus_cfg is not None else None

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
    # Admission-slice computation
    # ------------------------------------------------------------------ #
    resolved_slice_index: int = (
        int(config.get("BERT_SLICE_INDEX", 0)) if slice_index is None else slice_index
    )

    splits_path = os.path.join(preprocessing_dir, "data_splits.parquet")
    splits_full = pd.read_parquet(splits_path)[["subject_id", "hadm_id"]].drop_duplicates()
    all_hadm_ids = sorted(splits_full["hadm_id"].unique().tolist())
    total_admissions = len(all_hadm_ids)

    slice_size_per_gpu = int(config.get("BERT_SLICE_SIZE_PER_GPU", 20000))
    per_job = slice_size_per_gpu * n_gpus
    n_slices = math.ceil(total_admissions / per_job)

    logger.info(
        "Total admissions: %d | Slice size per job: %d (%d per GPU × %d GPU(s)) "
        "| Total slices: %d",
        total_admissions, per_job, slice_size_per_gpu, n_gpus, n_slices,
    )

    if resolved_slice_index >= n_slices:
        logger.info(
            "slice_index=%d >= n_slices=%d — nothing to do.", resolved_slice_index, n_slices
        )
        return

    slice_start = resolved_slice_index * per_job
    slice_end   = min(slice_start + per_job, total_admissions)
    slice_hadm_ids = set(all_hadm_ids[slice_start:slice_end])

    logger.info(
        "Processing slice %d/%d: hadm_ids[%d:%d] (%d admissions)",
        resolved_slice_index, n_slices - 1, slice_start, slice_end, len(slice_hadm_ids),
    )

    # Filter splits to current slice
    splits_df = splits_full[splits_full["hadm_id"].isin(slice_hadm_ids)].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Load lab data filtered to slice
    # ------------------------------------------------------------------ #
    lab_panel_config: dict = {}
    labs_df = None

    lab_panel_config_path = os.path.join(classifications_dir, "lab_panel_config.yaml")
    if os.path.exists(lab_panel_config_path):
        with open(lab_panel_config_path, encoding="utf-8") as fh:
            lab_panel_config = yaml.safe_load(fh)
        labs_path = os.path.join(features_dir, "labs_features.parquet")
        if os.path.exists(labs_path):
            logger.info("Loading labs_features.parquet (filtering to slice)…")
            labs_df = pd.read_parquet(labs_path)
            labs_df = labs_df[labs_df["hadm_id"].isin(slice_hadm_ids)].reset_index(drop=True)
            logger.info("  Loaded %d lab rows for this slice", len(labs_df))
        else:
            logger.warning("labs_features.parquet not found — lab embeddings skipped.")
    else:
        logger.warning("lab_panel_config.yaml not found — lab embeddings skipped.")

    # ------------------------------------------------------------------ #
    # Build feature tasks (filtered to this slice's admissions)
    # ------------------------------------------------------------------ #
    logger.info("Building feature task list for slice %d…", resolved_slice_index)
    all_tasks = _build_feature_tasks(config, lab_panel_config, labs_df, splits_df)
    logger.info(
        "Pipeline plan — %d feature(s) to embed across %d GPU(s):",
        len(all_tasks), n_gpus,
    )
    for i, t in enumerate(all_tasks, 1):
        logger.info("  %2d. %s (%d texts)", i, t["embedding_col"], len(t["texts"]))

    # ------------------------------------------------------------------ #
    # Add max_length to each task (used for cost estimates below)
    # ------------------------------------------------------------------ #
    global_max_length = int(config.get("BERT_MAX_LENGTH", 8192))
    for task in all_tasks:
        if task["kind"] == "lab":
            task["max_length"] = min(global_max_length, _LAB_MAX_LENGTH)
        else:
            task["max_length"] = min(
                global_max_length,
                _MAX_LENGTH_CAP.get(task["text_col"], global_max_length),
            )

    # ------------------------------------------------------------------ #
    # Step A — Split slice admissions evenly between GPU workers
    # ------------------------------------------------------------------ #
    # Each worker processes ALL 18 features for its own subset of admissions,
    # cutting per-worker memory usage to ~1/n_gpus of the full slice.
    # Workers run sequentially below (see Step C) to prevent concurrent writes
    # to the shared output parquets.
    all_hadm_in_slice = sorted(slice_hadm_ids)   # deterministic order
    n_hadm = len(all_hadm_in_slice)
    gpu_hadm_ids: list[set] = []
    for g in range(n_gpus):
        start_g = g * n_hadm // n_gpus
        end_g   = (g + 1) * n_hadm // n_gpus
        gpu_hadm_ids.append(set(all_hadm_in_slice[start_g:end_g]))

    logger.info(
        "Slice %d split: %s",
        resolved_slice_index,
        " | ".join(
            f"GPU {g}: {len(ids)} admissions" for g, ids in enumerate(gpu_hadm_ids)
        ),
    )

    # ------------------------------------------------------------------ #
    # Step B — Build per-GPU task lists
    # ------------------------------------------------------------------ #
    # All 18 features for each worker, but texts/subject_hadm filtered to
    # that worker's hadm_id subset; sorted by cost descending within each
    # worker (LPT ordering for better progress visibility).
    partitions: list[list[dict]] = []
    for g in range(n_gpus):
        worker_hadm_ids = gpu_hadm_ids[g]
        worker_tasks = []
        for task in all_tasks:
            mask = task["subject_hadm"]["hadm_id"].isin(worker_hadm_ids).values
            worker_sh    = task["subject_hadm"][mask].reset_index(drop=True)
            worker_texts = [t for t, m in zip(task["texts"], mask) if m]
            worker_tasks.append({
                **task,
                "texts":        worker_texts,
                "subject_hadm": worker_sh,
            })
        # LPT ordering within worker: most expensive features first
        worker_tasks.sort(
            key=lambda _task: len(_task["texts"]) * _task["max_length"], reverse=True
        )
        partitions.append(worker_tasks)

    logger.info("Worker task plans (LPT-ordered within each worker):")
    for g, (device, worker_tasks) in enumerate(zip(devices, partitions)):
        total_cost = sum(len(t["texts"]) * t["max_length"] for t in worker_tasks)
        logger.info(
            "  GPU %d (%s): %d admissions, %d features, cost %d — %s",
            g, device, len(gpu_hadm_ids[g]), len(worker_tasks), total_cost,
            ", ".join(t["embedding_col"] for t in worker_tasks),
        )

    # ------------------------------------------------------------------ #
    # Step C — Run workers (parallel for multi-GPU, in-process for single)
    # ------------------------------------------------------------------ #
    # Multi-GPU: each worker writes to output_path + f".worker{rank}".
    # The main process merges the per-worker files in Step D.
    # Single-GPU / CPU: worker runs in-process and writes directly to
    # output_path — no per-worker temp files are created.
    import queue
    result_queue: queue.SimpleQueue = queue.SimpleQueue()

    if n_gpus == 1 or not torch.cuda.is_available():
        # Single device or CPU-only: run directly in this process, no temp files.
        _worker(
            0, devices[0], partitions[0], config, result_queue,
            resolved_slice_index, n_workers=1,
        )
        results = [result_queue.get()]
    else:
        import torch.multiprocessing as mp
        ctx = cast(mp.SpawnContext, mp.get_context("spawn"))   # required for CUDA
        result_queue_mp = ctx.Queue()

        processes = []
        for rank, (device, partition) in enumerate(zip(devices, partitions)):
            p = ctx.Process(
                target=_worker,
                args=(rank, device, partition, config, result_queue_mp,
                      resolved_slice_index, n_gpus),
                name=f"embed-worker-{rank}",
            )
            p.start()
            logger.info(
                "Spawned worker %d on %s (pid=%d) — %d admissions, %d features",
                rank, device, p.pid, len(gpu_hadm_ids[rank]), len(partition),
            )
            processes.append(p)

        # Wait for all workers to finish
        for p in processes:
            p.join()
            logger.info(
                "Worker %s (pid=%d) finished with exit code %d",
                p.name, p.pid, p.exitcode,
            )

        # Drain the result queue
        results = []
        while not result_queue_mp.empty():
            results.append(result_queue_mp.get())
        results.sort(key=lambda r: r["rank"])

        # ------------------------------------------------------------------ #
        # Step D — Merge per-worker parquets into final output parquets       #
        # (multi-GPU only — single-GPU writes directly, no merge needed)      #
        # ------------------------------------------------------------------ #
        logger.info("Merging per-worker parquets into final outputs…")

        all_output_paths: set[str] = set()
        for r in results:
            for path in r["completed"]:
                all_output_paths.add(path)

        for output_path in sorted(all_output_paths):
            worker_paths = [
                output_path + f".worker{rank}"
                for rank in range(n_gpus)
                if os.path.exists(output_path + f".worker{rank}")
            ]

            if not worker_paths:
                logger.warning(
                    "No worker files found for %s — skipping merge",
                    os.path.basename(output_path),
                )
                continue

            if len(worker_paths) == 1:
                # Only one worker file present (e.g. other worker failed).
                # Count rows before rename so we can log without re-reading.
                n_rows = len(pd.read_parquet(worker_paths[0], columns=["hadm_id"]))
                tmp_path = output_path + ".tmp"
                os.replace(worker_paths[0], tmp_path)
                os.replace(tmp_path, output_path)
            else:
                # Merge all worker parquets; track row count from in-memory df.
                dfs = [pd.read_parquet(wp) for wp in worker_paths]
                merged = pd.concat(dfs, ignore_index=True)
                n_rows = len(merged)
                tmp_path = output_path + ".tmp"
                merged.to_parquet(tmp_path, index=False)
                os.replace(tmp_path, output_path)
                # Clean up worker files
                for wp in worker_paths:
                    os.remove(wp)

            logger.info(
                "  Merged %d worker file(s) → %s (%d rows)",
                len(worker_paths),
                os.path.basename(output_path),
                n_rows,
            )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    total_completed = sum(len(r["completed"]) for r in results)
    total_failed    = sum(len(r["failed"])    for r in results)

    logger.info("")
    logger.info(
        "Slice %d complete: %d succeeded, %d failed",
        resolved_slice_index, total_completed, total_failed,
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
    """
    CLI entry point for running embed_features standalone.

    Usage:
        python src/preprocessing/embed_features.py \
            --config config/preprocessing.yaml \
            --slice-index 0

    This allows embed_features to be submitted as a dedicated SLURM job
    (embed_job.sh) independently of run_pipeline.py.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Embed CDSS text features using BERT.",
    )
    parser.add_argument(
        "--config",
        default="config/preprocessing.yaml",
        help="Path to preprocessing.yaml (default: config/preprocessing.yaml)",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        dest="slice_index",
        help=(
            "0-based index of the admission slice to process. "
            "Defaults to BERT_SLICE_INDEX in the config (usually 0). "
            "submit_all.sh passes this automatically for each SLURM job."
        ),
    )
    args = parser.parse_args()

    # Set up logging in the same format as run_pipeline.py
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = _load_config(args.config)
    logger.info("Loaded configuration from %s", args.config)
    run(config, slice_index=args.slice_index)


if __name__ == "__main__":
    main()
