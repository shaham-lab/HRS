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
}


def _get_device(requested: str):
    """Return a torch device, falling back to CPU if CUDA is unavailable."""
    import torch  # type: ignore

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning(
            "CUDA requested but not available – falling back to CPU."
        )
        return torch.device("cpu")
    return torch.device("cpu")


def _effective_batch_size(base_batch_size: int, effective_max_length: int,
                           reference_length: int = 512) -> int:
    """
    Scale batch size to maintain a roughly constant token budget per GPU step.
    base_batch_size is calibrated for reference_length tokens.
    """
    scale = reference_length / max(effective_max_length, 1)
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


def run(config: dict) -> None:
    """Embed all text features and save embedding parquets."""
    required_keys = [
        "FEATURES_DIR",
        "EMBEDDINGS_DIR",
        "CLASSIFICATIONS_DIR",
        "PREPROCESSING_DIR",
        "BERT_MODEL_NAME",
        "BERT_MAX_LENGTH",
        "BERT_BATCH_SIZE",
        "BERT_DEVICE",
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")

    features_dir = str(config["FEATURES_DIR"])
    embeddings_dir = str(config["EMBEDDINGS_DIR"])
    classifications_dir = str(config["CLASSIFICATIONS_DIR"])
    preprocessing_dir = str(config["PREPROCESSING_DIR"])
    model_name = str(config["BERT_MODEL_NAME"])
    max_length = int(config["BERT_MAX_LENGTH"])
    batch_size = int(config["BERT_BATCH_SIZE"])
    device_str = str(config["BERT_DEVICE"])
    force_reembed = bool(config.get("BERT_FORCE_REEMBED", False))

    # ------------------------------------------------------------------ #
    # Determine total embedding tasks for the top-level progress bar
    # ------------------------------------------------------------------ #
    n_text_features = len([f for f in _TEXT_FEATURES
                            if os.path.exists(os.path.join(features_dir, f[0]))])

    # Peek at lab_panel_config to count lab group tasks (before loading model)
    lab_panel_config_path = os.path.join(classifications_dir, "lab_panel_config.yaml")
    lab_panel_config_loaded = False
    lab_panel_config: dict[str, list[int]] = {}
    if os.path.exists(lab_panel_config_path):
        try:
            import yaml  # type: ignore
            with open(lab_panel_config_path, encoding="utf-8") as fh:
                lab_panel_config = yaml.safe_load(fh)
            lab_panel_config_loaded = True
        except Exception:
            pass

    n_lab_groups = len(lab_panel_config) if lab_panel_config_loaded else 0
    total_tasks = n_text_features + n_lab_groups

    # ------------------------------------------------------------------ #
    # Resume status: fast existence pre-screen for logging + early exit
    # ------------------------------------------------------------------ #
    os.makedirs(embeddings_dir, exist_ok=True)

    expected_text_outputs: list[tuple[str, str]] = [
        (os.path.join(embeddings_dir, output_filename), embedding_col)
        for (_, _, output_filename, embedding_col) in _TEXT_FEATURES
    ]
    n_text_done = sum(1 for (p, _) in expected_text_outputs if os.path.exists(p))
    n_text_total = len(_TEXT_FEATURES)

    if force_reembed:
        logger.info("BERT_FORCE_REEMBED=true — per-feature resume checks disabled.")
    elif n_text_done > 0:
        logger.info(
            "Resume mode: %d / %d text feature embeddings already present — "
            "will verify and skip completed ones.",
            n_text_done, n_text_total,
        )
    else:
        logger.info("Fresh run: no existing text embedding outputs found.")

    # ------------------------------------------------------------------ #
    # Load model and tokenizer
    # ------------------------------------------------------------------ #
    try:
        from transformers import AutoTokenizer, BertModel  # type: ignore
        from transformers import logging as hf_logging  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required for embed_features.py. "
            "Install it with: pip install transformers torch"
        ) from exc

    hf_logging.set_verbosity_error()

    device = _get_device(device_str)
    logger.info("Loading BERT model: %s", model_name)
    logger.info("  Device: %s", device)
    logger.info("  Max token length: %d", max_length)
    logger.info("  Batch size: %d", batch_size)
    logger.info("  Total embedding tasks: %d (%d text features + %d lab groups)",
                total_tasks, n_text_features, n_lab_groups)
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[misc]
    model = BertModel.from_pretrained(model_name, add_pooling_layer=False)  # type: ignore[misc]
    model.to(device)
    model.eval()  # set once here; _embed_texts no longer calls model.eval()

    # ------------------------------------------------------------------ #
    # Embed each standard text feature file
    # ------------------------------------------------------------------ #
    with tqdm(total=total_tasks, desc="embed_features", unit="feature", dynamic_ncols=True) as pbar:
        for (
            input_filename,
            text_col,
            output_filename,
            embedding_col,
        ) in _TEXT_FEATURES:
            input_path = os.path.join(features_dir, input_filename)
            if not os.path.exists(input_path):
                logger.warning(
                    "Feature file not found, skipping: %s", input_path
                )
                pbar.update(1)
                continue

            pbar.set_description(f"embed_features — {embedding_col}")
            logger.info("Embedding '%s' from %s…", text_col, input_path)
            df = pd.read_parquet(input_path)
            output_path = os.path.join(embeddings_dir, output_filename)

            # ── Resume check ──────────────────────────────────────────────
            if not force_reembed and _output_is_valid(
                output_path, expected_rows=len(df), embedding_col=embedding_col
            ):
                logger.info(
                    "  [SKIP] %s already complete (%d rows) — resuming from next feature.",
                    output_filename, len(df),
                )
                pbar.update(1)
                continue
            # ──────────────────────────────────────────────────────────────

            if text_col not in df.columns:
                raise ValueError(
                    f"Expected column '{text_col}' not found in {input_path}. "
                    f"Available columns: {list(df.columns)}"
                )

            texts: list[str] = [str(t) for t in df[text_col].tolist()]
            effective_max_length = min(
                max_length,
                _MAX_LENGTH_CAP.get(text_col, max_length),
            )
            effective_batch_size = _effective_batch_size(batch_size, effective_max_length)
            logger.info(
                "  Embedding '%s': %d texts, effective max_length=%d, effective_batch_size=%d",
                text_col, len(texts), effective_max_length, effective_batch_size,
            )
            embeddings = _embed_texts(
                texts, tokenizer, model, device, effective_max_length, effective_batch_size
            )

            out_df = df[["subject_id", "hadm_id"]].copy()
            out_df[embedding_col] = list(embeddings)

            tmp_path = output_path + ".tmp"
            out_df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, output_path)
            logger.info(
                "  [%d/%d] %s: %d texts embedded (dim=%d)",
                pbar.n + 1, total_tasks, embedding_col, len(texts), embeddings.shape[1],
            )
            pbar.update(1)

        # ------------------------------------------------------------------ #
        # Embed 13 lab group features using lab_panel_config.yaml
        # ------------------------------------------------------------------ #
        if not lab_panel_config_loaded:
            logger.warning(
                "lab_panel_config.yaml not found at %s — lab group embeddings skipped. "
                "Run build_lab_panel_config.py first.",
                lab_panel_config_path,
            )
            return

        labs_path = os.path.join(features_dir, "labs_features.parquet")
        if not os.path.exists(labs_path):
            logger.warning(
                "labs_features.parquet not found at %s — lab group embeddings skipped.",
                labs_path,
            )
            return

        logger.info("Loading labs_features.parquet for lab group embeddings…")
        labs_df = pd.read_parquet(labs_path)

        splits_path = os.path.join(preprocessing_dir, "data_splits.parquet")
        if not os.path.exists(splits_path):
            logger.warning(
                "data_splits.parquet not found at %s — lab group embeddings skipped. "
                "Run create_splits.py first.",
                splits_path,
            )
            return

        splits_df = pd.read_parquet(splits_path)[["subject_id", "hadm_id"]].drop_duplicates()

        # Pre-group labs_df once so each iteration does a dict lookup instead of
        # scanning the full DataFrame (Bottleneck 6).
        logger.info("Pre-grouping labs by panel…")
        itemid_to_group = {
            itemid: group_name
            for group_name, itemids in lab_panel_config.items()
            for itemid in itemids
        }
        labs_df["_group"] = labs_df["itemid"].map(itemid_to_group)
        labs_by_group: dict[str, pd.DataFrame] = {
            group_name: grp.drop(columns=["_group"])
            for group_name, grp in labs_df.groupby("_group")
            if group_name in lab_panel_config
        }
        logger.info("  Pre-grouped %d lab rows into %d groups", len(labs_df), len(labs_by_group))

        # Cap lab group sequence length at 2048 — lab text lines are long but
        # not full clinical-note length.
        lab_max_length = min(max_length, 2048)

        for group_name, itemids in lab_panel_config.items():
            pbar.set_description(f"embed_features — lab_{group_name}_embedding")
            output_filename = f"lab_{group_name}_embeddings.parquet"
            output_path = os.path.join(embeddings_dir, output_filename)
            embedding_col = f"lab_{group_name}_embedding"

            logger.info("Embedding lab group '%s' (%d itemids)…", group_name, len(itemids))

            # Use pre-grouped DataFrame instead of re-scanning labs_df each iteration
            group_df = labs_by_group.get(group_name, pd.DataFrame())
            if group_df.empty:
                logger.debug("Lab group '%s' has no rows in labs_features.parquet — all zero vectors.", group_name)
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

            # Left-join to full admission universe so admissions with no events
            # in this group get an empty string → zero vector
            group_text = splits_df.merge(
                group_text, on=["subject_id", "hadm_id"], how="left"
            )
            group_text["text"] = group_text["text"].fillna("")

            texts_group: list[str] = group_text["text"].tolist()
            n_non_empty = int((group_text["text"] != "").sum())

            # ── Resume check ──────────────────────────────────────────────
            if not force_reembed and _output_is_valid(
                output_path, expected_rows=len(group_text), embedding_col=embedding_col
            ):
                logger.info(
                    "  [SKIP] %s already complete (%d rows) — resuming from next group.",
                    output_filename, len(group_text),
                )
                pbar.update(1)
                continue
            # ──────────────────────────────────────────────────────────────

            lab_batch_size = _effective_batch_size(batch_size, lab_max_length)
            embeddings_group = _embed_texts(
                texts_group, tokenizer, model, device, lab_max_length, lab_batch_size
            )

            out_df = group_text[["subject_id", "hadm_id"]].copy()
            out_df[embedding_col] = list(embeddings_group)
            tmp_path = output_path + ".tmp"
            out_df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, output_path)
            logger.info(
                "  [%d/%d] lab_%s: %d admissions (%d with events, %d zero vectors)",
                pbar.n + 1, total_tasks, group_name,
                len(texts_group), n_non_empty, len(texts_group) - n_non_empty,
            )
            pbar.update(1)
