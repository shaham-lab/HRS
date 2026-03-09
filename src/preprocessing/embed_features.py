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


def _embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Return a (N, hidden_size) array using mean pooling over content tokens."""
    import torch  # type: ignore

    model.eval()
    all_embeddings: list[np.ndarray] = []

    n_batches = (len(texts) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(texts), batch_size), total=n_batches,
                      desc="Embedding batches", unit="batch"):
        batch_texts = texts[start: start + batch_size]
        # Replace None/empty with a space so the tokenizer doesn't fail
        batch_texts_safe = [t if isinstance(t, str) and t.strip() else " "
                            for t in batch_texts]
        empty_flags = [
            not (isinstance(t, str) and t.strip()) for t in batch_texts
        ]

        encoded = tokenizer(
            batch_texts_safe,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        # Mean pooling over all non-padding content tokens in the final hidden layer
        attention_mask = encoded["attention_mask"]
        token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden)
        # Expand mask to match embedding dimension
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

        # Zero-out embeddings for originally empty texts
        for i, is_empty in enumerate(empty_flags):
            if is_empty:
                mean_embeddings[i] = np.zeros_like(mean_embeddings[i])

        all_embeddings.append(mean_embeddings)

    return np.vstack(all_embeddings)


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
    # Load model and tokenizer
    # ------------------------------------------------------------------ #
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required for embed_features.py. "
            "Install it with: pip install transformers torch"
        ) from exc

    device = _get_device(device_str)
    logger.info("Loading BERT model: %s", model_name)
    logger.info("  Device: %s", device)
    logger.info("  Max token length: %d", max_length)
    logger.info("  Batch size: %d", batch_size)
    logger.info("  Total embedding tasks: %d (%d text features + %d lab groups)",
                total_tasks, n_text_features, n_lab_groups)
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[misc]
    model = AutoModel.from_pretrained(model_name)  # type: ignore[misc]
    model.to(device)

    os.makedirs(embeddings_dir, exist_ok=True)

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
                continue

            pbar.set_description(f"embed_features — {embedding_col}")
            logger.info("Embedding '%s' from %s…", text_col, input_path)
            df = pd.read_parquet(input_path)

            if text_col not in df.columns:
                raise ValueError(
                    f"Expected column '{text_col}' not found in {input_path}. "
                    f"Available columns: {list(df.columns)}"
                )

            texts: list[str] = [str(t) for t in df[text_col].tolist()]
            embeddings = _embed_texts(
                texts, tokenizer, model, device, max_length, batch_size
            )

            out_df = df[["subject_id", "hadm_id"]].copy()
            out_df[embedding_col] = list(embeddings)

            output_path = os.path.join(embeddings_dir, output_filename)
            out_df.to_parquet(output_path, index=False)
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

        for group_name, itemids in lab_panel_config.items():
            pbar.set_description(f"embed_features — lab_{group_name}_embedding")
            output_filename = f"lab_{group_name}_embeddings.parquet"
            output_path = os.path.join(embeddings_dir, output_filename)
            embedding_col = f"lab_{group_name}_embedding"

            logger.info("Embedding lab group '%s' (%d itemids)…", group_name, len(itemids))

            # Filter to this group's itemids and aggregate per admission
            group_df = labs_df[labs_df["itemid"].isin(itemids)]
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
            embeddings_group = _embed_texts(
                texts_group, tokenizer, model, device, max_length, batch_size
            )

            out_df = group_text[["subject_id", "hadm_id"]].copy()
            out_df[embedding_col] = list(embeddings_group)
            out_df.to_parquet(output_path, index=False)
            logger.info(
                "  [%d/%d] lab_%s: %d admissions (%d with events, %d zero vectors)",
                pbar.n + 1, total_tasks, group_name,
                len(texts_group), n_non_empty, len(texts_group) - n_non_empty,
            )
            pbar.update(1)
