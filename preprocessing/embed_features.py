"""
embed_features.py – BERT sentence embeddings for all text features.

Loads text parquets from FEATURES_DIR, embeds each text column using
the [CLS] token of the configured BERT model, and saves embedding parquets
to EMBEDDINGS_DIR.

Device selection:
  - Uses BERT_DEVICE from config.
  - Falls back to CPU with a warning if CUDA is requested but unavailable.

Empty / null text → zero vector of the same embedding dimension.

Expected config keys:
    FEATURES_DIR      – directory containing raw text feature parquets
    EMBEDDINGS_DIR    – output directory for embedding parquets
    BERT_MODEL_NAME   – HuggingFace model identifier
    BERT_MAX_LENGTH   – maximum token length
    BERT_BATCH_SIZE   – batch size for embedding inference
    BERT_DEVICE       – 'cuda' or 'cpu'
"""

import logging
import os

import numpy as np
import pandas as pd

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
    """Return a (N, hidden_size) array of [CLS] embeddings."""
    import torch  # type: ignore

    model.eval()
    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
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

        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Zero-out embeddings for originally empty texts
        for i, is_empty in enumerate(empty_flags):
            if is_empty:
                cls_embeddings[i] = np.zeros_like(cls_embeddings[i])

        all_embeddings.append(cls_embeddings)

        if (start // batch_size + 1) % 10 == 0:
            logger.info(
                "  Embedded %d / %d texts…", start + len(batch_texts), len(texts)
            )

    return np.vstack(all_embeddings)


def run(config: dict) -> None:
    """Embed all text features and save embedding parquets."""
    required_keys = [
        "FEATURES_DIR",
        "EMBEDDINGS_DIR",
        "BERT_MODEL_NAME",
        "BERT_MAX_LENGTH",
        "BERT_BATCH_SIZE",
        "BERT_DEVICE",
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")

    features_dir = config["FEATURES_DIR"]
    embeddings_dir = config["EMBEDDINGS_DIR"]
    model_name = config["BERT_MODEL_NAME"]
    max_length = int(config["BERT_MAX_LENGTH"])
    batch_size = int(config["BERT_BATCH_SIZE"])
    device_str = config["BERT_DEVICE"]

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
    logger.info("Loading BERT model '%s' on device '%s'…", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    os.makedirs(embeddings_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Embed each text feature file
    # ------------------------------------------------------------------ #
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

        logger.info("Embedding '%s' from %s…", text_col, input_path)
        df = pd.read_parquet(input_path)

        if text_col not in df.columns:
            raise ValueError(
                f"Expected column '{text_col}' not found in {input_path}. "
                f"Available columns: {list(df.columns)}"
            )

        texts = df[text_col].tolist()
        embeddings = _embed_texts(
            texts, tokenizer, model, device, max_length, batch_size
        )

        out_df = df[["subject_id", "hadm_id"]].copy()
        out_df[embedding_col] = list(embeddings)

        output_path = os.path.join(embeddings_dir, output_filename)
        out_df.to_parquet(output_path, index=False)
        logger.info(
            "Saved embeddings to %s  (shape=%s, embedding_dim=%d)",
            output_path, out_df.shape, embeddings.shape[1],
        )
