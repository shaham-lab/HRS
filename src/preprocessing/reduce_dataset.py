import argparse
import json
import logging
import os
import pickle
import sys
from typing import Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA, TruncatedSVD

from preprocessing_utils import _load_config


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _effective_components(target_dim: int, fit_rows: int, feature_dim: int, method: str) -> int:
    """Clamp n_components to valid bounds for the chosen method."""
    max_dim = feature_dim - 1 if method == "svd" and feature_dim > 1 else feature_dim
    if fit_rows > 0:
        max_dim = min(max_dim, fit_rows)
    return max(1, min(target_dim, max_dim))


def run(config: Dict) -> None:
    reduction_enabled = bool(config.get("REDUCTION_ENABLED", False))
    if not reduction_enabled:
        logger.info("REDUCTION_ENABLED is false — skipping reduce_dataset.")
        return

    method = config.get("REDUCTION_METHOD", "svd").lower()
    if method not in {"svd", "pca_nonzero"}:
        raise ValueError(f"Unsupported REDUCTION_METHOD: {method}")
    target_dim = int(config.get("REDUCED_EMBEDDING_DIM", 128))

    classifications_dir = config.get("CLASSIFICATIONS_DIR")
    if not classifications_dir:
        raise ValueError("CLASSIFICATIONS_DIR missing from configuration")
    final_path = os.path.join(classifications_dir, "final_cdss_dataset.parquet")
    if not os.path.exists(final_path):
        raise FileNotFoundError(f"final_cdss_dataset.parquet not found at {final_path}")

    output_dir = config.get("REDUCTION_OUTPUT_DIR")
    if not output_dir:
        raise ValueError("REDUCTION_OUTPUT_DIR missing from configuration")
    os.makedirs(output_dir, exist_ok=True)
    reduced_path = os.path.join(output_dir, "reduced_cdss_dataset.parquet")
    transforms_path = os.path.join(output_dir, "fitted_transforms.pkl")
    variance_path = os.path.join(output_dir, "variance_stats.json")

    logger.info("Loading split column to build training mask…")
    pf = pq.ParquetFile(final_path)
    split_col = pf.read(columns=["split"]).column(0).combine_chunks()
    split_np = split_col.to_numpy()
    is_train = split_np == "train"
    n_rows = len(is_train)
    logger.info("Rows: %d (train=%d)", n_rows, int(is_train.sum()))

    arrays: List[pa.Array] = []
    names: List[str] = []
    fitted_transforms: Dict[str, object] = {}
    variance_stats: Dict[str, float] = {}

    for field in pf.schema:
        col_name = field.name
        logger.info("Processing column: %s", col_name)
        if not col_name.endswith("_embedding"):
            col_table = pf.read(columns=[col_name])
            arrays.append(col_table.column(0).combine_chunks())
            names.append(col_name)
            continue

        # Embedding column
        col_table = pf.read(columns=[col_name])
        emb_col = col_table.column(0).combine_chunks()
        # Convert to numpy (list of lists) then stack
        X = np.stack(emb_col.to_numpy()).astype(np.float32)

        nonzero_mask = np.linalg.norm(X, axis=1) > 1e-6
        fit_mask = is_train & nonzero_mask
        fit_rows = int(fit_mask.sum())
        feature_dim = X.shape[1]
        n_components = _effective_components(target_dim, fit_rows, feature_dim, method)

        model = PCA(n_components=n_components) if method == "pca_nonzero" else TruncatedSVD(n_components=n_components)

        if fit_rows == 0:
            logger.warning("No non-zero train rows for %s; leaving zeros.", col_name)
            fitted_transforms[col_name] = None
            variance_stats[col_name] = 0.0
            X_reduced = np.zeros((n_rows, target_dim), dtype=np.float32)
        else:
            logger.info(
                "Fitting %s with n_components=%d on %d rows",
                method,
                n_components,
                fit_rows,
            )
            model.fit(X[fit_mask])
            X_reduced = np.zeros((n_rows, target_dim), dtype=np.float32)
            if nonzero_mask.any():
                transformed = model.transform(X[nonzero_mask]).astype(np.float32)
                X_reduced[nonzero_mask, : transformed.shape[1]] = transformed
            fitted_transforms[col_name] = model
            variance_stats[col_name] = float(getattr(model, "explained_variance_ratio_", np.array([])).sum())

        flat = X_reduced.reshape(-1)
        embedding_array = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), target_dim)
        arrays.append(embedding_array)
        names.append(col_name)

        del X
        del X_reduced

    logger.info("Writing reduced dataset to %s", reduced_path)
    table = pa.Table.from_arrays(arrays, names=names)
    pq.write_table(table, reduced_path, compression="snappy")

    logger.info("Writing fitted transforms to %s", transforms_path)
    with open(transforms_path, "wb") as f:
        pickle.dump(fitted_transforms, f)

    logger.info("Writing variance stats to %s", variance_path)
    with open(variance_path, "w", encoding="utf-8") as f:
        json.dump(variance_stats, f, indent=2)

    logger.info("reduce_dataset complete.")


def main():
    _setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "preprocessing.yaml"),
        help="Path to preprocessing config",
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
