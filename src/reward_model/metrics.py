import logging
import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute ECE using equal-mass (adaptive) binning via np.percentile."""
    if len(probs) == 0:
        return float("nan")
    bin_edges = np.percentile(probs, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0
    bin_edges = np.unique(bin_edges)
    inds = np.searchsorted(bin_edges[:-1], probs, side="right") - 1
    inds = np.clip(inds, 0, len(bin_edges) - 2)

    ece = 0.0
    total = len(probs)
    for i in range(len(bin_edges) - 1):
        in_bin = inds == i
        if not np.any(in_bin):
            continue
        avg_conf = probs[in_bin].mean()
        avg_acc = labels[in_bin].mean()
        ece += (np.sum(in_bin) / total) * abs(avg_conf - avg_acc)
    return float(ece)


def _safe_metric(func, labels: np.ndarray, probs: np.ndarray) -> float:
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")
    return float(func(labels, probs))


def compute_metrics(
    logits_list: List[torch.Tensor],
    labels_list: List[torch.Tensor],
    masked: bool = False,
) -> Dict[Union[int, str], object]:
    """Compute AUROC, AUPRC, and ECE per target with dynamic NaN masking.

    Args:
        logits_list: List of T raw logit tensors.
        labels_list: List of T label tensors (float32, NaN for non-applicable).
        masked: Tag indicating whether inputs were masked; stored in returned
            dict under key ``'masked'``.

    Returns:
        ``{0: {'auroc': ..., 'auprc': ..., 'ece': ...}, 1: {...}, ...,
           'masked': masked}``.
    """
    result: Dict[Union[int, str], object] = {}

    for i, (logits, labels) in enumerate(zip(logits_list, labels_list)):
        probs = torch.sigmoid(logits).detach().cpu().view(-1).numpy()
        labels_np = labels.detach().cpu().view(-1).numpy().astype(float)
        valid_mask = ~np.isnan(labels_np)
        p_valid = probs[valid_mask]
        l_valid = labels_np[valid_mask]
        result[i] = {
            "auroc": _safe_metric(roc_auc_score, l_valid, p_valid),
            "auprc": _safe_metric(average_precision_score, l_valid, p_valid),
            "ece": _compute_ece(p_valid, l_valid),
        }
    result["masked"] = masked
    return result


def _append_metrics_row(metrics_path: Path, row: Dict, num_targets: int = 2) -> None:
    """Append one epoch's metrics to *metrics_path* (rank 0 only).

    Uses an atomic write (write to temp file, rename) so that a SLURM
    preemption mid-write cannot corrupt the Parquet file.  If the file does
    not yet exist it is created with the correct schema.

    Columns written (Architecture §9):
        epoch, wall_time_s, masking_random_pct, masking_adversarial_pct,
        masking_none_pct, loss_total,
        loss_target_<i>, auroc_target_<i>, auprc_target_<i>, ece_target_<i>
        for each i in 0..num_targets-1.

    Args:
        metrics_path: Path to ``training_metrics.parquet``.
        row: Dict mapping column name to scalar value for this epoch.
        num_targets: Number of classification targets T.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "epoch",
        "wall_time_s",
        "masking_random_pct",
        "masking_adversarial_pct",
        "masking_none_pct",
        "loss_total",
    ]
    for i in range(num_targets):
        columns += [
            f"loss_target_{i}",
            f"auroc_target_{i}",
            f"auprc_target_{i}",
            f"ece_target_{i}",
        ]

    schema_fields = [
        ("epoch", pa.int64()),
        ("wall_time_s", pa.float64()),
        ("masking_random_pct", pa.float64()),
        ("masking_adversarial_pct", pa.float64()),
        ("masking_none_pct", pa.float64()),
        ("loss_total", pa.float64()),
    ]
    for i in range(num_targets):
        schema_fields += [
            (f"loss_target_{i}", pa.float64()),
            (f"auroc_target_{i}", pa.float64()),
            (f"auprc_target_{i}", pa.float64()),
            (f"ece_target_{i}", pa.float64()),
        ]
    schema = pa.schema(schema_fields)

    new_row = {name: [row[name]] for name in columns}
    new_table = pa.table(new_row, schema=schema)

    if metrics_path.exists():
        existing = pq.read_table(metrics_path)
        table = pa.concat_tables([existing, new_table])
    else:
        table = new_table

    tmp_path = metrics_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, metrics_path)
