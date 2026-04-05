import csv
import logging
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


class MetricsLogger:
    def __init__(self, metrics_path: Path, label_names: list) -> None:
        self.csv_path = metrics_path.with_suffix(".csv")
        self.label_names = label_names

    def append_row(self, row: Dict) -> None:
        """Append one epoch's metrics to a CSV file (rank 0 only).

        Columns written:
            epoch, time(seconds), loss_total,
            loss_<label>, auroc_<label>, auprc_<label>, ece_<label>
            for each label in label_names.
        """
        columns = ["epoch", "time(seconds)", "loss_total"]
        for name in self.label_names:
            columns += [f"loss_{name}", f"auroc_{name}", f"auprc_{name}", f"ece_{name}"]

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if write_header:
                writer.writeheader()
            writer.writerow({
                k: round(row[k], 4) if isinstance(row[k], float) else row[k]
                for k in columns
            })
