import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute ECE using equal-mass (adaptive) binning via np.percentile."""
    if len(probs) == 0:
        return float("nan")
    percentiles = np.linspace(0.0, 100.0, n_bins + 1)
    bin_edges = np.percentile(probs, percentiles)
    # Deduplicate edges to avoid empty bins from flat probability regions.
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return float("nan")

    ece = 0.0
    total = len(probs)
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        # Close the rightmost bin on both sides.
        if i < len(bin_edges) - 2:
            in_bin = (probs >= lower) & (probs < upper)
        else:
            in_bin = (probs >= lower) & (probs <= upper)
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


def compute_loss(
    logits_list: List[torch.Tensor],
    labels_list: List[torch.Tensor],
    pos_weights: List[float],
    loss_weights: List[float],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Compute total weighted loss over T targets with dynamic NaN masking.

    Args:
        logits_list: List of T raw logit tensors, each shape ``(N, 1)`` or ``(N,)``.
        labels_list: List of T label tensors, each shape ``(N,)``; float32 with
            NaN where a target is not applicable for a sample.
        pos_weights: List of T positive class weights.
        loss_weights: List of T normalised loss weights (must sum to 1.0).

    Returns:
        ``(total_loss, [loss_0, loss_1, ...])``.  Component losses are scalar
        tensors on the same device as their logits.
    """
    device = logits_list[0].device
    component_losses: List[torch.Tensor] = []
    total_loss = torch.tensor(0.0, device=device)

    for i, (logits, labels, pw, lw) in enumerate(
        zip(logits_list, labels_list, pos_weights, loss_weights)
    ):
        valid_mask = ~torch.isnan(labels)
        if not valid_mask.any():
            loss_i = torch.tensor(0.0, device=device)
        else:
            pw_tensor = torch.tensor(pw, device=device)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
            loss_i = loss_fn(
                logits.view(-1)[valid_mask],
                labels.view(-1)[valid_mask].float(),
            )
        component_losses.append(loss_i)
        total_loss = total_loss + lw * loss_i

    return total_loss, component_losses


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
    result: Dict[Union[int, str], object] = {"masked": masked}

    for i, (logits, labels) in enumerate(zip(logits_list, labels_list)):
        probs_np = torch.sigmoid(logits).detach().cpu().view(-1).numpy()
        labels_np = labels.detach().cpu().view(-1).numpy().astype(float)

        valid_mask = ~np.isnan(labels_np)
        probs_valid = probs_np[valid_mask]
        labels_valid = labels_np[valid_mask]

        if len(labels_valid) == 0:
            result[i] = {"auroc": float("nan"), "auprc": float("nan"), "ece": float("nan")}
            continue

        result[i] = {
            "auroc": _safe_metric(roc_auc_score, labels_valid, probs_valid),
            "auprc": _safe_metric(average_precision_score, labels_valid, probs_valid),
            "ece": _compute_ece(probs_valid, labels_valid),
        }

    return result
