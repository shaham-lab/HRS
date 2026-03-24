import logging
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(probs, bins) - 1
    ece = 0.0
    total = len(probs)
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask):
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += (np.sum(mask) / total) * abs(avg_conf - avg_acc)
    return float(ece)


def _safe_metric(func, labels: np.ndarray, probs: np.ndarray) -> float:
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")
    return float(func(labels, probs))


def compute_loss(
    logits_y1: torch.Tensor,
    logits_y2: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    pos_weight_y1: float,
    pos_weight_y2: float,
    w1: float,
    w2: float,
):
    pos_w1_tensor = torch.tensor(pos_weight_y1, device=logits_y1.device)
    loss_y1_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w1_tensor)
    loss_y1 = loss_y1_fn(logits_y1.view(-1), y1.view(-1).float())

    survivor_mask = ~torch.isnan(y2)
    if not survivor_mask.any():
        loss_y2 = torch.tensor(0.0, device=logits_y2.device)
    else:
        pos_w2_tensor = torch.tensor(pos_weight_y2, device=logits_y2.device)
        loss_y2_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w2_tensor)
        loss_y2 = loss_y2_fn(logits_y2.view(-1)[survivor_mask], y2.view(-1)[survivor_mask].float())

    total_loss = w1 * loss_y1 + w2 * loss_y2
    return total_loss, loss_y1, loss_y2


def compute_metrics(
    logits_y1: torch.Tensor,
    logits_y2: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
) -> Dict[str, float]:
    probs_y1 = torch.sigmoid(logits_y1).detach().cpu().view(-1).numpy()
    probs_y2 = torch.sigmoid(logits_y2).detach().cpu().view(-1).numpy()

    labels_y1 = y1.detach().cpu().view(-1).numpy().astype(float)
    labels_y2 = y2.detach().cpu().view(-1).numpy().astype(float)

    survivor_mask = ~np.isnan(labels_y2)
    y2_probs_survivors = probs_y2[survivor_mask]
    y2_labels_survivors = labels_y2[survivor_mask]

    metrics = {
        "auroc_y1": _safe_metric(roc_auc_score, labels_y1, probs_y1),
        "auprc_y1": _safe_metric(average_precision_score, labels_y1, probs_y1),
        "ece_y1": _compute_ece(probs_y1, labels_y1),
        "auroc_y2": float("nan"),
        "auprc_y2": float("nan"),
        "ece_y2": float("nan"),
    }

    if y2_probs_survivors.size > 0 and np.unique(y2_labels_survivors).size >= 2:
        metrics.update(
            {
                "auroc_y2": _safe_metric(roc_auc_score, y2_labels_survivors, y2_probs_survivors),
                "auprc_y2": _safe_metric(
                    average_precision_score, y2_labels_survivors, y2_probs_survivors
                ),
                "ece_y2": _compute_ece(y2_probs_survivors, y2_labels_survivors),
            }
        )

    return metrics
