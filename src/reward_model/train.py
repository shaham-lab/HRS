"""DDP training entry point for the CDSS-ML reward model.

Launched by ``torchrun`` via ``reward_job.sh``.  Orchestrates the full training
loop: DDP initialisation, dataset loading, masking curriculum, AdamW + cosine
LR, early stopping, checkpointing, and metric logging.

Usage::

    # Fresh run
    torchrun --nproc_per_node=2 src/reward_model/train.py \\
        --config config/reward_model.yaml

    # Resume from latest checkpoint
    torchrun --nproc_per_node=2 src/reward_model/train.py \\
        --config config/reward_model.yaml --resume

See Detailed Design §5 (train.py), §6 (DDP implementation), §7 (adversarial
gradient computation), §8 (checkpoint and resume).
"""

import argparse
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from src.reward_model.loss import compute_loss, compute_metrics
from src.reward_model.mimic4_data_loader import Mimic4DataLoader
from src.reward_model.masking import MaskingSchedule
from src.reward_model.model import RewardModel
from src.reward_model.reward_model_utils import (
    ALWAYS_VISIBLE_SLOTS,
    DatasetBundle,
    RewardModelConfig,
    RowGroupBlockSampler,
    broadcast_tensor,
    get_device,
    load_and_validate_config,
    unwrap_ddp,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDP initialisation
# ---------------------------------------------------------------------------


def _init_ddp(num_gpus: int) -> Tuple[int, int, int, bool]:
    """Initialise the DDP process group from torchrun environment variables.

    Reads ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``, and
    ``MASTER_PORT`` set by ``torchrun``.  Calls
    ``dist.init_process_group(backend='nccl')`` and
    ``torch.cuda.set_device(local_rank)``.

    If fewer than 2 CUDA devices are available at runtime, DDP initialisation
    is skipped.  Single-process mode is logged at WARNING and the returned
    ``is_ddp`` flag is ``False``.

    Args:
        num_gpus: Number of GPUs requested via config.

    Returns:
        Tuple of ``(rank, local_rank, world_size, is_ddp)``.  In single-process
        mode all three integers are 0 / 1 / 1 respectively and ``is_ddp`` is
        ``False``.
    """
    configured_gpus = int(num_gpus)
    available_gpus = torch.cuda.device_count()
    if configured_gpus == 1 or available_gpus < 2:
        logger.warning("Insufficient CUDA devices for DDP — running in single-process mode")
        return 0, 0, 1, False

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size <= 1:
        logger.warning("WORLD_SIZE <= 1 — running in single-process mode")
        return 0, 0, 1, False

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, True


# ---------------------------------------------------------------------------
# Dataset loading and broadcast
# ---------------------------------------------------------------------------


def _load_and_broadcast_dataset(
    config: RewardModelConfig,
    rank: int,
    device: torch.device,
    is_ddp: bool,
) -> Tuple[DatasetBundle, float, float]:
    """Load dataset on rank 0 and broadcast pos_weight scalars to all ranks.

    Rank 0 calls ``Mimic4DataLoader(config).load()`` to load and validate the Parquet
    dataset, build the feature index map, and compute positive class weights.
    It then broadcasts ``pos_weight_y1`` and ``pos_weight_y2`` to all other
    ranks via ``broadcast_tensor()``.

    Non-rank-0 processes do not call ``Mimic4DataLoader.load()`` — they receive only
    the two scalar weights and return ``None`` for the bundle.  All ranks wait
    at a barrier after the broadcast.

    Args:
        config: Validated ``RewardModelConfig`` loaded from YAML.
        rank: This process's global rank.
        device: CUDA device for this process.
        is_ddp: Whether DDP is active.  When ``False``, no broadcast is
            performed.

    Returns:
        Tuple of ``(bundle, pos_weight_y1, pos_weight_y2)``.
    """
    bundle: Optional[DatasetBundle] = None
    if rank == 0:
        bundle = Mimic4DataLoader(config).load()
        pos_weight_y1 = bundle.pos_weight_y1
        pos_weight_y2 = bundle.pos_weight_y2
    else:
        pos_weight_y1 = 0.0
        pos_weight_y2 = 0.0

    tensor_y1 = torch.tensor(pos_weight_y1, device=device)
    tensor_y2 = torch.tensor(pos_weight_y2, device=device)
    if is_ddp:
        broadcast_tensor(tensor_y1, src_rank=0)
        broadcast_tensor(tensor_y2, src_rank=0)
        obj_list = [bundle]
        dist.broadcast_object_list(obj_list, src=0)
        bundle = obj_list[0]
        dist.barrier()

    return bundle, float(tensor_y1.item()), float(tensor_y2.item())


# ---------------------------------------------------------------------------
# Learning-rate scheduler
# ---------------------------------------------------------------------------


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: RewardModelConfig,
    steps_per_epoch: int,
    start_step: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build the cosine annealing scheduler with linear warmup.

    Warmup phase: linearly ramps the LR from 0 to ``LEARNING_RATE`` over
    ``LR_WARMUP_EPOCHS`` epochs (converted to steps).  After warmup: cosine
    decay from ``LEARNING_RATE`` down to ``LR_MIN`` over the remaining *steps*.
    ``scheduler.step()`` is invoked once per **batch** (step-based scheduling).

    Args:
        optimizer: The AdamW optimiser instance.
        config: Validated ``RewardModelConfig``.
        steps_per_epoch: Number of batches per epoch (used to convert warmup
            epochs to steps).
        start_step: Step offset when resuming to fast-forward the scheduler.

    Returns:
         A ``torch.optim.lr_scheduler`` instance ready to call
         ``scheduler.step()`` once per batch.
    """
    warmup_steps = max(config.LR_WARMUP_EPOCHS * max(steps_per_epoch, 1), 0)
    total_steps = max(config.MAX_EPOCHS * max(steps_per_epoch, 1), 1)

    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.0,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=config.LR_MIN
        )
        scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=config.LR_MIN
        )

    # The fast-forward path is kept for correctness even though start_step is
    # typically 0 in current usage.
    for _ in range(start_step):
        scheduler.step()
    return scheduler


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Return the path of the highest-epoch checkpoint in *checkpoint_dir*.

    Checkpoint filenames follow the pattern ``epoch_<N>.pt``.  The latest
    checkpoint is identified by the epoch number embedded in the filename, not
    by filesystem modification time.

    Args:
        checkpoint_dir: Directory to search.

    Returns:
        Path to the latest ``epoch_<N>.pt`` file, or ``None`` if no checkpoint
        exists.
    """
    checkpoints = []
    for path in checkpoint_dir.glob("epoch_*.pt"):
        try:
            epoch_str = path.stem.split("_", maxsplit=1)[1]
            epoch_num = int(epoch_str)
            checkpoints.append((epoch_num, path))
        except (IndexError, ValueError):
            continue
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda pair: pair[0], reverse=True)
    return checkpoints[0][1]


def _save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_dev_loss: float,
    feature_index_map: Dict[str, Tuple[int, int]],
    config: RewardModelConfig,
) -> Path:
    """Write a checkpoint to *checkpoint_dir*/epoch_<epoch>.pt (rank 0 only).

    Checkpoint contents (Detailed Design §8):
        - Model state dict, unwrapped from DDP via ``unwrap_ddp()``
        - Optimiser state dict
        - LR scheduler state dict
        - Current epoch number
        - Best dev loss seen so far
        - Feature index map snapshot
        - Full config snapshot serialised from the Pydantic model

    Uses an atomic write (temp file + ``os.replace``) to prevent partial
    checkpoints on SLURM preemption.

    Args:
        checkpoint_dir: Directory to write the checkpoint into.
        model: The (possibly DDP-wrapped) model.
        optimizer: AdamW optimiser.
        scheduler: LR scheduler.
        epoch: Current epoch number (0-indexed).
        best_dev_loss: Best dev loss seen up to and including this epoch.
        feature_index_map: Feature index map snapshot from the loaded dataset.
        config: Validated ``RewardModelConfig``.

    Returns:
        Path of the written checkpoint file.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": unwrap_ddp(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "masking_schedule_state": {
            "start_ratios": config.MASKING_START_RATIOS,
            "end_ratios": config.MASKING_END_RATIOS,
            "transition_shape": config.MASKING_TRANSITION_SHAPE,
            "transition_midpoint_epoch": config.MASKING_TRANSITION_MIDPOINT_EPOCH,
            "total_epochs": config.MAX_EPOCHS,
            "k": config.MASKING_K,
            "always_visible_slots": list(ALWAYS_VISIBLE_SLOTS),
        },
        "best_dev_loss": best_dev_loss,
        "feature_index_map": feature_index_map,
        "config": config.model_dump(),
    }

    target_path = checkpoint_dir / f"epoch_{epoch}.pt"
    tmp_path = target_path.with_suffix(".pt.tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, target_path)
    return target_path


def _save_best_model(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    epoch: int,
    best_dev_loss: float,
    feature_index_map: Dict[str, Tuple[int, int]],
    config: RewardModelConfig,
) -> None:
    """Overwrite *checkpoint_dir*/best_model.pt atomically (rank 0 only).

    ``best_model.pt`` always reflects the highest-performing epoch seen so far.
    It is written independently of the rolling epoch checkpoint and is always
    retained regardless of ``CHECKPOINT_KEEP_N``.

    Args:
        checkpoint_dir: Directory containing ``best_model.pt``.
        model: The (possibly DDP-wrapped) model.
        epoch: Epoch that achieved the new best dev loss.
        best_dev_loss: The new best dev loss value.
        feature_index_map: Feature index map snapshot.
        config: Validated ``RewardModelConfig``.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": unwrap_ddp(model).state_dict(),
        "epoch": epoch,
        "best_dev_loss": best_dev_loss,
        "feature_index_map": feature_index_map,
        "config": config.model_dump(),
    }
    target_path = checkpoint_dir / "best_model.pt"
    tmp_path = target_path.with_suffix(".pt.tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, target_path)


def _prune_old_checkpoints(checkpoint_dir: Path, keep_n: int) -> None:
    """Delete epoch checkpoints beyond the *keep_n* most recent (rank 0 only).

    Identifies all ``epoch_<N>.pt`` files in *checkpoint_dir* by their epoch
    numbers, retains the *keep_n* highest, and deletes the rest.
    ``best_model.pt`` is never touched.

    Args:
        checkpoint_dir: Checkpoint directory to prune.
        keep_n: Number of most-recent epoch checkpoints to retain
            (``CHECKPOINT_KEEP_N`` in config).
    """
    checkpoints = []
    for path in checkpoint_dir.glob("epoch_*.pt"):
        try:
            epoch_num = int(path.stem.split("_", maxsplit=1)[1])
            checkpoints.append((epoch_num, path))
        except (IndexError, ValueError):
            continue

    checkpoints.sort(key=lambda pair: pair[0], reverse=True)
    for _, path in checkpoints[keep_n:]:
        try:
            path.unlink()
        except FileNotFoundError:
            continue


# ---------------------------------------------------------------------------
# Metrics logging
# ---------------------------------------------------------------------------


def _append_metrics_row(metrics_path: Path, row: Dict[str, float]) -> None:
    """Append one epoch's metrics to *metrics_path* (rank 0 only).

    Uses an atomic write (write to temp file, rename) so that a SLURM
    preemption mid-write cannot corrupt the Parquet file.  If the file does
    not yet exist it is created with the correct schema.

    Columns written (Architecture §9):
        epoch, wall_time_s, masking_random_pct, masking_adversarial_pct,
        masking_none_pct, loss_total, loss_y1, loss_y2,
        auroc_y1, auprc_y1, ece_y1, auroc_y2, auprc_y2, ece_y2.

    Args:
        metrics_path: Path to ``training_metrics.parquet``.
        row: Dict mapping column name to scalar value for this epoch.
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
        "loss_y1",
        "loss_y2",
        "auroc_y1",
        "auprc_y1",
        "ece_y1",
        "auroc_y2",
        "auprc_y2",
        "ece_y2",
    ]
    schema = pa.schema(
        [
            ("epoch", pa.int64()),
            ("wall_time_s", pa.float64()),
            ("masking_random_pct", pa.float64()),
            ("masking_adversarial_pct", pa.float64()),
            ("masking_none_pct", pa.float64()),
            ("loss_total", pa.float64()),
            ("loss_y1", pa.float64()),
            ("loss_y2", pa.float64()),
            ("auroc_y1", pa.float64()),
            ("auprc_y1", pa.float64()),
            ("ece_y1", pa.float64()),
            ("auroc_y2", pa.float64()),
            ("auprc_y2", pa.float64()),
            ("ece_y2", pa.float64()),
        ]
    )

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


# ---------------------------------------------------------------------------
# Training batch
# ---------------------------------------------------------------------------


def _run_train_batch(
    model: torch.nn.Module,
    X: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    masking_schedule: MaskingSchedule,
    optimizer: torch.optim.Optimizer,
    pos_weight_y1: float,
    pos_weight_y2: float,
    config: RewardModelConfig,
    epoch: int,
    is_ddp: bool,
) -> Tuple[float, float, float]:
    """Execute one mini-batch forward/backward/step.

    Masking mode is sampled from *masking_schedule* for this batch.

    For **random** and **none** modes, a single forward/backward pass is run
    with the (possibly masked) input.  DDP all-reduce fires normally on
    ``loss.backward()``.

    For **adversarial** mode, two passes are required (Detailed Design §7):

      1. Clone *X*, call ``requires_grad_(True)`` on the clone.  Run forward
         and backward inside ``model.no_sync()`` (DDP) or ``nullcontext()``
         (single-GPU).  Read ``clone.grad`` to obtain ``∂L/∂X``.
      2. Call ``optimizer.zero_grad()``.  Apply
         ``masking_schedule.apply_adversarial_mask(X, grad_X)`` using the
         gradient from step 1.  Run the second forward/backward pass normally
         (DDP all-reduce fires here).

    Args:
        model: The (possibly DDP-wrapped) ``RewardModel``.
        X: Input batch tensor ``(batch_size, input_dim)``, float32.
        y1: Mortality labels ``(batch_size,)``, int8.
        y2: Readmission labels ``(batch_size,)``, float32 with NaN for deceased.
        masking_schedule: Active ``MaskingSchedule`` instance.
        optimizer: AdamW optimiser.
        pos_weight_y1: Positive class weight for Y1.
        pos_weight_y2: Positive class weight for Y2 (survivors only).
        config: Validated ``RewardModelConfig``.
        epoch: Current epoch (passed to ``masking_schedule.sample_mode()``).
        is_ddp: Whether the model is DDP-wrapped (determines whether
            ``model.no_sync()`` is available or ``nullcontext()`` is used).

    Returns:
        Tuple of ``(total_loss, loss_y1, loss_y2)`` as Python floats for
        epoch-level accumulation before logging.
    """
    mode = masking_schedule.sample_mode(epoch)

    optimizer.zero_grad()

    if mode == "adversarial":
        context = model.no_sync() if is_ddp and hasattr(model, "no_sync") else nullcontext()
        X_grad = X.clone().requires_grad_(True)
        with context:
            logits_y1, logits_y2 = model(X_grad)
            loss_total, loss_y1_t, loss_y2_t = compute_loss(
                logits_y1,
                logits_y2,
                y1,
                y2,
                pos_weight_y1,
                pos_weight_y2,
                config.LOSS_WEIGHT_Y1,
                config.LOSS_WEIGHT_Y2,
            )
            loss_total.backward()
        grad_X = X_grad.grad.detach()
        optimizer.zero_grad()
        X_masked = masking_schedule.apply_adversarial_mask(X, grad_X)
        logits_y1, logits_y2 = model(X_masked)
        loss_total, loss_y1_t, loss_y2_t = compute_loss(
            logits_y1,
            logits_y2,
            y1,
            y2,
            pos_weight_y1,
            pos_weight_y2,
            config.LOSS_WEIGHT_Y1,
            config.LOSS_WEIGHT_Y2,
        )
        loss_total.backward()
    else:
        if mode == "random":
            X = masking_schedule.apply_random_mask(X)
        else:
            X = masking_schedule.apply_no_mask(X)
        logits_y1, logits_y2 = model(X)
        loss_total, loss_y1_t, loss_y2_t = compute_loss(
            logits_y1,
            logits_y2,
            y1,
            y2,
            pos_weight_y1,
            pos_weight_y2,
            config.LOSS_WEIGHT_Y1,
            config.LOSS_WEIGHT_Y2,
        )
        loss_total.backward()

    optimizer.step()

    return (
        float(loss_total.detach().item()),
        float(loss_y1_t.detach().item()),
        float(loss_y2_t.detach().item()),
    )


# ---------------------------------------------------------------------------
# Dev evaluation
# ---------------------------------------------------------------------------


def _eval_dev(
    model: torch.nn.Module,
    dev_dataset: torch.utils.data.Dataset,
    pos_weight_y1: float,
    pos_weight_y2: float,
    config: RewardModelConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Run full dev-split evaluation on rank 0 (no DDP, no masking).

    Iterates the complete *dev_dataset* via a standard (non-distributed)
    ``DataLoader`` with ``torch.no_grad()``, accumulates logits and labels,
    computes total loss via ``compute_loss()``, and computes metrics via
    ``compute_metrics()``.  The dev tensor is never fully resident in RAM —
    rows are read lazily from ``ParquetDataset`` one batch at a time.

    Called once at the end of each epoch by rank 0 only.

    Args:
        model: The (possibly DDP-wrapped) ``RewardModel``.  Unwrapped
            internally via ``unwrap_ddp()`` before evaluation.
        dev_dataset: Dev-split ``ParquetDataset`` from ``DatasetBundle``.
        pos_weight_y1: Positive class weight for Y1 loss computation.
        pos_weight_y2: Positive class weight for Y2 loss computation.
        config: Validated ``RewardModelConfig``.
        device: CUDA device for rank 0.

    Returns:
        Dict with keys: ``loss_total``, ``loss_y1``, ``loss_y2``,
        ``auroc_y1``, ``auprc_y1``, ``ece_y1``,
        ``auroc_y2``, ``auprc_y2``, ``ece_y2``.
    """
    eval_model = unwrap_ddp(model)
    eval_model.eval()

    dataloader = DataLoader(
        dev_dataset,
        batch_size=config.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.DATALOADER_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    total_loss = 0.0
    total_loss_y1 = 0.0
    total_loss_y2 = 0.0
    n_batches = 0

    logits_y1_all = []
    logits_y2_all = []
    y1_all = []
    y2_all = []

    with torch.no_grad():
        for X, y1, y2 in dataloader:
            X = X.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)

            logits_y1, logits_y2 = eval_model(X)
            loss_total, loss_y1_t, loss_y2_t = compute_loss(
                logits_y1,
                logits_y2,
                y1,
                y2,
                pos_weight_y1,
                pos_weight_y2,
                config.LOSS_WEIGHT_Y1,
                config.LOSS_WEIGHT_Y2,
            )

            total_loss += float(loss_total.detach().item())
            total_loss_y1 += float(loss_y1_t.detach().item())
            total_loss_y2 += float(loss_y2_t.detach().item())
            n_batches += 1

            logits_y1_all.append(logits_y1.detach())
            logits_y2_all.append(logits_y2.detach())
            y1_all.append(y1.detach())
            y2_all.append(y2.detach())

    if n_batches == 0:
        return {
            "loss_total": float("nan"),
            "loss_y1": float("nan"),
            "loss_y2": float("nan"),
            "auroc_y1": float("nan"),
            "auprc_y1": float("nan"),
            "ece_y1": float("nan"),
            "auroc_y2": float("nan"),
            "auprc_y2": float("nan"),
            "ece_y2": float("nan"),
        }

    logits_y1_cat = torch.cat(logits_y1_all, dim=0)
    logits_y2_cat = torch.cat(logits_y2_all, dim=0)
    y1_cat = torch.cat(y1_all, dim=0)
    y2_cat = torch.cat(y2_all, dim=0)

    metrics = compute_metrics(logits_y1_cat, logits_y2_cat, y1_cat, y2_cat)
    metrics.update(
        {
            "loss_total": total_loss / n_batches,
            "loss_y1": total_loss_y1 / n_batches,
            "loss_y2": total_loss_y2 / n_batches,
        }
    )
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """DDP training loop entry point launched by torchrun.

    Full 11-step algorithm (Detailed Design §5, train.py):

      1.  Parse CLI arguments: ``--config``, ``--resume``.
      2.  Load and validate config via ``load_and_validate_config()``.
          Configure root logging handler (rank 0: INFO to stdout;
          all ranks: ERROR/CRITICAL).
      3.  Initialise DDP process group via ``_init_ddp()``.  Fall back to
          single-process mode if fewer than 2 CUDA devices are available
          (logged at WARNING).
      4.  Rank 0 calls ``Mimic4DataLoader(config).load()``; broadcasts
          ``pos_weight_y1`` and ``pos_weight_y2`` via
          ``_load_and_broadcast_dataset()``.  Non-rank-0 processes wait at
          barrier; receive scalar weights only.
      5.  Instantiate ``RewardModel(input_dim, layer_widths, dropout_rate,
          activation)``; move to ``get_device(local_rank)``; wrap in
          ``DistributedDataParallel`` if multi-GPU.
      6.  Instantiate ``MaskingSchedule`` from config masking keys.
      7.  Instantiate ``AdamW`` with ``LEARNING_RATE``, ``WEIGHT_DECAY``,
          ``ADAM_BETA1``, ``ADAM_BETA2``.  Build cosine + warmup scheduler
          via ``_build_lr_scheduler()``.
      8.  If ``--resume``: rank 0 finds latest checkpoint via
          ``_find_latest_checkpoint()``; loads model, optimizer, scheduler,
          epoch, best_dev_loss; validates feature index map snapshot against
          current dataset (raises on mismatch); broadcasts model state dict
          via ``dist.broadcast_object_list()``; ``dist.barrier()``.
      9.  Construct ``RowGroupBlockSampler`` over ``train_dataset`` (round-robin
          row group partitioning across ranks); wrap in ``DataLoader`` with
          ``batch_size = BATCH_SIZE_PER_GPU``.
      10. Epoch loop from ``start_epoch`` to ``MAX_EPOCHS``:
            a. ``sampler.set_epoch(epoch)`` for per-epoch reshuffling.
            b. Batch loop: call ``_run_train_batch()`` per mini-batch.
            c. ``dist.barrier()`` (all ranks wait before rank 0 I/O).
            d. Rank 0: call ``_eval_dev()``; ``_append_metrics_row()``;
               check early stopping patience.
            e. Rank 0: if dev loss improved, ``_save_checkpoint()``,
               ``_save_best_model()``, ``_prune_old_checkpoints()``.
            f. Broadcast early stopping boolean from rank 0 to all ranks
               via ``broadcast_tensor()``.  All ranks break simultaneously.
            g. ``dist.barrier()`` (all ranks resume next epoch together).
      11. Rank 0: write final checkpoint.  ``dist.destroy_process_group()``.
    """
    parser = argparse.ArgumentParser(description="Train the CDSS-ML reward model")
    parser.add_argument("--config", required=True, help="Path to reward_model.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    config = load_and_validate_config(args.config)

    initial_rank = int(os.environ.get("RANK", "0"))
    if initial_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            stream=sys.stdout,
        )
    else:
        logging.basicConfig(level=logging.ERROR)

    rank, local_rank, world_size, is_ddp = _init_ddp(config.NUM_GPUS)
    device = get_device(local_rank)

    bundle, pos_weight_y1, pos_weight_y2 = _load_and_broadcast_dataset(config, rank, device, is_ddp)

    feature_index_map: Optional[Dict[str, Tuple[int, int]]] = bundle.feature_index_map if bundle else None
    input_dim: Optional[int] = bundle.input_dim if bundle else None
    train_dataset = bundle.train_dataset if bundle else None
    dev_dataset = bundle.dev_dataset if bundle else None

    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    metrics_path = Path(config.METRICS_PATH)

    if feature_index_map is None or input_dim is None or train_dataset is None or dev_dataset is None:
        raise RuntimeError("Dataset must be available on all ranks after broadcast")

    start_epoch = 0
    best_dev_loss = float("inf")
    ckpt_state: Optional[dict] = None

    if args.resume:
        latest = _find_latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise RuntimeError("Resume requested but no checkpoint found")
        if rank == 0:
            ckpt_state = torch.load(latest, map_location="cpu")
            ckpt_feature_map = ckpt_state["feature_index_map"]
            if feature_index_map != ckpt_feature_map:
                raise RuntimeError("Feature index map mismatch between checkpoint and current dataset")
        if is_ddp:
            obj_state: list = [ckpt_state]
            dist.broadcast_object_list(obj_state, src=0)
            ckpt_state = obj_state[0]
            dist.barrier()
        if ckpt_state is None:
            raise RuntimeError("Checkpoint state could not be loaded")
        start_epoch = ckpt_state["epoch"] + 1
        best_dev_loss = ckpt_state.get("best_dev_loss", best_dev_loss)

    model = RewardModel(
        input_dim=input_dim,
        layer_widths=config.LAYER_WIDTHS,
        dropout_rate=config.DROPOUT_RATE,
        activation=config.ACTIVATION,
    ).to(device)
    if is_ddp:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(config.ADAM_BETA1, config.ADAM_BETA2),
    )

    if ckpt_state is not None:
        ms = ckpt_state["masking_schedule_state"]
        masking_schedule = MaskingSchedule(
            feature_index_map=feature_index_map if feature_index_map is not None else {},
            start_ratios=ms["start_ratios"],
            end_ratios=ms["end_ratios"],
            transition_shape=ms["transition_shape"],
            transition_midpoint_epoch=ms["transition_midpoint_epoch"],
            total_epochs=ms["total_epochs"],
            k=ms["k"],
            always_visible_slots=ms.get("always_visible_slots", ALWAYS_VISIBLE_SLOTS),
        )
    else:
        masking_schedule = MaskingSchedule(
            feature_index_map=feature_index_map if feature_index_map is not None else {},
            start_ratios=config.MASKING_START_RATIOS,
            end_ratios=config.MASKING_END_RATIOS,
            transition_shape=config.MASKING_TRANSITION_SHAPE,
            transition_midpoint_epoch=config.MASKING_TRANSITION_MIDPOINT_EPOCH,
            total_epochs=config.MAX_EPOCHS,
            k=config.MASKING_K,
            always_visible_slots=ALWAYS_VISIBLE_SLOTS,
        )

    if is_ddp and train_dataset is None:
        raise RuntimeError("Train dataset unavailable for DDP training")

    if train_dataset is not None:
        sampler = RowGroupBlockSampler(
            dataset=train_dataset,
            rank=rank,
            world_size=world_size if is_ddp else 1,
            shuffle=True,
            seed=0,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE_PER_GPU,
            sampler=sampler,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = None

    steps_per_epoch = len(train_loader) if train_loader is not None else 1
    start_step = start_epoch * steps_per_epoch
    scheduler = _build_lr_scheduler(optimizer, config, steps_per_epoch, start_step)

    if ckpt_state is not None:
        unwrap_ddp(model).load_state_dict(ckpt_state["model_state_dict"])
        optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt_state["scheduler_state_dict"])

    epochs_without_improve = 0

    last_epoch_completed = start_epoch - 1

    for epoch in range(start_epoch, config.MAX_EPOCHS):
        if train_loader is None:
            break

        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        epoch_loss_total = 0.0
        epoch_loss_y1 = 0.0
        epoch_loss_y2 = 0.0
        batches = 0

        epoch_start = time.time()

        for X, y1, y2 in train_loader:
            X = X.to(device, non_blocking=True)
            y1 = y1.to(device, non_blocking=True)
            y2 = y2.to(device, non_blocking=True)

            total, l1, l2 = _run_train_batch(
                model,
                X,
                y1,
                y2,
                masking_schedule,
                optimizer,
                pos_weight_y1,
                pos_weight_y2,
                config,
                epoch,
                is_ddp,
            )
            scheduler.step()

            epoch_loss_total += total
            epoch_loss_y1 += l1
            epoch_loss_y2 += l2
            batches += 1

        if is_ddp:
            dist.barrier()

        if rank == 0:
            probs = masking_schedule.get_mode_probabilities(epoch)
            dev_metrics = _eval_dev(model, dev_dataset, pos_weight_y1, pos_weight_y2, config, device)
            model.train()

            row = {
                "epoch": epoch,
                "wall_time_s": float(time.time() - epoch_start),
                "masking_random_pct": probs[0] * 100.0,
                "masking_adversarial_pct": probs[1] * 100.0,
                "masking_none_pct": probs[2] * 100.0,
                "loss_total": dev_metrics["loss_total"],
                "loss_y1": dev_metrics["loss_y1"],
                "loss_y2": dev_metrics["loss_y2"],
                "auroc_y1": dev_metrics["auroc_y1"],
                "auprc_y1": dev_metrics["auprc_y1"],
                "ece_y1": dev_metrics["ece_y1"],
                "auroc_y2": dev_metrics["auroc_y2"],
                "auprc_y2": dev_metrics["auprc_y2"],
                "ece_y2": dev_metrics["ece_y2"],
            }
            _append_metrics_row(metrics_path, row)

            current_dev_loss = dev_metrics["loss_total"]
            improved = current_dev_loss < best_dev_loss
            if improved:
                best_dev_loss = current_dev_loss
                epochs_without_improve = 0
                _save_checkpoint(
                    checkpoint_dir,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_dev_loss,
                    feature_index_map if feature_index_map is not None else {},
                    config,
                )
                _save_best_model(
                    checkpoint_dir,
                    model,
                    epoch,
                    best_dev_loss,
                    feature_index_map if feature_index_map is not None else {},
                    config,
                )
                _prune_old_checkpoints(checkpoint_dir, config.CHECKPOINT_KEEP_N)
            else:
                epochs_without_improve += 1

            should_stop = epochs_without_improve >= config.EARLY_STOPPING_PATIENCE
        else:
            should_stop = False

        if is_ddp:
            stop_tensor = torch.tensor(1 if should_stop else 0, device=device)
            broadcast_tensor(stop_tensor, src_rank=0)
            should_stop = bool(stop_tensor.item())

        if should_stop:
            break

        if is_ddp:
            dist.barrier()

        last_epoch_completed = epoch

    if is_ddp:
        dist.barrier()
    if rank == 0:
        _save_checkpoint(
            checkpoint_dir,
            model,
            optimizer,
            scheduler,
            last_epoch_completed if last_epoch_completed >= 0 else 0,
            best_dev_loss,
            feature_index_map if feature_index_map is not None else {},
            config,
        )
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
