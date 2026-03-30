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

from checkpoint_manager import CheckpointManager
from loss import compute_loss, compute_metrics
from mimic4_data_loader import Mimic4DataLoader
from masking import MaskingSchedule
from model import RewardModel
from dataset_bundle import DatasetBundle
from reward_model_config import RewardModelConfig, load_and_validate_config
from reward_model_utils import get_device
from row_group_block_sampler import RowGroupBlockSampler

logger = logging.getLogger(__name__)


def broadcast_tensor(tensor: torch.Tensor, src_rank: int) -> torch.Tensor:
    """Broadcast a tensor from src_rank to all ranks if distributed is initialised."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(tensor, src=src_rank)
    return tensor


def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a DDP-wrapped model to its underlying module."""
    return model.module if hasattr(model, "module") else model


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
) -> Tuple[DatasetBundle, list]:
    """Load dataset on rank 0 and broadcast pos_weights to all ranks.

    Rank 0 calls ``Mimic4DataLoader(config).load()`` to load and validate the Parquet
    dataset, build the feature index map, and compute positive class weights.
    It then broadcasts each ``pos_weight`` scalar to all other ranks via
    ``broadcast_tensor()``.

    Non-rank-0 processes do not call ``Mimic4DataLoader.load()`` — they receive only
    the scalar weights and return ``None`` for the bundle.  All ranks wait
    at a barrier after the broadcast.

    Args:
        config: Validated ``RewardModelConfig`` loaded from YAML.
        rank: This process's global rank.
        device: CUDA device for this process.
        is_ddp: Whether DDP is active.  When ``False``, no broadcast is
            performed.

    Returns:
        Tuple of ``(bundle, pos_weights)`` where pos_weights is a List[float].
    """
    T = config.NUM_TARGETS
    bundle: Optional[DatasetBundle] = None
    if rank == 0:
        bundle = Mimic4DataLoader(config).load()
        pos_weights_local = list(bundle.pos_weights)
    else:
        pos_weights_local = [0.0] * T

    weight_tensors = [torch.tensor(w, device=device) for w in pos_weights_local]
    if is_ddp:
        for t in weight_tensors:
            broadcast_tensor(t, src_rank=0)
        obj_list = [bundle]
        dist.broadcast_object_list(obj_list, src=0)
        bundle = obj_list[0]
        dist.barrier()

    return bundle, [float(t.item()) for t in weight_tensors]


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
# Metrics logging
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Dev evaluation
# ---------------------------------------------------------------------------


def _eval_dev(
    model: torch.nn.Module,
    dev_dataset: torch.utils.data.Dataset,
    pos_weights: list,
    config: RewardModelConfig,
    device: torch.device,
) -> Dict:
    """Run full dev-split evaluation on rank 0 (no DDP, no masking).

    Returns a dict with keys ``loss_total``, ``loss_target_<i>``,
    ``auroc_target_<i>``, ``auprc_target_<i>``, ``ece_target_<i>``
    for each target i in 0..NUM_TARGETS-1.
    """
    T = config.NUM_TARGETS
    eval_model = unwrap_ddp(model)
    eval_model.eval()

    dataloader = DataLoader(
        dev_dataset,
        batch_size=config.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.DATALOADER_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    total_loss_acc = 0.0
    component_loss_acc = [0.0] * T
    n_batches = 0

    logits_all = [[] for _ in range(T)]
    labels_all = [[] for _ in range(T)]

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            batch_labels = [batch[i + 1].to(device) for i in range(T)]

            logits_list = list(eval_model(X))
            loss_total, component_losses = compute_loss(
                logits_list, batch_labels, pos_weights, config.LOSS_WEIGHTS
            )

            total_loss_acc += float(loss_total.detach().item())
            for i, cl in enumerate(component_losses):
                component_loss_acc[i] += float(cl.detach().item())
            n_batches += 1

            for i in range(T):
                logits_all[i].append(logits_list[i].detach())
                labels_all[i].append(batch_labels[i].detach())

    nan_result: Dict = {"loss_total": float("nan")}
    for i in range(T):
        nan_result[f"loss_target_{i}"] = float("nan")
        nan_result[f"auroc_target_{i}"] = float("nan")
        nan_result[f"auprc_target_{i}"] = float("nan")
        nan_result[f"ece_target_{i}"] = float("nan")
    if n_batches == 0:
        return nan_result

    logits_cat = [torch.cat(logits_all[i], dim=0) for i in range(T)]
    labels_cat = [torch.cat(labels_all[i], dim=0) for i in range(T)]

    per_target = compute_metrics(logits_cat, labels_cat, masked=False)

    result: Dict = {"loss_total": total_loss_acc / n_batches}
    for i in range(T):
        result[f"loss_target_{i}"] = component_loss_acc[i] / n_batches
        target_metrics = per_target[i]
        result[f"auroc_target_{i}"] = target_metrics["auroc"]
        result[f"auprc_target_{i}"] = target_metrics["auprc"]
        result[f"ece_target_{i}"] = target_metrics["ece"]
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CDSS-ML reward model")
    parser.add_argument("--config", required=True, help="Path to reward_model.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return parser.parse_args()


def _setup_logging(initial_rank: int) -> None:
    if initial_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            stream=sys.stdout,
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def _init_runtime(config: RewardModelConfig) -> Tuple[int, int, int, bool, torch.device]:
    rank, local_rank, world_size, is_ddp = _init_ddp(config.NUM_GPUS)
    device = get_device(local_rank)
    return rank, local_rank, world_size, is_ddp, device


def _load_datasets_and_weights(
    config: RewardModelConfig, rank: int, device: torch.device, is_ddp: bool
) -> Tuple[DatasetBundle, list]:
    bundle, pos_weights = _load_and_broadcast_dataset(config, rank, device, is_ddp)
    if bundle is None:
        raise RuntimeError("Dataset must be available on all ranks after broadcast")
    return bundle, pos_weights


def _resume_from_checkpoint(
    args: argparse.Namespace,
    checkpoint_manager: CheckpointManager,
    feature_index_map: Dict[str, Tuple[int, int]],
    config: RewardModelConfig,
    rank: int,
    is_ddp: bool,
) -> Tuple[Optional[dict], int, float]:
    start_epoch = 0
    best_dev_loss = float("inf")
    ckpt_state: Optional[dict] = None

    if args.resume:
        latest = checkpoint_manager.find_latest()
        if latest is None:
            raise RuntimeError("Resume requested but no checkpoint found")
        if rank == 0:
            ckpt_state = checkpoint_manager.load(latest, feature_index_map)
        if is_ddp:
            obj_state: list = [ckpt_state]
            dist.broadcast_object_list(obj_state, src=0)
            ckpt_state = obj_state[0]
            dist.barrier()
        if ckpt_state is None:
            raise RuntimeError("Checkpoint state could not be loaded")
        start_epoch = ckpt_state["epoch"] + 1
        best_dev_loss = ckpt_state.get("best_dev_loss", best_dev_loss)

    return ckpt_state, start_epoch, best_dev_loss


def _build_masking_schedule(
    config: RewardModelConfig, feature_index_map: Dict[str, Tuple[int, int]], ckpt_state: Optional[dict]
) -> MaskingSchedule:
    if ckpt_state is not None:
        ms = ckpt_state["masking_schedule_state"]
        return MaskingSchedule(
            feature_index_map=feature_index_map,
            start_ratios=ms["start_ratios"],
            end_ratios=ms["end_ratios"],
            transition_shape=ms["transition_shape"],
            transition_midpoint_epoch=ms["transition_midpoint_epoch"],
            total_epochs=ms["total_epochs"],
            num_always_visible=ms["num_always_visible"],
            random_k_min_fraction=ms["random_k_min_fraction"],
            random_k_max_fraction=ms["random_k_max_fraction"],
            adversarial_k_min_fraction=ms["adversarial_k_min_fraction"],
            adversarial_k_max_fraction=ms["adversarial_k_max_fraction"],
        )

    return MaskingSchedule(
        feature_index_map=feature_index_map,
        start_ratios=config.MASKING_START_RATIOS,
        end_ratios=config.MASKING_END_RATIOS,
        transition_shape=config.MASKING_TRANSITION_SHAPE,
        transition_midpoint_epoch=config.MASKING_TRANSITION_MIDPOINT_EPOCH,
        total_epochs=config.MAX_EPOCHS,
        num_always_visible=config.NUM_ALWAYS_VISIBLE_FEATURES,
        random_k_min_fraction=config.MASKING_RANDOM_K_MIN_FRACTION,
        random_k_max_fraction=config.MASKING_RANDOM_K_MAX_FRACTION,
        adversarial_k_min_fraction=config.MASKING_ADVERSARIAL_K_MIN_FRACTION,
        adversarial_k_max_fraction=config.MASKING_ADVERSARIAL_K_MAX_FRACTION,
    )


def _build_train_loader(
    train_dataset: Optional[torch.utils.data.Dataset],
    config: RewardModelConfig,
    rank: int,
    world_size: int,
    is_ddp: bool,
) -> Optional[DataLoader]:
    if train_dataset is None:
        return None

    sampler = RowGroupBlockSampler(
        dataset=train_dataset,
        rank=rank,
        world_size=world_size if is_ddp else 1,
        shuffle=True,
        seed=0,
    )
    return DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE_PER_GPU,
        sampler=sampler,
        num_workers=config.DATALOADER_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def _build_model(
    input_dim: int, config: RewardModelConfig, device: torch.device, local_rank: int, is_ddp: bool
) -> torch.nn.Module:
    model = RewardModel(
        input_dim=input_dim,
        layer_widths=config.LAYER_WIDTHS,
        dropout_rates=config.DROPOUT_RATES,
        activation=config.ACTIVATION,
        num_targets=config.NUM_TARGETS,
    ).to(device)
    if is_ddp:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    return model


def _build_optimizer(model: torch.nn.Module, config: RewardModelConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(config.ADAM_BETA1, config.ADAM_BETA2),
    )


def _maybe_load_states(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_state: Optional[dict],
) -> None:
    if ckpt_state is None:
        return
    unwrap_ddp(model).load_state_dict(ckpt_state["model_state_dict"])
    optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt_state["scheduler_state_dict"])


class TrainManager:
    def __init__(
        self,
        config: RewardModelConfig,
        rank: int,
        local_rank: int,
        world_size: int,
        is_ddp: bool,
        device: torch.device,
    ):
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_ddp = is_ddp
        self.device = device

        self.bundle, self.pos_weights = _load_datasets_and_weights(
            self.config, self.rank, self.device, self.is_ddp
        )
        self.feature_index_map: Dict[str, Tuple[int, int]] = self.bundle.feature_index_map
        self.train_dataset = self.bundle.train_dataset
        self.dev_dataset = self.bundle.dev_dataset
        self.input_dim: int = self.bundle.input_dim

        self.checkpoint_manager = CheckpointManager(
            Path(self.config.CHECKPOINT_DIR), self.config.CHECKPOINT_KEEP_N
        )
        self.metrics_path = Path(self.config.METRICS_PATH)

        self.model = _build_model(self.input_dim, self.config, self.device, self.local_rank, self.is_ddp)
        self.optimizer = _build_optimizer(self.model, self.config)

        self.masking_schedule: Optional[MaskingSchedule] = None
        self.train_loader: Optional[DataLoader] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None

    def setup_training_state(self, ckpt_state: Optional[dict], start_epoch: int) -> None:
        self.masking_schedule = _build_masking_schedule(
            self.config, self.feature_index_map, ckpt_state
        )
        self.train_loader = _build_train_loader(
            self.train_dataset, self.config, self.rank, self.world_size, self.is_ddp
        )
        if self.is_ddp and self.train_loader is None:
            raise RuntimeError("Train dataset unavailable for DDP training")

        steps_per_epoch = len(self.train_loader) if self.train_loader is not None else 1
        start_step = start_epoch * steps_per_epoch
        self.scheduler = _build_lr_scheduler(
            self.optimizer, self.config, steps_per_epoch, start_step
        )
        _maybe_load_states(self.model, self.optimizer, self.scheduler, ckpt_state)

    def _run_train_batch(self, X: torch.Tensor, labels: list, epoch: int) -> Tuple[float, ...]:
        """Execute one mini-batch forward/backward/step."""
        if self.masking_schedule is None:
            raise RuntimeError("Masking schedule must be initialised before training")

        mode = self.masking_schedule.sample_mode(epoch)

        self.optimizer.zero_grad()

        if mode == "adversarial":
            context = (
                self.model.no_sync()
                if self.is_ddp and hasattr(self.model, "no_sync")
                else nullcontext()
            )
            X_grad = X.clone().requires_grad_(True)
            with context:
                logits_list = list(self.model(X_grad))
                loss_total, _ = compute_loss(
                    logits_list, labels, self.pos_weights, self.config.LOSS_WEIGHTS
                )
                loss_total.backward()
            grad_X = X_grad.grad.detach()
            self.optimizer.zero_grad()
            X_masked = self.masking_schedule.apply_adversarial_mask(X, grad_X)
            logits_list = list(self.model(X_masked))
            loss_total, component_losses = compute_loss(
                logits_list, labels, self.pos_weights, self.config.LOSS_WEIGHTS
            )
            loss_total.backward()
        else:
            if mode == "random":
                X = self.masking_schedule.apply_random_mask(X)
            else:
                X = self.masking_schedule.apply_no_mask(X)
            logits_list = list(self.model(X))
            loss_total, component_losses = compute_loss(
                logits_list, labels, self.pos_weights, self.config.LOSS_WEIGHTS
            )
            loss_total.backward()

        self.optimizer.step()

        return (float(loss_total.detach().item()),) + tuple(
            float(l.detach().item()) for l in component_losses
        )

    def train_epochs(self, start_epoch: int, best_dev_loss: float) -> Tuple[int, float]:
        """Run the epoch loop and return the last completed epoch and best dev loss."""
        if self.masking_schedule is None or self.scheduler is None:
            raise RuntimeError("Training state must be set up before training epochs")

        T = self.config.NUM_TARGETS
        epochs_without_improve = 0
        last_epoch_completed = start_epoch - 1

        for epoch in range(start_epoch, self.config.MAX_EPOCHS):
            if self.train_loader is None:
                break

            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            epoch_start = time.time()

            for batch in self.train_loader:
                X = batch[0].to(self.device, non_blocking=True)
                labels = [batch[i + 1].to(self.device, non_blocking=True) for i in range(T)]

                self._run_train_batch(X, labels, epoch)
                self.scheduler.step()

            if self.is_ddp:
                dist.barrier()

            if self.rank == 0:
                probs = self.masking_schedule.get_mode_probabilities(epoch)
                dev_metrics = _eval_dev(
                    self.model, self.dev_dataset, self.pos_weights, self.config, self.device
                )
                self.model.train()

                row: Dict = {
                    "epoch": epoch,
                    "wall_time_s": float(time.time() - epoch_start),
                    "masking_random_pct": probs[0] * 100.0,
                    "masking_adversarial_pct": probs[1] * 100.0,
                    "masking_none_pct": probs[2] * 100.0,
                    "loss_total": dev_metrics["loss_total"],
                }
                for i in range(T):
                    row[f"loss_target_{i}"] = dev_metrics[f"loss_target_{i}"]
                    row[f"auroc_target_{i}"] = dev_metrics[f"auroc_target_{i}"]
                    row[f"auprc_target_{i}"] = dev_metrics[f"auprc_target_{i}"]
                    row[f"ece_target_{i}"] = dev_metrics[f"ece_target_{i}"]
                _append_metrics_row(self.metrics_path, row, num_targets=T)

                current_dev_loss = dev_metrics["loss_total"]
                improved = current_dev_loss < best_dev_loss
                if improved:
                    best_dev_loss = current_dev_loss
                    epochs_without_improve = 0
                    self.checkpoint_manager.save_epoch_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        best_dev_loss,
                        self.feature_index_map if self.feature_index_map is not None else {},
                        self.config,
                    )
                    self.checkpoint_manager.save_best_model(
                        self.model,
                        epoch,
                        best_dev_loss,
                        self.feature_index_map if self.feature_index_map is not None else {},
                        self.config,
                    )
                    self.checkpoint_manager.prune_old_checkpoints()
                else:
                    epochs_without_improve += 1

                should_stop = epochs_without_improve >= self.config.EARLY_STOPPING_PATIENCE
            else:
                should_stop = False

            if self.is_ddp:
                stop_tensor = torch.tensor(1 if should_stop else 0, device=self.device)
                broadcast_tensor(stop_tensor, src_rank=0)
                should_stop = bool(stop_tensor.item())

            if should_stop:
                break

            if self.is_ddp:
                dist.barrier()

            last_epoch_completed = epoch

        if self.is_ddp:
            dist.barrier()
        if self.rank == 0:
            self.checkpoint_manager.save_epoch_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                last_epoch_completed if last_epoch_completed >= 0 else 0,
                best_dev_loss,
                self.feature_index_map if self.feature_index_map is not None else {},
                self.config,
            )
        if self.is_ddp:
            dist.barrier()

        return last_epoch_completed, best_dev_loss
