"""Training utilities and manager class for the CDSS-ML reward model.

Houses data loading, masking curriculum, model/optimizer construction, and the
RewardModelManager class. The torchrun entry point lives in
``reward_model_main.py``.
"""

import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from checkpoint_manager import CheckpointManager
from metrics import MetricsLogger, compute_metrics
from mimic4_data_loader import Mimic4DataLoader
from masking import MaskingSchedule
from reward_model import RewardModel
from dataset_bundle import DatasetBundle
from reward_model_config import RewardModelConfig
from row_group_block_sampler import RowGroupBlockSampler
from reward_model_utils import unwrap_ddp

logger = logging.getLogger(__name__)


class RewardModelManager:
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

        self.bundle, self.pos_weights = self._load_datasets_and_weights()
        self.feature_index_map: Dict[str, Tuple[int, int]] = self.bundle.feature_index_map
        self.train_dataset = self.bundle.train_dataset
        self.dev_dataset = self.bundle.dev_dataset
        self.input_dim: int = self.bundle.input_dim

        if self.rank == 0:
            logger.info("Successfully loaded datasets and broadcasted positive weights.")

        self.checkpoint_manager = CheckpointManager(
            Path(self.config.CHECKPOINT_DIR), self.config.CHECKPOINT_KEEP_N
        )
        self.metrics_logger = MetricsLogger(Path(self.config.METRICS_PATH), self.config.NUM_TARGETS)

        self.model = self._build_model()
        self.optimizer = self._build_optimizer()

        self.masking_schedule: Optional[MaskingSchedule] = None
        self.train_loader: Optional[DataLoader] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None

        if self.rank == 0:
            logger.info(f"Initialized RewardModel with input_dim={self.input_dim} and DDP={self.is_ddp}")

    def _maybe_load_states(self, ckpt_state: Optional[dict]) -> None:
        if ckpt_state is None:
            return
        unwrap_ddp(self.model).load_state_dict(ckpt_state["model_state_dict"])
        self.optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt_state["scheduler_state_dict"])

    def resume_from_checkpoint(self) -> Tuple[dict, int, float]:
        latest = self.checkpoint_manager.find_latest()
        if latest is None:
            raise RuntimeError("Resume requested but no checkpoint found")
        ckpt_state: Optional[dict] = None
        if self.rank == 0:
            ckpt_state = self.checkpoint_manager.load(latest, self.feature_index_map)
        if self.is_ddp:
            obj_state: list = [ckpt_state]
            dist.broadcast_object_list(obj_state, src=0)
            ckpt_state = obj_state[0]
            dist.barrier()
        if ckpt_state is None:
            raise RuntimeError("Checkpoint state could not be loaded")
        start_epoch = ckpt_state["epoch"] + 1
        best_dev_loss = ckpt_state.get("best_dev_loss", float("inf"))
        return ckpt_state, start_epoch, best_dev_loss

    def broadcast_tensor(self, tensor: torch.Tensor, src_rank: int) -> torch.Tensor:
        """Broadcast a tensor from src_rank to all ranks if distributed is initialised."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(tensor, src=src_rank)
        return tensor

    def _load_and_broadcast_dataset(self) -> Tuple[DatasetBundle, list]:
        """Load dataset on rank 0 and broadcast pos_weights to all ranks."""
        T = self.config.NUM_TARGETS
        bundle: Optional[DatasetBundle] = None
        if self.rank == 0:
            bundle = Mimic4DataLoader(self.config).load()
            pos_weights_local = list(bundle.pos_weights)
        else:
            pos_weights_local = [0.0] * T

        weight_tensors = [torch.tensor(w, device=self.device) for w in pos_weights_local]
        if self.is_ddp:
            for t in weight_tensors:
                self.broadcast_tensor(t, src_rank=0)
            obj_list = [bundle]
            dist.broadcast_object_list(obj_list, src=0)
            bundle = obj_list[0]
            dist.barrier()

        return bundle, [float(t.item()) for t in weight_tensors]

    def _load_datasets_and_weights(self) -> Tuple[DatasetBundle, list]:
        bundle, pos_weights = self._load_and_broadcast_dataset()
        if bundle is None:
            raise RuntimeError("Dataset must be available on all ranks after broadcast")
        return bundle, pos_weights

    def _build_model(self) -> torch.nn.Module:
        model = RewardModel(
            input_dim=self.input_dim,
            layer_widths=self.config.LAYER_WIDTHS,
            dropout_rates=self.config.DROPOUT_RATES,
            activation=self.config.ACTIVATION,
            num_targets=self.config.NUM_TARGETS,
        ).to(self.device)
        if self.is_ddp:
            model = DistributedDataParallel(model, device_ids=[self.local_rank])
        return model

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2),
        )

    def _build_lr_scheduler(self, steps_per_epoch: int, start_step: int) -> torch.optim.lr_scheduler.LRScheduler:
        warmup_steps = max(self.config.LR_WARMUP_EPOCHS * max(steps_per_epoch, 1), 0)
        total_steps = max(self.config.MAX_EPOCHS * max(steps_per_epoch, 1), 1)

        if warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.0,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=self.config.LR_MIN
            )
            scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=self.config.LR_MIN
            )

        for _ in range(start_step):
            scheduler.step()
        return scheduler

    def compute_loss(
        self, logits_list: list[torch.Tensor], labels_list: list[torch.Tensor]
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        """Compute total weighted loss over T targets with dynamic NaN masking."""
        device = logits_list[0].device
        total_loss = torch.tensor(0.0, device=device)
        component_losses: list[torch.Tensor] = []

        for logits, labels, pw, w in zip(
            logits_list, labels_list, self.pos_weights, self.config.LOSS_WEIGHTS
        ):
            mask = ~torch.isnan(labels.view(-1))
            if not mask.any():
                loss_i = torch.tensor(0.0, device=device)
            else:
                pw_tensor = torch.tensor(pw, device=device)
                loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
                loss_i = loss_fn(logits.view(-1)[mask], labels.view(-1)[mask].float())
            component_losses.append(loss_i)
            total_loss = total_loss + w * loss_i

        return total_loss, component_losses

    # ---------------------------------------------------------------------------
    # Dev evaluation
    # ---------------------------------------------------------------------------

    def _eval_dev(self) -> Dict:
        """Run full dev-split evaluation on rank 0 (no DDP, no masking)."""
        T = self.config.NUM_TARGETS
        eval_model = unwrap_ddp(self.model)
        eval_model.eval()

        dataloader = DataLoader(
            self.dev_dataset,
            batch_size=self.config.BATCH_SIZE_PER_GPU,
            shuffle=False,
            num_workers=self.config.DATALOADER_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        total_loss_acc = 0.0
        component_loss_acc = [0.0] * T
        n_batches = 0

        logits_all = [[] for _ in range(T)]
        labels_all = [[] for _ in range(T)]

        with torch.no_grad():
            for batch in dataloader:
                X = batch[0].to(self.device)
                batch_labels = [batch[i + 1].to(self.device) for i in range(T)]

                logits_list = list(eval_model(X))
                loss_total, component_losses = self.compute_loss(logits_list, batch_labels)

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

    def setup_training_state(self, ckpt_state: Optional[dict], start_epoch: int) -> None:
        self.masking_schedule = self._build_masking_schedule(ckpt_state)
        self.train_loader = self._build_train_loader()
        if self.is_ddp and self.train_loader is None:
            raise RuntimeError("Train dataset unavailable for DDP training")

        steps_per_epoch = len(self.train_loader) if self.train_loader is not None else 1
        start_step = start_epoch * steps_per_epoch
        self.scheduler = self._build_lr_scheduler(steps_per_epoch, start_step)
        self._maybe_load_states(ckpt_state)
        if self.rank == 0:
            logger.info(f"Training state setup complete. Resuming from epoch {start_epoch}.")

    def _build_masking_schedule(self, ckpt_state: Optional[dict]) -> MaskingSchedule:
        if ckpt_state is not None:
            ms = ckpt_state["masking_schedule_state"]
            return MaskingSchedule(
                feature_index_map=self.feature_index_map,
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
            feature_index_map=self.feature_index_map,
            start_ratios=self.config.MASKING_START_RATIOS,
            end_ratios=self.config.MASKING_END_RATIOS,
            transition_shape=self.config.MASKING_TRANSITION_SHAPE,
            transition_midpoint_epoch=self.config.MASKING_TRANSITION_MIDPOINT_EPOCH,
            total_epochs=self.config.MAX_EPOCHS,
            num_always_visible=self.config.NUM_ALWAYS_VISIBLE_FEATURES,
            random_k_min_fraction=self.config.MASKING_RANDOM_K_MIN_FRACTION,
            random_k_max_fraction=self.config.MASKING_RANDOM_K_MAX_FRACTION,
            adversarial_k_min_fraction=self.config.MASKING_ADVERSARIAL_K_MIN_FRACTION,
            adversarial_k_max_fraction=self.config.MASKING_ADVERSARIAL_K_MAX_FRACTION,
        )

    def _build_train_loader(self) -> Optional[DataLoader]:
        if self.train_dataset is None:
            return None

        sampler = RowGroupBlockSampler(
            dataset=self.train_dataset,
            rank=self.rank,
            world_size=self.world_size if self.is_ddp else 1,
            shuffle=True,
            seed=0,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE_PER_GPU,
            sampler=sampler,
            num_workers=self.config.DATALOADER_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

    def _run_train_batch(self, X: torch.Tensor, labels: list, epoch: int) -> Tuple[float, ...]:
        """Execute one mini-batch forward/backward/step."""

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
                # forward pass to calculate gradients
                logits_list = list(self.model(X_grad))
                loss_total, _ = self.compute_loss(logits_list, labels)
                loss_total.backward()

            gradients = X_grad.grad.detach()
            self.optimizer.zero_grad()
            # mask the features with highest gradients
            X = self.masking_schedule.apply_adversarial_mask(X, gradients)
        else:
            if mode == "random":
                # choose randomly which features to mask
                X = self.masking_schedule.apply_random_mask(X)

        # forward step with masked (or unmasked) input
        logits_list = list(self.model(X))
        loss_total, component_losses = self.compute_loss(logits_list, labels)
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

        if self.rank == 0:
            logger.info(f"Starting training loop from epoch {start_epoch} to {self.config.MAX_EPOCHS}")

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
                dev_metrics = self._eval_dev()
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
                logger.info(
                    f"Epoch {epoch} completed in {time.time() - epoch_start:.2f}s | "
                    f"Dev Loss: {dev_metrics['loss_total']:.4f} | "
                    f"Masking (R/A/N): {probs[0]*100:.0f}%/{probs[1]*100:.0f}%/{probs[2]*100:.0f}%"
                )
                self.metrics_logger.append_row(row)

                current_dev_loss = dev_metrics["loss_total"]
                improved = current_dev_loss < best_dev_loss
                if improved:
                    best_dev_loss = current_dev_loss
                    epochs_without_improve = 0
                    logger.info(f"New best dev loss: {best_dev_loss:.4f} (improved). Saving best_model.pt.")
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
                    logger.info(
                        f"Dev loss did not improve. Early stopping patience: "
                        f"{epochs_without_improve}/{self.config.EARLY_STOPPING_PATIENCE}"
                    )

                should_stop = epochs_without_improve >= self.config.EARLY_STOPPING_PATIENCE
            else:
                should_stop = False

            if self.is_ddp:
                stop_tensor = torch.tensor(1 if should_stop else 0, device=self.device)
                self.broadcast_tensor(stop_tensor, src_rank=0)
                should_stop = bool(stop_tensor.item())

            if should_stop:
                if self.rank == 0:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
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

        if self.rank == 0:
            logger.info(
                f"Training finished. Last completed epoch: {last_epoch_completed}, "
                f"Best Dev Loss: {best_dev_loss:.4f}"
            )
        return last_epoch_completed, best_dev_loss
