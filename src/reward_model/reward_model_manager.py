"""Training utilities and manager class for the CDSS-ML reward model.

Houses data loading, masking curriculum, model/optimizer construction, and the
RewardModelManager class. The entry point lives in ``reward_model_main.py``.
"""
import math
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from checkpoint_manager import CheckpointManager
from metrics import MetricsLogger, compute_metrics
from mimic4_data_loader import Mimic4DataLoader
from masking import MaskingSchedule
from reward_model import RewardModel
from dataset_bundle import DatasetBundle
from reward_model_config import RewardModelConfig
from row_group_block_sampler import RowGroupBlockSampler

logger = logging.getLogger(__name__)


class RewardModelManager:
    def __init__(
        self,
        config: RewardModelConfig,
        accelerator: Accelerator,
    ):
        self.config = config
        self.accelerator = accelerator

        self.bundle, self.pos_weights = self._load_datasets_and_weights()
        self.feature_index_map: Dict[str, Tuple[int, int]] = self.bundle.feature_index_map
        self.train_dataset = self.bundle.train_dataset
        self.dev_dataset = self.bundle.dev_dataset
        self.label_names: list = self.bundle.label_names

        logger.info("Successfully loaded datasets.")

        self.checkpoint_manager = CheckpointManager(
            Path(self.config.CHECKPOINT_DIR)
        )
        self.metrics_logger = MetricsLogger(Path(self.config.METRICS_PATH), self.label_names)

        self.model = self._build_model()
        self.optimizer = self._build_optimizer()

        self.masking_schedule: Optional[MaskingSchedule] = None
        self.train_loader: Optional[DataLoader] = self._build_train_loader()
        self.warmup_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.plateau_scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None
        self.warmup_total_steps: int = 0
        self.warmup_steps_taken: int = 0
        self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
        )

        logger.info(f"Initialized RewardModel with input_dim={self.config.INPUT_DIM}")

    def resume_from_checkpoint(self) -> Tuple[dict, int, float]:
        latest = self.checkpoint_manager.find_latest()
        if latest is None:
            raise RuntimeError("Resume requested but no checkpoint found")
        ckpt_state = self.checkpoint_manager.load(latest)
        if ckpt_state is None:
            raise RuntimeError("Checkpoint state could not be loaded")
        start_epoch = ckpt_state["epoch"] + 1
        best_dev_loss = ckpt_state.get("best_dev_loss", float("inf"))
        return ckpt_state, start_epoch, best_dev_loss

    def _load_datasets_and_weights(self) -> Tuple[DatasetBundle, list]:
        bundle = Mimic4DataLoader(self.config).load()
        pos_weights = [math.sqrt(w) for w in bundle.pos_weights]
        logger.info(
            "pos_weight after sqrt scaling: y1=%.4f  y2=%.4f",
            pos_weights[0], pos_weights[1]
        )
        return bundle, pos_weights

    def _build_model(self) -> torch.nn.Module:
        return RewardModel(self.config)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2),
        )

    def _build_warmup_scheduler(self, steps_per_epoch: int) -> torch.optim.lr_scheduler.LRScheduler:
        warmup_steps = max(self.config.LR_WARMUP_EPOCHS * max(steps_per_epoch, 1), 0)
        return torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

    def _build_plateau_scheduler(self) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.LR_PLATEAU_FACTOR,
            patience=self.config.LR_PLATEAU_PATIENCE,
            min_lr=self.config.LR_MIN
        )

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

        #For debugging purpose
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error("NaN/Inf total_loss=%.4f", float(total_loss))
            for i, (logits, labels, loss_i) in enumerate(
                    zip(logits_list, labels_list, component_losses)
            ):
                logger.error(
                    "  target_%d: loss=%.4f "
                    "logits min=%.4f max=%.4f has_nan=%s | "
                    "labels min=%.4f max=%.4f has_nan=%s",
                    i,
                    float(loss_i),
                    float(logits.min()), float(logits.max()),
                    str(torch.isnan(logits).any().item()),
                    float(labels[~torch.isnan(labels)].min())
                    if (~torch.isnan(labels)).any() else float("nan"),
                    float(labels[~torch.isnan(labels)].max())
                    if (~torch.isnan(labels)).any() else float("nan"),
                    str(torch.isnan(labels).all().item()),
                )

        return total_loss, component_losses

    # ---------------------------------------------------------------------------
    # Dev evaluation
    # ---------------------------------------------------------------------------

    def _eval_dev(self) -> Dict:
        """Run full dev-split evaluation (no masking)."""
        num_targets = self.config.NUM_TARGETS
        self.model.eval()

        dataloader = DataLoader(
            self.dev_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        total_loss_acc = 0.0
        component_loss_acc = [0.0] * num_targets
        n_batches = 0

        logits_all = [[] for _ in range(num_targets)]
        labels_all = [[] for _ in range(num_targets)]

        with torch.no_grad():
            for batch in dataloader:

                X = batch[0].to(self.accelerator.device).float().contiguous()
                pad_size = (16 - X.shape[1] % 16) % 16
                if pad_size > 0:
                    X = torch.nn.functional.pad(X, (0, pad_size)).contiguous()

                batch_labels = [batch[i + 1].to(self.accelerator.device).float().contiguous() for i in range(num_targets)]

                logits_list = list(self.model(X))
                loss_total, component_losses = self.compute_loss(logits_list, batch_labels)

                total_loss_acc += float(loss_total.detach().item())
                for i, cl in enumerate(component_losses):
                    component_loss_acc[i] += float(cl.detach().item())
                n_batches += 1

                for i in range(num_targets):
                    logits_all[i].append(logits_list[i].detach())
                    labels_all[i].append(batch_labels[i].detach())

        nan_result: Dict = {"loss_total": float("nan")}
        for i in range(num_targets):
            nan_result[f"loss_target_{i}"] = float("nan")
            nan_result[f"auroc_target_{i}"] = float("nan")
            nan_result[f"auprc_target_{i}"] = float("nan")
            nan_result[f"ece_target_{i}"] = float("nan")
        if n_batches == 0:
            return nan_result

        logits_cat = [torch.cat(logits_all[i], dim=0) for i in range(num_targets)]
        labels_cat = [torch.cat(labels_all[i], dim=0) for i in range(num_targets)]

        per_target = compute_metrics(logits_cat, labels_cat, masked=False)

        result: Dict = {"loss_total": total_loss_acc / n_batches}
        for i in range(num_targets):
            result[f"loss_target_{i}"] = component_loss_acc[i] / n_batches
            target_metrics = per_target[i]
            result[f"auroc_target_{i}"] = target_metrics["auroc"]
            result[f"auprc_target_{i}"] = target_metrics["auprc"]
            result[f"ece_target_{i}"] = target_metrics["ece"]
        return result

    def setup_training_state(self, ckpt_state: Optional[dict], start_epoch: int) -> None:
        self.masking_schedule = self._build_masking_schedule()
        steps_per_epoch = len(self.train_loader) if self.train_loader is not None else 1
        self.warmup_total_steps = max(self.config.LR_WARMUP_EPOCHS * max(steps_per_epoch, 1), 0)
        self.warmup_scheduler = self._build_warmup_scheduler(steps_per_epoch)
        self.plateau_scheduler = self._build_plateau_scheduler()
        self.warmup_steps_taken = 0
        if ckpt_state is not None:
            clean_model = self.accelerator.unwrap_model(self.model)
            clean_model.load_state_dict(ckpt_state["model_state_dict"])
            self.optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
            self.warmup_steps_taken = self.warmup_total_steps
            if ckpt_state.get("plateau_scheduler_state_dict") is not None:
                self.plateau_scheduler.load_state_dict(ckpt_state["plateau_scheduler_state_dict"])
        if self.train_loader is not None:
            logger.info(
                "Train loader: %d batches/epoch, batch_size=%d",
                len(self.train_loader),
                self.config.BATCH_SIZE,
            )
        logger.info(f"Training state setup complete. Resuming from epoch {start_epoch}.")

    def _build_masking_schedule(self) -> MaskingSchedule:
        return MaskingSchedule(self.config, self.feature_index_map)

    def _build_train_loader(self) -> Optional[DataLoader]:
        if self.train_dataset is None:
            return None

        sampler = RowGroupBlockSampler(dataset=self.train_dataset, shuffle=True, seed=0)
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
        )

    def _run_train_batch(self, X: torch.Tensor, labels: list, epoch: int) -> Tuple[float, ...]:
        """Execute one mini-batch forward/backward/step."""

        pad_size = (16 - X.shape[1] % 16) % 16
        if pad_size > 0:
            X = torch.nn.functional.pad(X, (0, pad_size)).contiguous()

        mode = self.masking_schedule.sample_mode(epoch)

        #logger.info("_run_train_batch: mode=%s", mode)

        self.optimizer.zero_grad()

        if mode == "adversarial":
            X_grad = X.clone().requires_grad_(True)
            with self.accelerator.no_sync(self.model):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits_list = list(self.model(X_grad))
                    loss_total, _ = self.compute_loss(logits_list, labels)
                self.accelerator.backward(loss_total)
        #if mode == "adversarial":
            #X_grad = X.clone().requires_grad_(True)
            #logger.info("_run_train_batch: adversarial forward")
            #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            #    logits_list = list(self.model(X_grad))
            #    loss_total, _ = self.compute_loss(logits_list, labels)
            #logger.info("_run_train_batch: adversarial backward")
            #self.accelerator.backward(loss_total)

            gradients = X_grad.grad.detach()
            self.optimizer.zero_grad()
            #logger.info("_run_train_batch: applying adversarial mask")
            X = self.masking_schedule.apply_adversarial_mask(X, gradients)
        else:
            if mode == "random":
                X = self.masking_schedule.apply_random_mask(X)

        # forward step with masked (or unmasked) input
        #logger.info("_run_train_batch: main forward")
        X = X.contiguous()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_list = list(self.model(X))
            loss_total, component_losses = self.compute_loss(logits_list, labels)
        #logger.info("_run_train_batch: main backward")
        self.accelerator.backward(loss_total)
        #logger.info("_run_train_batch: backward done")
        #for debugging
        layer_norms = {}
        total_sq = 0.0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2).item()
                layer_norms[name] = norm
                total_sq += math.pow(norm,2)
        grad_norm = math.sqrt(total_sq)
        logger.debug("grad_norm=%.4f", grad_norm)
        # Log top 5 layers by gradient norm every batch
        top_layers = sorted(
            layer_norms.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(
            "grad_norm=%.2f | top layers: %s",
            grad_norm,
            " | ".join(f"{n}={v:.2f}" for n, v in top_layers))

        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return (float(loss_total.detach().item()), grad_norm) + tuple(
            float(l.detach().item()) for l in component_losses
        )

    def train_epochs(self, start_epoch: int, best_dev_loss: float) -> Tuple[int, float]:
        """Run the epoch loop and return the last completed epoch and best dev loss."""
        if (
            self.masking_schedule is None
            or self.warmup_scheduler is None
            or self.plateau_scheduler is None
        ):
            raise RuntimeError("Training state must be set up before training epochs")

        num_targets = self.config.NUM_TARGETS
        epochs_without_improve = 0
        last_epoch_completed = start_epoch - 1

        if self.accelerator.is_main_process:
            logger.info(f"Starting training loop from epoch {start_epoch} to {self.config.MAX_EPOCHS}")
            probs_start = self.masking_schedule.get_mode_probabilities(start_epoch)
            logger.info(
                "Masking schedule at epoch %d: Random=%.0f%%  "
                "Adversarial=%.0f%%  None=%.0f%%",
                start_epoch,
                probs_start[0] * 100,
                probs_start[1] * 100,
                probs_start[2] * 100,
            )

        for epoch in range(start_epoch, self.config.MAX_EPOCHS):
            if self.train_loader is None:
                break

            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            epoch_start = time.time()

            batch_losses = []
            if self.accelerator.is_main_process:
                logger.info("Epoch %d — starting data iteration", epoch)
            for batch_idx, batch in enumerate(self.train_loader):
                if self.accelerator.is_main_process:
                    logger.info("Epoch %d | batch %d — loaded from DataLoader", epoch, batch_idx)

                X = batch[0].float().contiguous()
                labels = [batch[i + 1].float().contiguous() for i in range(num_targets)]

                if self.accelerator.is_main_process:
                    logger.info("Epoch %d | batch %d — batch tensors ready", epoch, batch_idx)

                loss_vals = self._run_train_batch(X, labels, epoch)
                if self.accelerator.is_main_process:
                    logger.info("Epoch %d | batch %d — train batch done", epoch, batch_idx)
                total_loss = loss_vals[0]
                grad_norm = loss_vals[1]
                #for debugging
                if self.accelerator.is_main_process and batch_idx % 1 == 0:  # every batch
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        "Epoch %d | batch %d/%d | loss=%.4f | "
                        "grad_norm=%.4f | lr=%.2e",
                        epoch, batch_idx, len(self.train_loader),
                        total_loss, grad_norm, lr,
                    )
                if (
                    epoch < self.config.LR_WARMUP_EPOCHS
                    and self.warmup_steps_taken < self.warmup_total_steps
                ):
                    self.warmup_scheduler.step()
                    self.warmup_steps_taken += 1
                batch_losses.append(loss_vals[0])

                # Log every 50 batches
                if self.accelerator.is_main_process and batch_idx % 50 == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        "Epoch %d | batch %d/%d | loss=%.4f | lr=%.2e",
                        epoch,
                        batch_idx,
                        len(self.train_loader),
                        loss_vals[0],
                        lr,
                    )
                    # Early NaN detection
                    if torch.isnan(torch.tensor(loss_vals[0])):
                        logger.error(
                            "NaN loss at epoch %d batch %d — "
                            "stopping training.",
                            epoch, batch_idx,
                        )
                        break

            probs = self.masking_schedule.get_mode_probabilities(epoch)
            dev_metrics = self._eval_dev()
            self.model.train()

            row: Dict = {
                "epoch": epoch,
                "time(seconds)": int(time.time() - epoch_start),
                "loss_total": dev_metrics["loss_total"],
            }
            for i, name in enumerate(self.label_names):
                row[f"loss_{name}"] = dev_metrics[f"loss_target_{i}"]
                row[f"auroc_{name}"] = dev_metrics[f"auroc_target_{i}"]
                row[f"auprc_{name}"] = dev_metrics[f"auprc_target_{i}"]
                row[f"ece_{name}"] = dev_metrics[f"ece_target_{i}"]
            current_dev_loss = dev_metrics["loss_total"]
            if epoch >= self.config.LR_WARMUP_EPOCHS:
                self.plateau_scheduler.step(current_dev_loss)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                logger.info(
                    "Epoch %d | %.1fs | Loss=%.4f | "
                    "Y1: AUROC=%.4f AUPRC=%.4f ECE=%.4f | "
                    "Y2: AUROC=%.4f AUPRC=%.4f ECE=%.4f | "
                    "Mask R/A/N=%.0f%%/%.0f%%/%.0f%%",
                    epoch,
                    time.time() - epoch_start,
                    dev_metrics["loss_total"],
                    dev_metrics["auroc_target_0"],
                    dev_metrics["auprc_target_0"],
                    dev_metrics["ece_target_0"],
                    dev_metrics["auroc_target_1"],
                    dev_metrics["auprc_target_1"],
                    dev_metrics["ece_target_1"],
                    probs[0] * 100,
                    probs[1] * 100,
                    probs[2] * 100,
                )
                improved = current_dev_loss < best_dev_loss
                if improved:
                    best_dev_loss = current_dev_loss
                    epochs_without_improve = 0
                    logger.info(f"New best dev loss: {best_dev_loss:.4f} (improved). Saving best_model.pt.")
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    self.checkpoint_manager.save_train_checkpoint(
                        unwrapped_model,
                        self.optimizer,
                        epoch,
                        best_dev_loss,
                        plateau_scheduler_state=self.plateau_scheduler.state_dict(),
                    )
                    self.checkpoint_manager.save_best_model(
                        unwrapped_model,
                        epoch,
                        best_dev_loss,
                    )
                else:
                    epochs_without_improve += 1
                    logger.info(
                        f"Dev loss did not improve. Early stopping patience: "
                        f"{epochs_without_improve}/{self.config.EARLY_STOPPING_PATIENCE}"
                    )
                self.metrics_logger.append_row(row)

            last_epoch_completed = epoch
            self.accelerator.wait_for_everyone()
            should_stop = False
            if self.accelerator.is_main_process:
                should_stop = epochs_without_improve >= self.config.EARLY_STOPPING_PATIENCE
                if should_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
            should_stop_tensor = torch.tensor(
                int(should_stop),
                device=self.accelerator.device,
            )
            should_stop_tensor = self.accelerator.reduce(should_stop_tensor, reduction="max")
            should_stop = bool(should_stop_tensor.item())
            if should_stop:
                break

        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            self.checkpoint_manager.save_train_checkpoint(
                unwrapped_model,
                self.optimizer,
                last_epoch_completed if last_epoch_completed >= 0 else 0,
                best_dev_loss,
                plateau_scheduler_state=self.plateau_scheduler.state_dict(),
            )
            logger.info(
                f"Training finished. Last completed epoch: {last_epoch_completed}, "
                f"Best Dev Loss: {best_dev_loss:.4f}"
            )
        self.accelerator.wait_for_everyone()
        return last_epoch_completed, best_dev_loss
