from pathlib import Path
from typing import Sequence

import lightning as L
import torch


class EpochCheckpointingCallback(L.Callback):
    def __init__(
        self,
        dirpath: str,
        monitor_candidates: Sequence[str] = ("eval/part_acc", "val/loss"),
        every_n_epochs: int = 50,
        save_weights_only: bool = False,
    ):
        self.dirpath = Path(dirpath)
        self.monitor_candidates = tuple(monitor_candidates)
        self.every_n_epochs = every_n_epochs
        self.save_weights_only = save_weights_only

        self.best_monitor: str | None = None
        self.best_mode: str | None = None
        self.best_score: float | None = None
        self._last_saved_epoch: int | None = None

    @staticmethod
    def _mode_for_metric(metric_name: str) -> str:
        lowered = metric_name.lower()
        if "acc" in lowered or "f1" in lowered or "precision" in lowered or "recall" in lowered:
            return "max"
        return "min"

    def _get_monitor_value(self, trainer: L.Trainer) -> tuple[str, float] | None:
        callback_metrics = trainer.callback_metrics
        for metric_name in self.monitor_candidates:
            metric_value = callback_metrics.get(metric_name)
            if metric_value is None:
                continue

            if isinstance(metric_value, torch.Tensor):
                if metric_value.numel() != 1:
                    continue
                metric_value = metric_value.detach().cpu().item()

            return metric_name, float(metric_value)

        return None

    def _is_better(self, current: float) -> bool:
        if self.best_score is None or self.best_mode is None:
            return True
        if self.best_mode == "max":
            return current > self.best_score
        return current < self.best_score

    def _save_checkpoint(self, trainer: L.Trainer, filename: str):
        trainer.save_checkpoint(
            str(self.dirpath / filename),
            weights_only=self.save_weights_only,
        )

    def _save_epoch_checkpoints(self, trainer: L.Trainer):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch + 1
        if self._last_saved_epoch == epoch:
            return

        self.dirpath.mkdir(parents=True, exist_ok=True)

        self._save_checkpoint(trainer, "last.ckpt")

        if epoch % self.every_n_epochs == 0:
            self._save_checkpoint(trainer, f"epoch-{epoch}.ckpt")

        monitor = self._get_monitor_value(trainer)
        if monitor is not None:
            metric_name, metric_value = monitor
            metric_mode = self._mode_for_metric(metric_name)

            if self.best_monitor != metric_name:
                self.best_monitor = metric_name
                self.best_mode = metric_mode
                self.best_score = None

            if self._is_better(metric_value):
                self.best_score = metric_value
                self._save_checkpoint(trainer, "best.ckpt")

        self._last_saved_epoch = epoch

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._save_epoch_checkpoints(trainer)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        num_val_batches = trainer.num_val_batches
        if isinstance(num_val_batches, list):
            num_val_batches = sum(num_val_batches)

        if num_val_batches == 0:
            self._save_epoch_checkpoints(trainer)
