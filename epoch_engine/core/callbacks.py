"""Module defining callback classes for use in EpochEngine training loops."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import math
import shutil
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import Trainer


class Callback(ABC):
    """Base class for all callbacks in EpochEngine.

    Callbacks can hook into different points of the training lifecycle
    to implement custom behavior like early stopping, checkpointing strategies,
    logging, or any other training-time logic.
    """

    def __new__(cls, *args, **kwargs):
        if cls is Callback:
            raise TypeError("Callback cannot be instantiated directly")
        return super().__new__(cls)

    def on_train_start(self, trainer: "Trainer") -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        pass

    def on_epoch_start(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the beginning of each epoch (``epoch`` is 0-indexed)."""
        pass

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Called at the end of each epoch after metrics are computed.

        ``metrics`` contains the full epoch metric dict with keys like
        ``'loss/train'``, ``'loss/valid'``, and any custom metric names.
        ``epoch`` is 0-indexed.
        """
        pass

    def on_batch_start(self, trainer: "Trainer", batch_idx: int) -> None:
        """Called at the beginning of each training batch."""
        pass

    def on_batch_end(
        self, trainer: "Trainer", batch_idx: int, loss: float
    ) -> None:
        """Called at the end of each training batch. ``loss`` is the scalar batch loss."""
        pass

    def on_validation_start(self, trainer: "Trainer") -> None:
        """Called at the beginning of validation."""
        pass

    def on_validation_end(
        self, trainer: "Trainer", valid_metrics: dict[str, float]
    ) -> None:
        """Called at the end of validation. ``valid_metrics`` mirrors the
        validation portion of the epoch metrics dict."""
        pass

    def on_checkpoint_save(
        self, trainer: "Trainer", checkpoint_path: Path
    ) -> None:
        """Called after a checkpoint is saved. ``checkpoint_path`` points to
        the just-written ``ckpt_epoch_N.pt`` file."""
        pass


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Sets ``should_stop=True`` after ``patience`` consecutive epochs with no
    improvement; the trainer halts at the end of that epoch.

    Args:
        monitor: Metric key to watch (e.g. ``'loss/valid'``, ``'accuracy/valid'``).
        patience: Epochs with no improvement before stopping.
        mode: ``'min'`` or ``'max'`` — direction of improvement.
        min_delta: Minimum change to qualify as an improvement.
        verbose: Whether to log counter updates and the stop event.

    Example:
        >>> early_stop = EarlyStopping(
        ...     monitor='loss/valid',
        ...     patience=5,
        ...     mode='min',
        ...     min_delta=0.001
        ... )
        >>> trainer = Trainer(..., callbacks=[early_stop])
    """

    def __init__(
        self,
        monitor: str = "loss/valid",
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 0.0,
        verbose: bool = True,
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_score: float | None = None
        self.counter = 0
        self.should_stop = False

        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Checks the monitored metric and increments the no-improvement
        counter; sets ``should_stop=True`` once ``patience`` is exceeded."""
        if self.monitor not in metrics:
            if epoch == 1 and self.verbose:
                trainer.logger.warning(
                    f"EarlyStopping: metric '{self.monitor}' not found in metrics"
                )
            return

        current_score = metrics[self.monitor]

        if self.best_score is None:
            self.best_score = current_score
        elif self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                trainer.logger.warning(
                    f"EarlyStopping: {self.counter}/{self.patience} - "
                    f"{self.monitor}={current_score:.4f} (best={self.best_score:.4f})"
                )

            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    trainer.logger.warning(
                        f"EarlyStopping: Stopping training at epoch {epoch + 1}"
                    )

    def _is_improvement(self, current: float) -> bool:
        """Returns ``True`` if ``current`` improves on ``best_score`` by at
        least ``min_delta`` in the configured direction."""
        if self.mode == "min":
            return current < (self.best_score - self.min_delta)
        else:
            return current > (self.best_score + self.min_delta)


class BestCheckpoint(Callback):
    """Save a checkpoint only when a monitored metric improves.

    Every epoch the trainer saves ``ckpt_epoch_N.pt``; this callback
    additionally writes a single ``best.pt`` to the same directory,
    overwriting it each time the monitored metric reaches a new best.

    Args:
        monitor: Metric key to watch (e.g. ``'loss/valid'``, ``'accuracy/valid'``).
        mode: ``'min'`` or ``'max'`` — direction of improvement.
        min_delta: Minimum change to qualify as an improvement.
        verbose: Whether to log when a new best is saved.

    Example:
        >>> best_ckpt = BestCheckpoint(monitor='loss/valid', mode='min')
        >>> trainer = Trainer(..., callbacks=[best_ckpt])
    """

    def __init__(
        self,
        monitor: str = "loss/valid",
        mode: str = "min",
        min_delta: float = 0.0,
        verbose: bool = True,
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score: float | None = None

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        if self.monitor not in metrics:
            if epoch == 1 and self.verbose:
                trainer.logger.warning(
                    f"BestCheckpoint: metric '{self.monitor}' not found in metrics"
                )
            return

        current = metrics[self.monitor]
        if self.best_score is None or self._is_improvement(current):
            self.best_score = current
            self._new_best = True
        else:
            self._new_best = False

    def on_checkpoint_save(
        self, trainer: "Trainer", checkpoint_path: Path
    ) -> None:
        """Copies ``checkpoint_path`` to ``best.pt`` if a new best was
        recorded in ``on_epoch_end``; otherwise does nothing."""
        if not getattr(self, "_new_best", False):
            return
        best_path = checkpoint_path.parent / "best.pt"
        shutil.copy(checkpoint_path, best_path)
        if self.verbose:
            trainer.logger.info(
                f"BestCheckpoint: new best {self.monitor}={self.best_score:.4f} "
                f"— saved to {best_path}"
            )

    def _is_improvement(self, current: float) -> bool:
        """Returns ``True`` if ``current`` improves on ``best_score`` by at
        least ``min_delta`` in the configured direction."""
        if self.mode == "min":
            return current < (self.best_score - self.min_delta)
        return current > (self.best_score + self.min_delta)


class CheckpointPruner(Callback):
    """Delete old epoch checkpoints, keeping only the most recent N.

    After each checkpoint save, sorts all ``ckpt_epoch_*.pt`` files by epoch
    number and removes any beyond the last ``keep_last_n``. ``best.pt`` is
    never touched.

    Args:
        keep_last_n: Number of most recent epoch checkpoints to keep.

    Example:
        >>> pruner = CheckpointPruner(keep_last_n=2)
        >>> trainer = Trainer(..., callbacks=[BestCheckpoint(), pruner])
    """

    def __init__(self, keep_last_n: int = 1):
        if keep_last_n < 1:
            raise ValueError(f"keep_last_n must be >= 1, got {keep_last_n}")
        self.keep_last_n = keep_last_n

    def on_checkpoint_save(
        self, trainer: "Trainer", checkpoint_path: Path
    ) -> None:
        """Deletes the oldest ``ckpt_epoch_*.pt`` files, keeping the
        ``keep_last_n`` most recent. ``best.pt`` is never removed."""
        epoch_ckpts = sorted(
            checkpoint_path.parent.glob("ckpt_epoch_*.pt"),
            key=lambda f: int(f.stem.split("_")[-1]),
        )
        for old_ckpt in epoch_ckpts[: -self.keep_last_n]:
            old_ckpt.unlink()


class NanCallback(Callback):
    """Stop training if any metric value is NaN.

    Sets ``should_stop=True`` at the end of any epoch where NaN is detected;
    the trainer reads this flag after ``on_epoch_end`` and halts.

    Args:
        monitor: Specific metric keys to check. If ``None``, all numeric
            values in the epoch metrics dict are checked. Defaults to ``None``.
        verbose: Whether to log a warning message when NaN is detected.
            Defaults to ``True``.

    Example:
        >>> nan_cb = NanCallback()
        >>> trainer = Trainer(..., callbacks=[nan_cb])
    """

    def __init__(
        self,
        monitor: list[str] | None = None,
        verbose: bool = True,
    ):
        self.monitor = monitor
        self.verbose = verbose
        self.should_stop = False

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Scans the monitored metrics for NaN and sets ``should_stop=True``
        if any are found."""
        candidates = (
            {k: metrics[k] for k in self.monitor if k in metrics}
            if self.monitor is not None
            else metrics
        )

        nan_keys = [
            k
            for k, v in candidates.items()
            if isinstance(v, (int, float)) and math.isnan(v)
        ]

        if nan_keys:
            self.should_stop = True
            if self.verbose:
                trainer.logger.warning(
                    f"NanCallback: NaN detected in {nan_keys} at epoch {epoch + 1}. "
                    "Stopping training."
                )
