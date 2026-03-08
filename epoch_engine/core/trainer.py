"""Main Trainer class that orchestrates PyTorch training and validation loops."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

from collections.abc import Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .callbacks import Callback
from .configs import MetricConfig
from .logger import TrainerLogger
from .managers import DeviceManager, RunManager
from .metrics import (
    MetricsLogger,
    MetricsPlotter,
    MetricsTracker,
    format_metrics,
)


class Trainer:
    """Orchestrates PyTorch model training, validation, checkpointing, and evaluation.

    Handles the full training lifecycle: device placement, AMP, per-epoch
    train/validation loops, metric tracking and logging, checkpoint saving,
    callback dispatch, and optional metric plot generation.

    Attributes:
        model (nn.Module): Model being trained (moved to the detected device).
        criterion (Callable): Loss function.
        optimizer (Optimizer): Optimizer instance.
        train_loader (DataLoader): DataLoader for the training split.
        valid_loader (DataLoader): DataLoader for the validation split.
        test_loader (DataLoader | None): Optional DataLoader for the test split.
        scheduler (LRScheduler | ReduceLROnPlateau | None): LR scheduler, or
            ``None`` if not provided.
        scheduler_level (str | None): ``'epoch'`` or ``'batch'``, or ``None``
            when no scheduler is set.
        last_epoch (int): Most recently completed epoch number (0 before any run).
        metrics_tracker (MetricsTracker): Accumulates and computes epoch metrics.
        metrics_plotter (MetricsPlotter): Generates PNG plots from metric history.
        logger (TrainerLogger | None): Console logger; set when ``run()`` is called.
        device_mgr (DeviceManager): Manages device selection and AMP.
        run_mgr (RunManager): Manages checkpoint I/O and run directory structure.
        extra_metrics (bool): ``True`` when custom metric functions are registered.
        run_id (str | None): Current run identifier; set when ``run()`` is called.
        metrics_to_plot (list[str]): Base metric names for which PNG plots are
            generated (always includes ``'loss'``).
        callbacks (list[Callback]): Registered callback instances.
        interrupted (bool): ``True`` if a callback set ``should_stop=True`` during
            the last ``run()`` call.
        task (str): ``'classification'`` or ``'regression'`` — controls target
            dtype casting and how predictions are extracted for custom metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        optimizer: Optimizer,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader | None = None,
        scheduler: LRScheduler | ReduceLROnPlateau | None = None,
        scheduler_level: str = "epoch",
        metrics: list[MetricConfig] | dict[str, Callable] | None = None,
        callbacks: list[Callback] | None = None,
        enable_amp: bool = False,
        task: str = "classification",
    ) -> None:
        """
        Args:
            model (nn.Module): Model to train. Moved to the auto-detected device.
            criterion (Callable): Loss function (e.g. ``nn.CrossEntropyLoss()``).
            optimizer (Optimizer): Optimizer (e.g. ``torch.optim.Adam``).
            train_loader (DataLoader): DataLoader for training batches.
            valid_loader (DataLoader): DataLoader for validation batches.
            test_loader (DataLoader, optional): DataLoader for test batches,
                used by :meth:`evaluate`. Defaults to ``None``.
            scheduler (LRScheduler | ReduceLROnPlateau, optional): LR
                scheduler. Defaults to ``None``.
            scheduler_level (str, optional): When to step the scheduler —
                ``'epoch'`` or ``'batch'``. Defaults to ``'epoch'``.
            metrics (list[MetricConfig] | dict[str, Callable], optional):
                Custom metrics to track. A ``dict`` enables plotting for all;
                a ``list[MetricConfig]`` allows per-metric plot control.
                Defaults to ``None``.
            callbacks (list[Callback], optional): Callbacks executed at
                training lifecycle hooks. Defaults to ``None``.
            enable_amp (bool, optional): Enable automatic mixed precision
                (CUDA only). Defaults to ``False``.
            task (str, optional): ``'classification'`` casts targets to
                ``.long()`` and extracts predictions via ``argmax``.
                ``'regression'`` casts targets to ``.float()`` and uses raw
                outputs via ``squeeze(-1)``. Defaults to ``'classification'``.

        Raises:
            TypeError: If ``model``, ``optimizer``, or any loader is not the
                expected type, or if ``criterion`` is not callable.
            ValueError: If ``scheduler_level`` is not ``'epoch'`` or
                ``'batch'``, or ``task`` is not ``'classification'`` or
                ``'regression'``.
        """
        # Determining device automatically and handling AMP (if enabled)
        self.device_mgr = DeviceManager(enable_amp=enable_amp)
        if self.device_mgr.enable_amp:
            self.scaler = torch.amp.GradScaler()

        # Validating model and moving it to device
        self._validate_arg(model, nn.Module, "model")
        self.model = self.device_mgr.move_to_device(model)

        # Validating and setting loss function
        if not callable(criterion):
            raise TypeError("`criterion` must be callable")
        self.criterion = criterion

        # Validating and setting optimizer
        self._validate_arg(optimizer, Optimizer, "optimizer")
        self.optimizer = optimizer

        # Setting dataloaders
        for loader, loader_name in [
            (train_loader, "train_loader"),
            (valid_loader, "valid_loader"),
        ]:
            self._validate_arg(loader, DataLoader, loader_name)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        if test_loader is not None:
            self._validate_arg(test_loader, DataLoader, "test_loader")
            self.test_loader = test_loader

        # Setting task type (classification or regression)
        if task not in ("classification", "regression"):
            raise ValueError(
                "Invalid task: must be 'classification' or 'regression'"
            )
        self.task = task

        # Setting scheduler-specific attributes
        if scheduler_level not in ("epoch", "batch"):
            raise ValueError(
                "Invalid scheduler_level: must be 'epoch' or 'batch'"
            )
        self.scheduler = scheduler
        self.scheduler_level = (
            scheduler_level if scheduler is not None else None
        )

        # Setting epoch indicator
        self.last_epoch = 0

        # Setting tracker/logger for metrics and run manager for checkpoints
        self.metrics_tracker = MetricsTracker()
        self.metrics_plotter = MetricsPlotter()
        self.run_mgr = RunManager()

        self.extra_metrics = False  # Flag if extra metrics are registered
        self.metrics_to_plot = [
            "loss"
        ]  # Initializing metrics to plot (plotting "loss" by default)

        if metrics is not None:
            self._register_metrics(metrics)

        self.callbacks = callbacks or []
        self.interrupted = False
        self.run_id = None
        self.logger = None  # Initialized in `run` once `run_id` is set

    @staticmethod
    def _validate_arg(value, expected_type, name: str) -> None:
        """Raises TypeError if `value` is not an instance of `expected_type`."""
        if not isinstance(value, expected_type):
            raise TypeError(
                f"`{name}` must be an instance of `{expected_type.__name__}`"
            )

    def _any_callback_stopped(self) -> bool:
        """Returns True if any callback has requested an early stop."""
        return any(getattr(cb, "should_stop", False) for cb in self.callbacks)

    def _call_callbacks(self, hook: str, **kwargs):
        """Call a specific hook on all callbacks."""
        for callback in self.callbacks:
            method = getattr(callback, hook, None)
            if method:
                method(trainer=self, **kwargs)

    def reset_scheduler(self) -> None:
        """Removes the scheduler and sets ``scheduler_level`` to ``None``."""
        self.scheduler = None
        self.scheduler_level = None

    def __call__(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs a forward pass and computes the loss.

        For ``'classification'``, targets are cast to ``.long()``. For
        ``'regression'``, outputs are squeezed via ``squeeze(-1)`` and targets
        cast to ``.float()``.

        Args:
            x_batch (torch.Tensor): Batch of input examples.
            y_batch (torch.Tensor): Batch of targets.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(loss, outputs)`` where
                ``outputs`` are the raw model logits/predictions.
        """
        outputs = self.model(x_batch)
        if self.task == "classification":
            loss = self.criterion(outputs, y_batch.long())
        # For regression, squeezing outputs to ensure they have the same shape as targets (e.g. [B, 1] -> [B])
        else:
            loss = self.criterion(outputs.squeeze(-1), y_batch.float())

        return loss, outputs

    def run(
        self,
        epochs: int,
        run_id: str | None = None,
        seed: int = 42,
        enable_tqdm: bool = True,
        clip_grad_norm: float | None = None,
        resume_from_best: bool = False,
    ) -> None:
        """Runs the training and validation loops for the given number of epochs.

        Starts a new run (generating a fresh ``run_id``) or resumes an existing
        one. On resume, model and optimizer states are loaded from the last saved
        checkpoint (or ``best.pt`` if ``resume_from_best=True``). PNG metric
        plots are always generated in a ``finally`` block, even if training is
        interrupted.

        Args:
            epochs (int): Number of epochs to train.
            run_id (str | None, optional): Run ID to resume. If ``None``,
                reuses ``self.run_id`` if set, otherwise starts a new run.
                Defaults to ``None``.
            seed (int, optional): Random seed for reproducibility. Defaults
                to ``42``.
            enable_tqdm (bool, optional): Show tqdm progress bars. Defaults
                to ``True``.
            clip_grad_norm (float | None, optional): Maximum gradient norm
                for clipping. Defaults to ``None``.
            resume_from_best (bool, optional): If ``True`` and ``best.pt``
                exists, resume from it instead of the last epoch checkpoint.
                Requires :class:`callbacks.BestCheckpoint` to have been used previously.
                Defaults to ``False``.
        """
        run_info = self.run_mgr.init_run(run_id=run_id or self.run_id)
        self.run_id = run_info.run_id
        ckpts_path = run_info.ckpts_path
        # Instantiating a logger for logging events related to Trainer
        self.logger = TrainerLogger(enable_tqdm=enable_tqdm)

        # Checking if there have been training runs before
        if run_info.last_logged_epoch > 0:
            best_path = ckpts_path / "best.pt"
            if resume_from_best and best_path.exists():
                ckpt_path = best_path
            else:
                if resume_from_best:
                    self.logger.warning(
                        "resume_from_best=True but no best.pt found — "
                        "falling back to last epoch checkpoint"
                    )
                ckpt_path = (
                    ckpts_path / f"ckpt_epoch_{run_info.last_logged_epoch}.pt"
                )

            checkpoint = self.run_mgr.load_checkpoint(path=ckpt_path)

            # Loading state dicts for model and optimizer as well as last trained epoch
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.last_epoch = checkpoint["epoch"]

            self.logger.info(
                f"Resuming run 'run_id={run_info.run_id}' from epoch {self.last_epoch}"
            )
        else:
            self.logger.info(
                f"Starting a new run for 'run_id={run_info.run_id}'"
            )

        # Setting seed
        if seed:
            self.device_mgr.set_seed(seed=seed)

        # Setting max norm for gradient clipping
        self.clip_grad_norm = clip_grad_norm

        # Computing the number of total epochs to be trained (useful in case of additional trainings)
        total_epochs = epochs + self.last_epoch
        # Running training/validation loops
        try:
            # Using custom context manager to make sure that history is not lost due to errors in loops below
            with MetricsLogger(
                run_id=self.run_id,
                truncate_after_epoch=(
                    self.last_epoch if self.last_epoch > 0 else None
                ),
            ) as metrics_logger:
                for epoch in range(self.last_epoch, total_epochs):
                    self._run_one_epoch(
                        epoch=epoch,
                        epochs=total_epochs,
                        enable_tqdm=enable_tqdm,
                        train=True,
                    )
                    self._run_one_epoch(
                        epoch=epoch,
                        epochs=total_epochs,
                        enable_tqdm=enable_tqdm,
                        train=False,
                    )
                    # Incrementing last epoch after successful training/validation
                    self.last_epoch = epoch + 1

                    # Computing and displaying epoch-level metrics
                    self.metrics = self.metrics_tracker.compute_metrics()
                    format_metrics(
                        epoch=self.last_epoch,
                        metrics=self.metrics,
                        use_tqdm=enable_tqdm,
                    )
                    self.metrics["epoch"] = epoch + 1

                    # Logging computed metrics
                    metrics_logger.log_metrics(metrics=self.metrics)

                    # Call epoch end callbacks
                    self._call_callbacks(
                        hook="on_epoch_end",
                        epoch=self.last_epoch,
                        metrics=self.metrics,
                    )

                    # Saving the checkpoint
                    ckpt_path = ckpts_path / f"ckpt_epoch_{self.last_epoch}.pt"
                    self.run_mgr.save_checkpoint(
                        path=ckpt_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=self.last_epoch,
                    )
                    self._call_callbacks(
                        hook="on_checkpoint_save", checkpoint_path=ckpt_path
                    )
                    if self._any_callback_stopped():
                        self.interrupted = True
                        break

                    # Resetting metrics counters after successful training/validation
                    self.metrics_tracker.reset()
        # Making sure that in case of error metric plots are generated for completed epoch runs
        finally:
            for metric in self.metrics_to_plot:
                self.metrics_plotter.create_plot(
                    metric_name=metric, run_id=self.run_id
                )
        self.logger.success(
            f"Finished run 'run_id={run_info.run_id}' at epoch {self.last_epoch}"
        )

    def _run_one_epoch(
        self, epoch: int, epochs: int, enable_tqdm: bool, train: bool
    ) -> None:
        """Processes all batches for one epoch (training or validation)."""
        split = "train" if train else "valid"
        loader = self.train_loader if train else self.valid_loader
        desc = f"Epoch {epoch + 1}/{epochs} [{'Training' if train else 'Validation'}]"

        self.model.train() if train else self.model.eval()
        with tqdm(
            total=len(loader),
            desc=desc,
            position=0,
            leave=True,
            unit="batches",
            disable=not enable_tqdm,
        ) as pbar:
            if train:
                pbar.set_postfix({"lr": self.optimizer.param_groups[0]["lr"]})
            with torch.set_grad_enabled(train):
                for batch in loader:
                    self._process_batch(batch=batch, split=split)
                    pbar.update(1)
        if train and self.scheduler and self.scheduler_level == "epoch":
            self.scheduler.step()

    def _amp_step(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward + backward pass using AMP."""
        with torch.amp.autocast(device_type=self.device_mgr.device.type):
            loss, outputs = self(x_batch=x_batch, y_batch=y_batch)
        self.scaler.scale(loss).backward()
        if self.clip_grad_norm:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.clip_grad_norm
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.scheduler and self.scheduler_level == "batch":
            self.scheduler.step()
        return loss, outputs

    def _standard_step(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor, train: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass, with backward + optimizer step when training."""
        loss, outputs = self(x_batch=x_batch, y_batch=y_batch)
        if train:
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.clip_grad_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler and self.scheduler_level == "batch":
                self.scheduler.step()
        return loss, outputs

    def _process_batch(self, batch: list[torch.Tensor], split: str) -> None:
        """Dispatches a batch through the appropriate step and collects metrics."""
        x_batch, y_batch = batch
        x_batch = self.device_mgr.move_to_device(x_batch)
        y_batch = self.device_mgr.move_to_device(y_batch)

        if self.device_mgr.enable_amp and split == "train":
            loss, outputs = self._amp_step(x_batch, y_batch)
        else:
            loss, outputs = self._standard_step(
                x_batch, y_batch, train=split == "train"
            )

        self._collect_metrics(
            loss=loss, outputs=outputs, labels=y_batch, split=split
        )

    def _collect_metrics(
        self,
        loss: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        split: str,
    ) -> None:
        """Records loss and registered metrics."""
        # Updating loss
        loss = loss.item()
        self.metrics_tracker.update(f"loss/{split}", loss)

        # Updating info for computing extra metrics (if registered)
        if self.extra_metrics:
            if self.task == "classification":
                predictions = torch.argmax(outputs, dim=1)
            else:
                predictions = outputs.squeeze(-1).detach()
            self.metrics_tracker.update_preds(
                split=split,
                preds=predictions.cpu().numpy(),
                targets=labels.cpu().numpy(),
            )

    def _register_metrics(
        self, metrics: list[MetricConfig] | dict[str, Callable]
    ) -> None:
        """Registers custom metric functions with ``MetricsTracker``.

        A ``dict[str, Callable]`` is converted to ``list[MetricConfig]`` with
        ``plot=True`` for all entries. A ``list[MetricConfig]`` is used as-is,
        allowing per-metric plot control. Sets ``self.extra_metrics = True``.
        """
        # Checking if metrics are dict (setting plotting to True for all)
        if isinstance(metrics, dict):
            metrics = [
                MetricConfig(name=k, fn=v, plot=True)
                for k, v in metrics.items()
            ]

        # Sequentially registering passed metrics
        for metric_config in metrics:
            if not isinstance(metric_config.fn, Callable):
                raise TypeError(
                    f"Metric function specified for `{metric_config.name}` is not a callable object."
                )
            self.metrics_tracker.register_metric(
                name=metric_config.name, fn=metric_config.fn
            )
            # Collecting metrics which need to be plotted
            if metric_config.plot:
                self.metrics_to_plot.append(metric_config.name)

        # Setting a flag for extra metrics (needed for computing such metrics)
        self.extra_metrics = True

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None = None) -> dict[str, float]:
        """Evaluates the model on the test set and returns metric scores.

        Args:
            loader (DataLoader | None, optional): DataLoader to evaluate on.
                Falls back to ``self.test_loader`` if ``None``. Raises
                ``ValueError`` if neither is available. Defaults to ``None``.

        Returns:
            dict[str, float]: Metric scores keyed as ``'<name>/test'``,
                e.g. ``{'loss/test': 0.31, 'accuracy/test': 0.94}``.
        """
        loader = loader or getattr(self, "test_loader", None)
        if loader is None:
            raise ValueError("Cannot evaluate since test loader is not set")
        self._validate_arg(loader, DataLoader, "loader")

        for batch in loader:
            self._process_batch(batch, split="test")

        metrics = self.metrics_tracker.compute_metrics(test=True)
        self.metrics_tracker.reset()

        return metrics
