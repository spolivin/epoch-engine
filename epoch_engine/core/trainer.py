"""Module for Trainer functionality."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .configs import (
    MetricConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
)
from .logger import TrainerLogger
from .managers import DeviceManager, RunManager
from .metrics import MetricsLogger, MetricsPlotter, MetricsTracker


class Trainer:
    """Trainer of PyTorch models.

    Attributes:
        model (nn.Module): PyTorch model instance.
        criterion (Callable): PyTorch loss function instance.
        train_loader (Dataloader): Torch Dataloader for training set.
        valid_loader (Dataloader): Torch Dataloader for validation set.
        test_loader (Dataloader | None): Torch Dataloader for test set.
        optimizer (Optimizer): Instance of PyTorch optimizer.
        scheduler (LRScheduler | ReduceLROnPlateau): Instance of PyTorch scheduler for learning rate.
        scheduler_level (str): Level on which to apply scheduler.
        last_epoch (int): Last epoch when training has been successfully finished.
        metrics_tracker (MetricsTracker): Instance of `MetricsTracker` for handling computing metrics.
        metrics_plotter (MetricsPlotter): Instance of `MetricsPlotter` for plotting metrics.
        logger (logging.Logger): Configured logger for logging Trainer events.
        device_mgr (DeviceManager): Instance of `DeviceManager` for handling device placement/AMP.
        run_mgr (RunManager): Instance of `RunManager` for handing run-specific operations.
        extra_metrics (bool): Flag for the case when additional metrics are computed.
        run_id (str): Trainer run identifier.
        metrics_to_plot (list[str]): List of metrics for which resulting plots need to be created.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader | None = None,
        amp: bool = False,
    ) -> None:
        """Initializes a class instance.

        Args:
            model (nn.Module): PyTorch model instance.
            criterion (Callable): PyTorch loss function instance.
            train_loader (Dataloader): Torch Dataloader for training set.
            valid_loader (Dataloader): Torch Dataloader for validation set.
            test_loader (Dataloader, optional): Torch Dataloader for test set. Defaults to None.
            amp (bool, optional): Flag to apply mixed precision training. Defaults to False.

        Raises:
            TypeError: If `model` is not an instance of `torch.nn.Module`.
            TypeError: If `criterion` is not a callable object.
            TypeError: If `train_loader` or `valid_loader` or `test_loader` are not instances of `torch.utils.data.DataLoader`.
        """
        # Determining device automatically and handling AMP (if enabled)
        self.device_mgr = DeviceManager(enable_amp=amp)
        if self.device_mgr.enable_amp:
            self.scaler = torch.amp.GradScaler()

        # Validating model and moving it to device
        if not isinstance(model, nn.Module):
            raise TypeError("`model` must be an instance of `torch.nn.Module`")
        self.model = self.device_mgr.move_to_device(model)

        # Validating and setting loss function
        if not isinstance(criterion, Callable):
            raise TypeError("'criterion' is not callable")
        self.criterion = criterion

        # Setting dataloaders for training/validation sets
        if not isinstance(train_loader, DataLoader) or not isinstance(
            valid_loader, DataLoader
        ):
            raise TypeError(
                "`train_loader` and `valid_loader` must be instances of `torch.utils.data.DataLoader`"
            )
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        # Setting dataloader for test set (if set)
        if test_loader is not None:
            if not isinstance(test_loader, DataLoader):
                raise TypeError(
                    "`test_loader` must be instances of `torch.utils.data.DataLoader`"
                )
            self.test_loader = test_loader

        # Setting scheduler-specific attributes
        self.scheduler = None
        self.scheduler_level = None

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

    @classmethod
    def from_config(
        cls,
        config: TrainerConfig,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig | None = None,
    ) -> "Trainer":
        """Configures Trainer, sets optimizer/scheduler and registers metrics.

        Args:
            config (TrainerConfig): Trainer configuration.
            optimizer_config (OptimizerConfig): Optimizer configuration.
            scheduler_config (SchedulerConfig, optional): Scheduler configuration. Defaults to None.

        Returns:
            Trainer: Initialized and fully configured Trainer.
        """
        # Retrieving main arguments for the Trainer
        model = config.model
        criterion = config.criterion
        train_loader, valid_loader, test_loader = (
            config.train_loader,
            config.valid_loader,
            config.test_loader,
        )
        amp = config.enable_amp

        # Initializing a Trainer instance
        trainer = cls(
            model, criterion, train_loader, valid_loader, test_loader, amp
        )

        # Configuring Trainer with optimizer and scheduler
        trainer._configure_trainer(
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
        )

        # Registering additional metrics if specified
        if config.metrics is not None:
            trainer._register_metrics(metrics=config.metrics)

        return trainer

    def _configure_trainer(
        self,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig | None = None,
    ) -> None:
        """Configures optimizer and (optionally) scheduler."""
        # Validating the optimizer config
        if not isinstance(optimizer_config, OptimizerConfig):
            raise TypeError(
                "`optimizer_config` must be of type `OptimizerConfig`"
            )
        # Creating an instance from config
        self.optimizer = optimizer_config.create(
            model_params=self.model.parameters()
        )

        # Using scheduler if provided
        if scheduler_config is not None:
            # Checking the correctness of setting scheduler level
            if scheduler_config.scheduler_level not in ("epoch", "batch"):
                raise ValueError("Invalid scheduler_level")
            # Instantiating a scheduler based on the way of setting it
            if not isinstance(scheduler_config, SchedulerConfig):
                raise TypeError(
                    "`scheduler_config` must be of type `SchedulerConfig`"
                )
            self.scheduler_level = scheduler_config.scheduler_level
            # Creating an instance from config
            self.scheduler = scheduler_config.create(optimizer=self.optimizer)

    def reset_scheduler(self) -> None:
        """Resetting and removing scheduler."""
        self.scheduler = None
        self.scheduler_level = None

    def __call__(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes loss and model outputs.

        Args:
            x_batch (TorchTensor): Batch of examples.
            y_batch (TorchTensor): Batch of labels.

        Returns:
            tuple[TorchTensor, TorchTensor]: Tuple containing value of loss function and output tensor.
        """
        outputs = self.model(x_batch)
        loss = self.criterion(outputs, y_batch.long())

        return loss, outputs

    def run(
        self,
        epochs: int,
        run_id: str | None = None,
        seed: int = 42,
        enable_tqdm: bool = True,
        clip_grad_norm: float | None = None,
    ) -> None:
        """Launches a training/validation procedure.

        When `run_id` is not provided, either initiates a new training run, where `runs`
        folder is created with subfolders/auxiliary files for this run, or resumes the run
        specified in `run_id` attribute (if already set up).

        When providing `run_id`, checks if training with this ID exists in history and resumes
        training from the last checkpoint.

        Args:
            epochs (int): Number of epochs to train for.
            run_id (str | None, optional): Trainer run identifier. Defaults to None.
            seed (int, optional): Random number generator seed. Defaults to 42.
            enable_tqdm (bool, optional): Flag to enable progress bar. Defaults to True.
            clip_grad_norm (float, optional): Max norm for gradient clipping. Defaults to None.
        """
        # Checking if `run_id` attribute is set or `run_id` argument is passed (resumed runs)
        try:
            run_info = self.run_mgr.init_run(run_id=run_id or self.run_id)
        # Fallback to case when `run_id=None` and `run_id` attribute is not set yet (first run)
        except AttributeError:
            run_info = self.run_mgr.init_run(run_id=run_id)
        self.run_id = run_info.run_id
        ckpts_path = run_info.ckpts_path
        # Instantiating a logger for logging events related to Trainer
        self.logger = TrainerLogger().get_logger()

        # Checking if there have been training runs before
        if run_info.last_logged_epoch > 0:
            # Loading the last checkpoint
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
            with MetricsLogger(run_id=self.run_id) as metrics_logger:
                for epoch in range(self.last_epoch, total_epochs):
                    # Processing all training batches
                    self._train_one_epoch(
                        epoch=epoch,
                        epochs=total_epochs,
                        enable_tqdm=enable_tqdm,
                    )
                    # Processing all validation batches
                    self._validate_one_epoch(
                        epoch=epoch,
                        epochs=total_epochs,
                        enable_tqdm=enable_tqdm,
                    )
                    # Incrementing last epoch after successful training/validation
                    self.last_epoch = epoch + 1
                    self.logger.info(f"Finished epoch {self.last_epoch}")

                    # Saving the checkpoint
                    ckpt_path = ckpts_path / f"ckpt_epoch_{self.last_epoch}.pt"
                    self.run_mgr.save_checkpoint(
                        path=ckpt_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=self.last_epoch,
                    )

                    # Computing epoch-level metrics
                    self.metrics = self.metrics_tracker.compute_metrics()
                    self.metrics["epoch"] = epoch + 1

                    # Logging computed metrics
                    metrics_logger.log_metrics(metrics=self.metrics)
                    # Resetting metrics counters after successful training/validation
                    self.metrics_tracker.reset()
        # Making sure that in case of error metric plots are generated for completed epoch runs
        finally:
            for metric in self.metrics_to_plot:
                self.metrics_plotter.create_plot(
                    metric_name=metric, run_id=self.run_id
                )
        self.logger.info(
            f"Finished run 'run_id={run_info.run_id}' successfully at epoch {self.last_epoch}"
        )

    def _train_one_epoch(
        self, epoch: int, epochs: int, enable_tqdm: bool
    ) -> None:
        """Processes all training batches for one epoch."""

        with tqdm(
            total=len(self.train_loader),
            desc=f"Epoch {epoch + 1}/{epochs} [Training]",
            position=0,
            leave=True,
            unit="batches",
            disable=not enable_tqdm,
        ) as pbar:
            # Setting the learning rate in progress bar
            pbar.set_postfix({"lr": self.optimizer.param_groups[0]["lr"]})
            # Setting model in training mode
            self.model.train()
            # Processing all training batches
            for batch in self.train_loader:
                self._process_batch(batch=batch, split="train")
                pbar.update(1)
        # Adjusting learning rate if set
        if self.scheduler and self.scheduler_level == "epoch":
            self.scheduler.step()

    @torch.no_grad()
    def _validate_one_epoch(
        self, epoch: int, epochs: int, enable_tqdm: bool
    ) -> None:
        """Processes all validation batches for one epoch."""

        with tqdm(
            total=len(self.valid_loader),
            desc=f"Epoch {epoch + 1}/{epochs} [Validation]",
            position=0,
            leave=True,
            unit="batches",
            disable=not enable_tqdm,
        ) as pbar:
            # Setting model to evaluation model
            self.model.eval()
            # Processing all validation batches (grad compute turned off)
            for batch in self.valid_loader:
                self._process_batch(batch=batch, split="valid")
                pbar.update(1)

    def _process_batch(self, batch: list[torch.Tensor], split: str) -> None:
        """Backpropagates model during training and collects batch-level metrics."""

        # Moving examples and labels batches to device chosen
        x_batch, y_batch = batch
        x_batch, y_batch = self.device_mgr.move_to_device(
            x_batch
        ), self.device_mgr.move_to_device(y_batch)

        if self.device_mgr.enable_amp and split == "train":
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
        else:
            loss, outputs = self(x_batch=x_batch, y_batch=y_batch)
            if split == "train":
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.clip_grad_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()
                # Adjusting learning rate at batch-level
                if self.scheduler and self.scheduler_level == "batch":
                    self.scheduler.step()

        # Collecting batch-level metrics
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
            predictions = torch.argmax(outputs, dim=1)
            self.metrics_tracker.update_preds(
                split=split,
                preds=predictions.cpu().numpy(),
                targets=labels.cpu().numpy(),
            )

    def _register_metrics(
        self, metrics: list[MetricConfig] | dict[str, Callable]
    ) -> None:
        """Registers a collection of metrics.

        Metrics can be passed as either in the form of "str-Callable" dictionary,
        in which case all such metrics will be set to be plotted at the end of trainer
        run, or as list[MetricConfig] in case of needing to "turn off" plotting some metrics.
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
        """Evaluates the model on test set."""
        if loader is not None:
            if not isinstance(loader, DataLoader):
                raise TypeError(
                    "`loader` must be an instance of `torch.utils.data.DataLoader`"
                )
        else:
            if not hasattr(self, "test_loader"):
                raise ValueError(
                    "Cannot evaluate since test loader is not set"
                )
            else:
                loader = self.test_loader

        for batch in loader:
            self._process_batch(batch, split="test")

        metrics = self.metrics_tracker.compute_metrics(test=True)
        self.metrics_tracker.reset()

        return metrics
