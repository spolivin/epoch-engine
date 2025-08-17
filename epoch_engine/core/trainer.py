"""Module for Trainer functionality."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import os
import uuid
from typing import Any, Callable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from tqdm import tqdm

from .checkpoint_handler import CheckpointHandler
from .configs import (
    MetricConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
)
from .metrics.metrics_logger import MetricsLogger
from .metrics.metrics_tracker import MetricsTracker
from .plotting import generate_plot_from_json
from .types import TorchDataloader, TorchModel, TorchTensor


class Trainer:
    """Trainer of PyTorch models.

    Attributes:
        model (TorchModel): PyTorch model instance.
        criterion (Any): PyTorch loss function instance.
        optimizer (Any): PyTorch optimizer instance.
        train_loader (TorchDataloader): Torch Dataloader for training set.
        valid_loader (TorchDataloader): Torch Dataloader for validation set.
        device (torch.device): Device to be used to train the model.
        optimizer (Optimizer): Instance of PyTorch optimizer.
        scheduler (LRScheduler | ReduceLROnPlateau): Instance of PyTorch scheduler for learning rate.
        scheduler_level (str): Level on which to apply scheduler.
        last_epoch (int): Last epoch when training has been successfully finished.
        metrics_tracker (MetricsTracker): Instance of `MetricsTracker` for handling computing metrics.
        ckpt_handler (CheckpointHandler): Instance of `CheckpointHandler` for saving/loading checkpoints.
        extra_metrics (bool): Flag for the case when additional metrics are computed.
        run_id (str): Trainer run identifier.
        metrics_to_plot (list[str]): List of metrics for which resulting plots need to be created.
    """

    def __init__(
        self,
        model: TorchModel,
        criterion: Callable,
        train_loader: TorchDataloader,
        valid_loader: TorchDataloader,
        train_on: str = "auto",
        enable_amp: bool = False,
    ) -> None:
        """Initializes a class instance.

        Args:
            model (TorchModel): PyTorch model instance.
            criterion (Any): PyTorch loss function instance.
            train_loader (TorchDataloader): Torch Dataloader for training set.
            valid_loader (TorchDataloader): Torch Dataloader for validation set.
            train_on (str, optional): Device to be used to train the model. Defaults to "auto".

        Raises:
            TypeError: If `model` is not an instance of `torch.nn.Module`.
            TypeError: If `train_loader` or `valid_loader` are not instances of `torch.utils.data.DataLoader`.
            ValueError: If `train_on` is not one of `"auto"`, `"cpu"`, `"cuda"`, or `"mps"`.
        """
        # Using the CUDA, MPS or CPU device or inferring one
        if train_on == "auto":
            device = self._auto_detect_device_type()
        elif train_on in ("cpu", "cuda", "mps"):
            device = train_on
        else:
            raise ValueError(
                "Invalid device specified. Use 'auto', 'cpu', 'cuda', or 'mps'."
            )
        self.device = torch.device(device)

        self.enable_amp = enable_amp
        if self.enable_amp:
            if self.device.type == "cuda":
                self.scaler = torch.amp.GradScaler()
            else:
                raise RuntimeError(
                    "CUDA device is not available but 'enable_amp' is set to True. "
                    "Mixed precision can only be used on CUDA."
                )

        # Model
        if not isinstance(model, TorchModel):
            raise TypeError("'model' must be an instance of 'torch.nn.Module'")
        self.model = model.to(self.device)

        # Setting loss function
        if not isinstance(criterion, Callable):
            raise TypeError("'criterion' is not callable")
        self.criterion = criterion

        # Setting dataloaders for training/validation sets
        if not isinstance(train_loader, TorchDataloader) or not isinstance(
            valid_loader, TorchDataloader
        ):
            raise TypeError(
                "'train_loader' and 'valid_loader' must be instances of 'torch.utils.data.DataLoader'"
            )
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Setting scheduler-related attributes
        self.scheduler = None
        self.scheduler_level = None

        # Setting epoch indicator
        self.last_epoch = 0

        # Setting tracker/logger for metrics and checkpoints handler
        self.metrics_tracker = MetricsTracker()
        self.ckpt_handler = CheckpointHandler()

        self.extra_metrics = False  # Flag if extra metrics are registered
        self.run_id = None  # Trainer run identifier
        self.metrics_to_plot = [
            "loss"
        ]  # Initializing metrics to plot (plotting "loss" by default)

    def _auto_detect_device_type(self) -> str:
        """Automatically detects the device type to use for training.

        Returns:
            str: The detected device type (CPU, CUDA, or MPS).
        """
        if torch.cuda.is_available():
            return "cuda"
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"
        else:
            return "cpu"

    @classmethod
    def from_config(cls, config: TrainerConfig) -> "Trainer":
        """Configures Trainer, sets optimizer/scheduler and registers metrics.

        Args:
            config (TrainerConfig): Trainer configuration.

        Returns:
            Trainer: Initialized Trainer.
        """
        # Retrieving main arguments for the Trainer
        model = config.model
        criterion = config.criterion
        train_loader, valid_loader = config.train_loader, config.valid_loader
        train_on = config.train_on
        enable_amp = config.enable_amp

        # Initializing a Trainer instance
        trainer = cls(
            model, criterion, train_loader, valid_loader, train_on, enable_amp
        )

        # Configuring Trainer with optimizer and scheduler
        trainer.configure_trainer(
            optimizer_config=config.optimizer_config,
            scheduler_config=config.scheduler_config,
        )

        # Registering additional metrics if specified
        if config.metrics is not None:
            trainer.register_metrics(config.metrics)

        return trainer

    def configure_trainer(
        self,
        optimizer_class: Optimizer | None = None,
        optimizer_params: dict[str, Any] | None = None,
        optimizer_config: OptimizerConfig | None = None,
        scheduler_class: LRScheduler | ReduceLROnPlateau | None = None,
        scheduler_params: dict[str, Any] = None,
        scheduler_level: str = "epoch",
        scheduler_config: SchedulerConfig | None = None,
    ) -> None:
        """Configures optimizer and (optionally) scheduler.

        Args:
            optimizer_class (Optimizer, optional): Optimizer class from PyTorch. Defaults to None.
            optimizer_params (dict[str, Any], optional): Optimizer parameters. Defaults to None.
            optimizer_config (OptimizerConfig, optional): Instance of OptimizerConfig. Defaults to None.
            scheduler_class (LRScheduler | ReduceLROnPlateau, optional): Scheduler class from PyTorch. Defaults to None.
            scheduler_params (dict[str, Any], optional): Scheduler parameters. Defaults to None.
            scheduler_level (str, optional): Level for scheduler ("epoch" or "batch"). Defaults to "epoch".
            scheduler_config (SchedulerConfig, optional): Instance of SchedulerConfig. Defaults to None.

        Raises:
            TypeError: if `optimizer_class` is not a subclass of `torch.optim.Optimizer`.
            TypeError: If `optimizer_params` or `scheduler_params` are not dictionaries.
            TypeError: If `optimizer_config` or `scheduler_config` are not instances of `OptimizerConfig` or `SchedulerConfig` respectively.
            TypeError: if `scheduler_class` is not a subclass of `torch.optim.lr_scheduler.LRScheduler` or `torch.optim.lr_scheduler.ReduceLROnPlateau`.
            ValueError: if `optimizer_class` or `optimizer_config` are not provided as arguments.
            ValueError: if `optimizer_class` is provided but `optimizer_params` are not.
            ValueError: if `scheduler_class` is provided but `scheduler_params` are not.
            ValueError: If scheduler_level is not "epoch" or "batch".
        """
        # Verifying the at least one way of setting optimizer is provided
        if optimizer_class is None and optimizer_config is None:
            raise ValueError(
                "Either 'optimizer_class' or 'optimizer_config' should be provided. "
                "If using 'optimizer_class' specify params in 'optimizer_params'."
            )
        # Verifying that optimizer params are provided and are of correct type when using optimizer class
        if optimizer_class is not None:
            if not issubclass(optimizer_class, Optimizer):
                raise TypeError(
                    "'optimizer_class' is not a subclass of 'torch.optim.Optimizer'"
                )
            if optimizer_params is None:
                raise ValueError(
                    "Optimizer class is provided by 'optimizer_params' are not initialized."
                )
            elif not isinstance(optimizer_params, dict):
                raise TypeError("'optimizer_params' must be a dictionary")
        # Instantiating an optimizer based on the way of setting it
        if not optimizer_config:
            self.optimizer = optimizer_class(
                self.model.parameters(), **optimizer_params
            )
        else:
            if not isinstance(optimizer_config, OptimizerConfig):
                raise TypeError(
                    "'optimizer_config' must be of type OptimizerConfig"
                )
            # Creating an instance from config
            self.optimizer = optimizer_config.create(
                model_params=self.model.parameters()
            )

        # Using scheduler if provided
        if scheduler_class is not None or scheduler_config is not None:
            if scheduler_class is not None:
                if not issubclass(
                    scheduler_class, (LRScheduler, ReduceLROnPlateau)
                ):
                    raise TypeError(
                        "'scheduler_class' must be a subclass of 'torch.optim.lr_scheduler.LRScheduler' "
                        "or 'torch.optim.lr_scheduler.ReduceLROnPlateau'."
                    )
                if scheduler_params is None:
                    raise ValueError(
                        "Scheduler class is provided by 'scheduler_params' are not initialized."
                    )
                elif not isinstance(scheduler_params, dict):
                    raise TypeError("'scheduler_params' must be a dictionary")
            # Checking the correctness of setting scheduler level
            if scheduler_level not in ("epoch", "batch"):
                raise ValueError("Invalid scheduler_level")
            # Instantiating a scheduler based on the way of setting it
            if not scheduler_config:
                self.scheduler_level = scheduler_level
                self.scheduler = scheduler_class(
                    self.optimizer, **scheduler_params
                )
            else:
                if not isinstance(scheduler_config, SchedulerConfig):
                    raise TypeError(
                        "'scheduler_config' must be of type SchedulerConfig"
                    )
                self.scheduler_level = scheduler_config.scheduler_level
                # Creating an instance from config
                self.scheduler = scheduler_config.create(
                    optimizer=self.optimizer
                )

    def reset_scheduler(self) -> None:
        """Resetting and removing scheduler."""
        self.scheduler = None
        self.scheduler_level = None

    def __call__(
        self, x_batch: TorchTensor, y_batch: TorchTensor
    ) -> tuple[TorchTensor, TorchTensor]:
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
    ) -> None:
        """Launches a training/validation procedure.

        When `run_id` is not provided, initiates a new training run, where `runs`
        folder is created with subfolders for this run and `traning_process.json` file
        with training/metrics history. When providing `run_id`, checks if training with this
        ID exists in history and resumes training from the last checkpoint.

        Args:
            epochs (int): Number of epochs to train for.
            run_id (str | None, optional): Trainer run identifier. Defaults to None.
            seed (int, optional): Random number generator seed. Defaults to 42.
            enable_tqdm (bool, optional): Flag to enable progress bar. Defaults to True.

        Raises:
            AttributeError: If `optimizer` is not set as a class attribute.
            ValueError: If `run_id` cannot be found in the training history.
        """
        # Checking that optimizer has been configured
        if not hasattr(self, "optimizer"):
            raise AttributeError(
                "Optimizer must be configured before running the trainer. Use 'configure_trainer' method."
            )

        ckpts_path = (
            "./runs/run_id={0}/checkpoints"  # Checkpoints path to check
        )
        if run_id:
            ckpts_path = ckpts_path.format(run_id)
            # Checking if the target folder with checkpoints exists
            if os.path.exists(ckpts_path):
                last_logged_epoch = len(os.listdir(ckpts_path))
                # Starting fresh run if resuming training is set by checkpoint files are missing
                if last_logged_epoch == 0:
                    print(
                        f"Checkpoint directory for 'run_id={run_id}' appears to be empty. "
                        f"Resuming not possible, starting new run for 'run_id={run_id}'."
                    )
                    self.run_id = run_id
                else:
                    # If checkpoint files are present, loading the latest checkpoint
                    last_ckpt_path = (
                        ckpts_path + "/" + f"ckpt_epoch_{last_logged_epoch}.pt"
                    )
                    self.load_checkpoint(path=last_ckpt_path)
            # Error raised is resuming training is set but 'run_id' does exist in history
            else:
                raise ValueError(
                    f"Cannot resume training from 'run_id={run_id}', since it does not exist in history. "
                    "Check the correctness of 'run_id'."
                )
        # If 'run_id' is not provided (resuming training is turned off), creating 'runs' folder and generating 'run_id'
        else:
            self.run_id = uuid.uuid4().hex[:6]  # Generating a new 'run_id'
            # Creating a folder for a new run
            ckpts_path = ckpts_path.format(self.run_id)
            os.makedirs(ckpts_path)

        ckpt_path = (
            ckpts_path + "/" + "ckpt_epoch_{0}.pt"
        )  # Name of the checkpoint saved after each epoch
        # Loading training history from the file or creating a new history

        # Setting seed
        if seed:
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            else:
                torch.manual_seed(seed)

        # Computing the number of total epochs to be trained (useful in case of additional trainings)
        total_epochs = epochs + self.last_epoch
        # Running training/validation loops
        try:
            # Using custom context manager to make sure that history is not lost due to errors in loops below
            with MetricsLogger(
                filename="./runs/training_history.json", run_id=self.run_id
            ) as logger:
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

                    # Saving the checkpoint
                    self.save_checkpoint(
                        path=ckpt_path.format(self.last_epoch)
                    )

                    # Computing epoch-level metrics
                    self.metrics = self.metrics_tracker.compute_metrics()
                    self.metrics["epoch"] = epoch + 1

                    # Logging computed metrics
                    logger.log_metrics(metrics=self.metrics)
                    # Resetting metrics counters after successful training/validation
                    self.metrics_tracker.reset()
        # Making sure that in case of error metric plots are generated for completed epoch runs
        finally:
            for metric in self.metrics_to_plot:
                generate_plot_from_json(metric_name=metric, run_id=self.run_id)

    def _train_one_epoch(
        self, epoch: int, epochs: int, enable_tqdm: bool
    ) -> None:
        """Processes all training batches for one epoch.

        Args:
            epoch (int): Current epoch.
            epochs (int): Total number of epochs.
            enable_tqdm (bool): Indicator to turn on tqdm progress bar.
        """

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
        """Processes all validation batches for one epoch.

        Args:
            epoch (int): Current epoch.
            epochs (int): Total number of epochs.
            enable_tqdm (bool): Indicator to turn on tqdm progress bar.
        """
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

    def _process_batch(self, batch: list[TorchTensor], split: str) -> None:
        """Backpropagates model during training and collects batch-level metrics.

        Args:
            batch (list[TorchTensor]): Examples and labels.
            split (str): Indicator of training or validation set.
        """

        # Moving examples and labels batches to device chosen
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

        if self.enable_amp and split == "train":
            with torch.amp.autocast(device_type=self.device.type):
                loss, outputs = self(x_batch=x_batch, y_batch=y_batch)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.scheduler and self.scheduler_level == "batch":
                self.scheduler.step()
        else:
            loss, outputs = self(x_batch=x_batch, y_batch=y_batch)
            if split == "train":
                loss.backward()
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
        loss: TorchTensor,
        outputs: TorchTensor,
        labels: TorchTensor,
        split: str,
    ) -> None:
        """Records loss and accuracy.

        Args:
            loss (TorchTensor): Value of loss function.
            outputs (TorchTensor): Output tensor.
            labels (TorchTensor): Batch of labels.
            split (str): Indicator of training or validation set.
        """
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

    def register_metrics(
        self, metrics: list[MetricConfig] | dict[str, Callable]
    ) -> None:
        """Registers a collection of metrics.

        Args:
            metrics (list[MetricConfig] | dict[str, Callable]): Dictionary of name-callable metrics.
        """
        if isinstance(metrics, dict):
            metrics = [
                MetricConfig(name=k, fn=v, plot=True)
                for k, v in metrics.items()
            ]

        for metric_config in metrics:
            self.register_metric(
                name=metric_config.name, metric_fn=metric_config.fn
            )
            if metric_config.plot:
                self.metrics_to_plot.append(metric_config.name)

        # Setting a flag for extra metrics (needed for computing such metrics)
        self.extra_metrics = True

    def register_metric(self, name: str, metric_fn: Callable) -> None:
        """Registers a custom metric function.

        Args:
            name (str): Name of the metric.
            metric_fn (Callable): Function to compute the metric.

        Raises:
            TypeError: if the provided function is not a callable object.
        """
        if not isinstance(metric_fn, Callable):
            raise TypeError(
                f"Metric function specified for '{name}' is not a callable object."
            )
        self.metrics_tracker.register_metric(name=name, fn=metric_fn)

    def save_checkpoint(self, path: str) -> None:
        """Saves trainer checkpoint.

        Args:
            path (str): Path to the checkpoint to be saved.
        """
        self.ckpt_handler.save_checkpoint(path=path, trainer=self)

    def load_checkpoint(self, path: str) -> None:
        """Loads trainer checkpoint.

        Args:
            path (str): Path to the checkpoint to be loaded.
        """
        self.ckpt_handler.load_checkpoint(path=path, trainer=self)

    @torch.no_grad()
    def evaluate(self, loader: TorchDataloader) -> dict[str, float]:
        """Evaluates the model on some test set.

        Args:
            loader (TorchDataloader): Torch Dataloader.

        Raises:
            TypeError: Exception raised in case input loader is not a DataLoader.

        Returns:
            dict[str, float]: Dictionary with computed metrics.
        """
        if not isinstance(loader, TorchDataloader):
            raise TypeError(
                "'loader' must be an instance of 'torch.utils.data.DataLoader'"
            )
        for batch in loader:
            self._process_batch(batch, split="test")

        metrics = self.metrics_tracker.compute_metrics(test=True)

        return metrics
