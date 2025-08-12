"""Module for Trainer functionality."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import os
import uuid
from typing import Any, Callable, TypeAlias

import torch
from tqdm import tqdm

from .checkpoint_handler import CheckpointHandler
from .configs import OptimizerConfig, SchedulerConfig
from .metrics.metrics_logger import MetricsLogger
from .metrics.metrics_tracker import MetricsTracker

TorchDataloader: TypeAlias = torch.utils.data.DataLoader
TorchModel: TypeAlias = torch.nn.Module
TorchTensor: TypeAlias = torch.Tensor


class Trainer:
    """Trainer of PyTorch models."""

    def __init__(
        self,
        model: TorchModel,
        criterion: Any,
        train_loader: TorchDataloader,
        valid_loader: TorchDataloader,
        train_on: str = "auto",
    ) -> None:
        """Initializes a class instance.

        Args:
            model (TorchModel): PyTorch model instance.
            criterion (Any): PyTorch loss function instance.
            train_loader (TorchDataloader): Torch Dataloader for training set.
            valid_loader (TorchDataloader): Torch Dataloader for validation set.
            train_on (str, optional): Device to be used to train the model. Defaults to "auto".

        Attributes:
            model (TorchModel): PyTorch model instance.
            criterion (Any): PyTorch loss function instance.
            optimizer (Any): PyTorch optimizer instance.
            train_loader (TorchDataloader): Torch Dataloader for training set.
            valid_loader (TorchDataloader): Torch Dataloader for validation set.
            device (torch.device): Device to be used to train the model.
            scheduler (Any): Instance of PyTorch scheduler for learning rate. Defaults to None.
            scheduler_level (str): Level on which to apply scheduler. Defaults to None.
            last_epoch (int): Last epoch when training has been successfully finished. Defaults to 0.
            metrics_tracker (MetricsTracker): Class for handling computing metrics.
            metrics_logger (MetricsLogger): Class for logging metrics.
            ckpt_handler (CheckpointHandler): Class for saving/loading checkpoints.
            extra_metrics (bool): Flag for the case when additional metrics are computed. Defaults to False.
            run_id (str): Trainer run identifier. Defaults to None.

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

        # Model
        if not isinstance(model, TorchModel):
            raise TypeError("'model' must be an instance of 'torch.nn.Module'")
        self.model = model.to(self.device)

        # Setting loss function
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
        self.metrics_logger = MetricsLogger(
            filename="./runs/training_history.json"
        )
        self.ckpt_handler = CheckpointHandler()

        self.extra_metrics = False  # Flag if extra metrics are registered
        self.run_id = None  # Trainer run identifier

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

    def configure_trainer(
        self,
        optimizer_class: Any | None = None,
        optimizer_params: dict[str, Any] | None = None,
        optimizer_config: OptimizerConfig | None = None,
        scheduler_class: Any = None,
        scheduler_params: dict[str, Any] = None,
        scheduler_level: str = "epoch",
        scheduler_config: SchedulerConfig | None = None,
    ) -> None:
        """Configures optimizer and (optionally) scheduler.

        Args:
            optimizer_class (Any, optional): Optimizer class from PyTorch. Defaults to None.
            optimizer_params (dict[str, Any], optional): Optimizer parameters. Defaults to None.
            optimizer_config (OptimizerConfig, optional): Instance of OptimizerConfig. Defaults to None.
            scheduler_class (Any, optional): Scheduler class from PyTorch. Defaults to None.
            scheduler_params (dict[str, Any], optional): Scheduler parameters. Defaults to None.
            scheduler_level (str, optional): Level for scheduler ("epoch" or "batch"). Defaults to "epoch".
            scheduler_config (SchedulerConfig, optional): Instance of SchedulerConfig. Defaults to None.

        Raises:
            TypeError: If `optimizer_params` or `scheduler_params` are not dictionaries.
            TypeError: If `optimizer_config` or `scheduler_config` are instances of `OptimizerConfig` or `SchedulerConfig` respectively.
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
            OSError: If when resuming the training from last checkpoint, checkpoint files are missing.
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
                # Error raised if resuming training is set by checkpoint files are missing
                if last_logged_epoch == 0:
                    raise OSError(
                        "Cannot resume training, since checkpoint directory for 'run_id={run_id}' appears to be empty."
                    )
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
        self.metrics_logger.load_history(run_id=self.run_id)

        # Setting seed
        if seed:
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            else:
                torch.manual_seed(seed)

        # Computing the number of total epochs to be trained (useful in case of additional trainings)
        total_epochs = epochs + self.last_epoch
        # Running training/validation loops
        for epoch in range(self.last_epoch, total_epochs):
            # Processing all training batches
            self._train_one_epoch(
                epoch=epoch, epochs=total_epochs, enable_tqdm=enable_tqdm
            )
            # Processing all validation batches
            self._validate_one_epoch(
                epoch=epoch, epochs=total_epochs, enable_tqdm=enable_tqdm
            )
            # Incrementing last epoch after successful training/validation
            self.last_epoch = epoch + 1

            # Saving the checkpoint
            self.save_checkpoint(path=ckpt_path.format(self.last_epoch))

            # Computing epoch-level metrics
            self.metrics = self.metrics_tracker.compute_metrics()
            self.metrics["epoch"] = epoch + 1

            # Logging computed metrics
            self.metrics_logger.log_metrics(
                metrics=self.metrics, run_id=self.run_id
            )
            # Resetting metrics counters after successful training/validation
            self.metrics_tracker.reset()

        # Saving the collected metrics in a file
        self.metrics_logger.save_history()

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

    def _process_batch(self, batch: list[torch.Tensor], split: str) -> None:
        """Backpropagates model during training and collects batch-level metrics.

        Args:
            batch (list[torch.Tensor]): Examples and labels.
            split (str): Indicator of training or validation set.
        """

        # Moving examples and labels batches to device chosen
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
        # Computing loss and output tensor for the batch
        loss, outputs = self(x_batch=x_batch, y_batch=y_batch)

        # If training set, backpropagate
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
        loss: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        split: str,
    ) -> None:
        """Records loss and accuracy.

        Args:
            loss (torch.Tensor): Value of loss function.
            outputs (torch.Tensor): Output tensor.
            labels (torch.Tensor): Batch of labels.
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

    def register_metrics(self, metrics_dict: dict[str, Callable]) -> None:
        """Registers a collection of metrics.

        Args:
            metrics_dict (dict[str, Callable]): Dictionary of name-callable metrics.
        """
        for metric_name, metric_fn in metrics_dict.items():
            self.register_metric(name=metric_name, metric_fn=metric_fn)

        # Setting a flag for extra metrics (needed for computing such metrics)
        self.extra_metrics = True

    def register_metric(self, name: str, metric_fn: Callable) -> None:
        """Registers a custom metric function.

        Args:
            name (str): Name of the metric.
            metric_fn (Callable): Function to compute the metric.
        """
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
