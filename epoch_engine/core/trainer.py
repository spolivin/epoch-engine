"""Module for Trainer functionality."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

from typing import Any, Callable, TypeAlias

import torch
from tqdm import tqdm

from .checkpoint_handler import CheckpointHandler
from .metrics_tracker import MetricsTracker

TorchDataloader: TypeAlias = torch.utils.data.dataloader.DataLoader


class Trainer:
    """Trainer of PyTorch models."""

    def __init__(
        self,
        model: Any,
        criterion: Any,
        train_loader: TorchDataloader,
        valid_loader: TorchDataloader,
        train_on: str = "auto",
    ) -> None:
        """Initializes a class instance.

        Args:
            model (Any): PyTorch model instance.
            criterion (Any): PyTorch loss function instance.
            train_loader (TorchDataloader): Torch Dataloader for training set.
            valid_loader (TorchDataloader): Torch Dataloader for validation set.
            train_on (str, optional): Device to be used to train the model. Defaults to "auto".

        Attributes:
            model (Any): PyTorch model instance.
            criterion (Any): PyTorch loss function instance.
            optimizer (Any): PyTorch optimizer instance.
            train_loader (TorchDataloader): Torch Dataloader for training set.
            valid_loader (TorchDataloader): Torch Dataloader for validation set.
            device (torch.device): Device to be used to train the model.
            scheduler (Any): Instance of PyTorch scheduler for learning rate. Defaults to None.
            scheduler_level (bool): Level on which to apply scheduler. Defaults to None.
            last_epoch (int): Last epoch when training has been successfully finished. Defaults to 0.
            metrics_tracker (MetricsTracker): Class for handling computing metrics.
            ckpt_handler (CheckpointHandler): Class for saving/loading checkpoints.

        Raises:
            TypeError: If model is not an instance of torch.nn.Module.
            TypeError: If train_loader or valid_loader are not instances of torch.utils.data.DataLoader.
            ValueError: If train_on is not one of "auto", "cpu", "cuda", or "mps".
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
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")
        self.model = model.to(self.device)

        # Setting loss function
        self.criterion = criterion

        # Setting dataloaders for training/validation sets
        if not isinstance(
            train_loader, torch.utils.data.DataLoader
        ) or not isinstance(valid_loader, torch.utils.data.DataLoader):
            raise TypeError(
                "train_loader and valid_loader must be instances of torch.utils.data.DataLoader"
            )
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Setting scheduler-related attributes
        self.scheduler = None
        self.scheduler_level = None

        # Setting epoch indicator
        self.last_epoch = 0

        # Setting tracker for metrics and checkpoints handler
        self.metrics_tracker = MetricsTracker()
        self.ckpt_handler = CheckpointHandler()

        self.extra_metrics = False

    def _auto_detect_device_type(self) -> str:
        """Automatically detects the device type to use for training.

        Returns:
            str: The detected device (CPU, CUDA, or MPS).
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

    def configure_optimizers(
        self,
        optimizer_class: Any,
        optimizer_params: dict[str, Any],
        scheduler_class: Any = None,
        scheduler_params: dict[str, Any] = None,
        scheduler_level: str = "epoch",
    ) -> None:
        """Configures optimizer and (optionally) scheduler.

        Args:
            optimizer_class (Any): Optimizer class from PyTorch.
            optimizer_params (dict[str, Any]): Optimizer parameters.
            scheduler_class (Any, optional): Scheduler class from PyTorch. Defaults to None.
            scheduler_params (dict[str, Any], optional): Scheduler parameters. Defaults to None.
            scheduler_level (str, optional): Level for scheduler ("epoch" or "batch"). Defaults to "epoch".

        Raises:
            TypeError: If optimizer_params or scheduler_params are not dictionaries.
            ValueError: If scheduler_level is not "epoch" or "batch".
        """
        if not isinstance(optimizer_params, dict):
            raise TypeError("optimizer_params must be a dictionary")
        self.optimizer = optimizer_class(
            self.model.parameters(), **optimizer_params
        )
        if scheduler_class is not None and scheduler_params is not None:
            if not isinstance(scheduler_params, dict):
                raise TypeError("scheduler_params must be a dictionary")
            if scheduler_level not in ("epoch", "batch"):
                raise ValueError("Invalid scheduler_level")
            self.scheduler_level = scheduler_level
            self.scheduler = scheduler_class(
                self.optimizer, **scheduler_params
            )
            print(
                f"Scheduler has been configured with level: {scheduler_level}"
            )

    def reset_scheduler(self) -> None:
        """Resetting and removing scheduler."""
        self.scheduler = None
        self.scheduler_level = None

    def __call__(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes loss and model outputs.

        Args:
            x_batch (torch.Tensor): Batch of examples.
            y_batch (torch.Tensor): Batch of labels.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing value of loss function and output tensor.
        """
        outputs = self.model(x_batch)
        loss = self.criterion(outputs, y_batch.long())

        return loss, outputs

    def run(
        self,
        epochs: int,
        seed: int = 42,
        enable_tqdm: bool = True,
    ) -> None:
        """Launches the trainer.

        Args:
            epochs (int): Number of epochs to train.
            seed (int, optional): Random number generator seed. Defaults to 42.
            enable_tqdm (bool, optional): Indicator to turn on tqdm progress bar. Defaults to True.
        """
        if not hasattr(self, "optimizer"):
            raise ValueError(
                "Optimizer must be configured before running the trainer. Use `configure_optimizers` method."
            )
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
            # Resetting metrics counters after successful training/validation
            self.metrics_tracker.reset()

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
            with torch.no_grad():
                for batch in self.valid_loader:
                    self._process_batch(batch=batch, split="valid")
                    pbar.update(1)

            # Aggregating batch-level metrics (loss, accuracy) to epoch-level
            self.metrics = self.metrics_tracker.compute_metrics()
            # Setting the epoch-level stats to the progress bar
            epoch_stats = {
                "loss": (
                    round(self.metrics["loss/train"], 4),
                    round(self.metrics["loss/valid"], 4),
                )
            }
            if self.extra_metrics:
                for metric in self.metrics_tracker.metric_fns.keys():
                    epoch_stats[metric] = (
                        round(self.metrics[f"{metric}/train"], 4),
                        round(self.metrics[f"{metric}/valid"], 4),
                    )
            pbar.set_postfix(epoch_stats)

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
        # Adding batch-level loss
        loss = loss.item()
        self.metrics_tracker.update(f"loss/{split}", loss)

        # Adding batch-level accuracy
        if self.extra_metrics:
            predictions = torch.argmax(outputs, dim=1)
            self.metrics_tracker.update_preds(
                split=split,
                preds=predictions.cpu().numpy(),
                targets=labels.cpu().numpy(),
            )

    def register_metric(self, name: str, metric_fn: Callable) -> None:
        """Registers a custom metric function.

        Args:
            name (str): Name of the metric.
            metric_fn (Callable): Function to compute the metric.
        """
        self.metrics_tracker.register_metric(name=name, fn=metric_fn)
        self.extra_metrics = True

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
