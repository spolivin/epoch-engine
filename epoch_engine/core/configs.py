"""Module for configurations."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

from dataclasses import dataclass
from typing import Any, Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from .types import TorchDataloader, TorchModel


class OptimizerConfig:
    """Torch Optimizer configuration.

    Attributes:
        optimizer_class (Optimizer): Torch optimizer class.
        optimizer_params (dict): Optimizer parameters for optimizer to be initialized with.
    """

    def __init__(
        self, optimizer_class: Optimizer, optimizer_params: dict
    ) -> None:
        """Instantiates an instance of `OptimizerConfig`.

        Args:
            optimizer_class (Optimizer): Torch optimizer class. Must inherit from `torch.optim.Optimizer`.
            optimizer_params (dict): Optimizer parameters for optimizer to be initialized with.

        Raises:
            TypeError: if `optimizer_class` is not a subclass of `torch.optim.Optimizer`.
            TypeError: if `optimizer_params` is not a dictionary.
        """
        # Validating the optimizer class
        if not issubclass(optimizer_class, Optimizer):
            raise TypeError(
                "'optimizer_class' is not a subclass of 'torch.optim.Optimizer'"
            )
        self.optimizer_class = optimizer_class
        # Validating the optimizer parameters
        if not isinstance(optimizer_params, dict):
            raise TypeError("'optimizer_params' must be a dictionary")
        self.optimizer_params = optimizer_params

    def create(self, model_params: Any) -> Optimizer:
        """Creates an optimizer instance.

        Args:
            model_params (Any): Torch model parameters.

        Returns:
            Optimizer: Optimizer class instance.
        """
        return self.optimizer_class(model_params, **self.optimizer_params)


class SchedulerConfig:
    """Torch Scheduler configuration.

    Attributes:
        scheduler_class (LRScheduler | ReduceLROnPlateau): Torch scheduler class.
        scheduler_params (dict): Scheduler parameters for scheduler to be initialized with.
        scheduler_level (str, optional): Level on which to apply learning rate adjustment.
    """

    def __init__(
        self,
        scheduler_class: LRScheduler | ReduceLROnPlateau,
        scheduler_params: dict,
        scheduler_level: str = "epoch",
    ):
        """Instantiates an instance of `SchedulerConfig`.

        Args:
            scheduler_class (LRScheduler | ReduceLROnPlateau): Torch scheduler class. Must inherit from `torch.optim.lr_scheduler.LRScheduler` or `torch.optim.lr_scheduler.ReduceLROnPlateau`.
            scheduler_params (dict): Scheduler parameters for scheduler to be initialized with.
            scheduler_level (str, optional): Level on which to apply learning rate adjustment (batch of epoch levels are supported). Defaults to "epoch".

        Raises:
            TypeError: if `scheduler_class` is not a subclass of `torch.optim.lr_scheduler.LRScheduler` or `torch.optim.lr_scheduler.ReduceLROnPlateau`.
            TypeError: if `scheduler_params` is not a dictionary.
            ValueError: if invalid scheduler level is provided.
        """
        # Validating the scheduler class
        if not issubclass(scheduler_class, (LRScheduler, ReduceLROnPlateau)):
            raise TypeError(
                "'scheduler_class' must be a subclass of 'torch.optim.lr_scheduler.LRScheduler' "
                "or 'torch.optim.lr_scheduler.ReduceLROnPlateau'."
            )
        self.scheduler_class = scheduler_class
        # Validating the scheduler parameters
        if not isinstance(scheduler_params, dict):
            raise TypeError("'scheduler_params' must be a dictionary")
        self.scheduler_params = scheduler_params
        # Validating the scheduler level
        if scheduler_level not in ("epoch", "batch"):
            raise ValueError("Invalid scheduler_level")
        self.scheduler_level = scheduler_level

    def create(self, optimizer: Optimizer) -> LRScheduler | ReduceLROnPlateau:
        """Creates a scheduler instance.

        Args:
            optimizer (Optimizer): Optimizer instance.

        Returns:
            LRScheduler | ReduceLROnPlateau: Scheduler class instance.
        """
        return self.scheduler_class(optimizer, **self.scheduler_params)


@dataclass
class MetricConfig:
    """Metric configuration.

    This is a dataclass for configuring the metric name which
    will be displayed in the training results, a way to compute it
    and an option to provide a plot of it at the end of training and
    validation.

    Attributes:
        name (str): Name of a metric.
        fn (Callabel): Function to use for computing the metric values.
            Must follow the following signature: (y_true, y_pred) -> float.
        plot (bool): Option to provide a plot of this metric at the end of
            training and validation.
    """

    name: str
    fn: Callable
    plot: bool = False


@dataclass
class TrainerConfig:
    """Full configuration for Trainer.

    This dataclass serves as a simplification during
    Trainer's instantiation with providing all necessary
    parameters in this config.

    Attributes:
        model (TorchModel): PyTorch model instance.
        criterion (Callable): PyTorch loss function instance.
        train_loader (TorchDataloader): Torch Dataloader for training set.
        valid_loader (TorchDataloader): Torch Dataloader for validation set.
        optimizer_config (OptimizerConfig): Configuration for Torch's optimizer.
        train_on (str): Option to determine device automatically or by specifying its name.
        enable_amp (bool): Option to use Mixed Precision for computations.
        scheduler_config (SchedulerConfig): Configuration for Torch's scheduler.
        metrics (list[MetricConfig] | dict[str, Callable]): List of metric configurations.
    """

    model: TorchModel
    criterion: Callable
    train_loader: TorchDataloader
    valid_loader: TorchDataloader
    optimizer_config: OptimizerConfig
    train_on: str = "auto"
    enable_amp: bool = False
    scheduler_config: SchedulerConfig | None = None
    metrics: list[MetricConfig] | dict[str, Callable] | None = None
