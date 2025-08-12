"""Module for configurations."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

from typing import Any


class OptimizerConfig:
    """Optimizer configuration."""

    def __init__(self, optimizer_class: Any, optimizer_params: dict) -> None:
        """Instantiates an instance.

        Args:
            optimizer_class (Any): Torch optimizer class.
            optimizer_params (dict): Optimizer parameters for optimizer to be initialized with.

        Raises:
            TypeError: if `optimizer_params` is not a dictionary.
        """
        self.optimizer_class = optimizer_class
        if not isinstance(optimizer_params, dict):
            raise TypeError("'optimizer_params' must be a dictionary")
        self.optimizer_params = optimizer_params

    def create(self, model_params: Any) -> Any:
        """Creates an optimizer instance.

        Args:
            model_params (Any): Torch model parameters.

        Returns:
            Any: Optimizer class instance.
        """
        return self.optimizer_class(model_params, **self.optimizer_params)


class SchedulerConfig:
    def __init__(
        self,
        scheduler_class: Any,
        scheduler_params: dict,
        scheduler_level: str = "epoch",
    ):
        """Instantiates an instance.

        Args:
            scheduler_class (Any): Torch scheduler class.
            scheduler_params (dict): Scheduler parameters for scheduler to be initialized with.
            scheduler_level (str, optional): . Defaults to "epoch".

        Raises:
            TypeError: if `scheduler_params` is not a dictionary.
            ValueError: if invalid scheduler level is provided.
        """
        self.scheduler_class = scheduler_class
        if not isinstance(scheduler_params, dict):
            raise TypeError("'scheduler_params' must be a dictionary")
        self.scheduler_params = scheduler_params
        if scheduler_level not in ("epoch", "batch"):
            raise ValueError("Invalid scheduler_level")
        self.scheduler_level = scheduler_level

    def create(self, optimizer: Any) -> Any:
        """Creates a scheduler instance.

        Args:
            optimizer (Any): Optimizer instance.

        Returns:
            Any: Scheduler class instance.
        """
        return self.scheduler_class(optimizer, **self.scheduler_params)
