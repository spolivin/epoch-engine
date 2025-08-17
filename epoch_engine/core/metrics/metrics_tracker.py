"""Module for handling tracking and computation of metrics."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License


from typing import Any, Callable


class MetricsTracker:
    """Metrics tracker."""

    def __init__(self) -> None:
        """Instantiates an instance."""
        self.metrics: dict[str, list[float]] = {}
        self.metric_fns: dict[str, Callable] = {}
        self.preds: dict[str, list[Any]] = {}
        self.targets: dict[str, list[Any]] = {}

    def register_metric(self, name: str, fn: Callable) -> None:
        """Registers a metric function. fn(y_true, y_pred) -> float"""
        self.metric_fns[name] = fn

    def update(self, name: str, value: float) -> None:
        """Updates the data for loss computation.

        Args:
            name (str): Metric name.
            value (float): New value to be appended to losses.
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def update_preds(self, split: str, preds, targets) -> None:
        """Updates the predictions from the new batch for registered metrics computation.

        Args:
            split (str): Train or valid split of data.
            preds: Predictions to be added.
            targets: Targets to be added.
        """
        if split not in self.preds:
            self.preds[split] = []
            self.targets[split] = []
        self.preds[split].extend(preds)
        self.targets[split].extend(targets)

    def compute_metrics(self, test: bool = False) -> dict[str, float]:
        """Computes the metrics.

        Args:
            test (bool, optional): Flag for computing metrics on test set.

        Returns:
            dict[str, float]: A dictionary containing the computed metrics.
        """
        results = {}
        # Average batch metrics
        for name, values in self.metrics.items():
            results[name] = sum(values) / len(values)
        # Registered metrics
        if self.metric_fns is not {}:
            # Computing train/validation metrics
            if not test:
                for name, fn in self.metric_fns.items():
                    for split in ["train", "valid"]:
                        if split in self.preds:
                            results[name + f"/{split}"] = fn(
                                self.targets[split], self.preds[split]
                            )
            # Computing test metrics
            else:
                for name, fn in self.metric_fns.items():
                    results[name + "/test"] = fn(
                        self.targets["test"], self.preds["test"]
                    )
        return results

    def reset(self) -> None:
        """Resets the metrics tracker."""
        self.metrics = {}
        self.preds = {}
        self.targets = {}
