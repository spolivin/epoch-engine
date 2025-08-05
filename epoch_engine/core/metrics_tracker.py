"""Module for handling tracking and computation of metrics."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License


from typing import Any, Callable


class MetricsTracker:
    def __init__(self) -> None:
        self.metrics: dict[str, list[float]] = {}
        self.metric_fns: dict[str, Callable] = {}
        self.preds: dict[str, list[Any]] = {}
        self.targets: dict[str, list[Any]] = {}

    def register_metric(self, name: str, fn: Callable) -> None:
        """Register a metric function. fn(y_true, y_pred) -> float"""
        self.metric_fns[name] = fn

    def update(self, name: str, value: float) -> None:
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def update_preds(self, split: str, preds, targets):
        if split not in self.preds:
            self.preds[split] = []
            self.targets[split] = []
        self.preds[split].extend(preds)
        self.targets[split].extend(targets)

    def compute_metrics(self) -> dict[str, float]:
        """Compute the metrics.

        Returns:
            Dict[str, float]: A dictionary containing the computed metrics.
        """
        results = {}
        # Average batch metrics
        for name, values in self.metrics.items():
            results[name] = sum(values) / len(values)
        # Registered metrics
        if self.metric_fns is not {}:
            for name, fn in self.metric_fns.items():
                for split in ["train", "valid"]:
                    results[name + f"/{split}"] = fn(
                        self.targets[split], self.preds[split]
                    )
        return results

    def reset(self) -> None:
        """Reset the metrics tracker."""
        self.metrics = {}
        self.preds = {}
        self.targets = {}
