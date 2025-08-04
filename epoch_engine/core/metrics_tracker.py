"""Module for handling tracking and computation of metrics."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License


class MetricsTracker:
    """Class for keeping track of metrics."""

    def __init__(self) -> None:
        """Initializes a class instance.

        Attributes:
            metrics (dict[str, list[float]]): Batch-level metrics in a dict format.
            total_correct_train (int): Number of correct predictions in train set.
            total_samples_train (int): Total number of examples in train set.
            total_correct_valid (int): Number of correct predictions in valid set.
            total_samples_valid (int): Total number of examples in valid set.
        """
        self.metrics = {}
        self.total_correct_train = 0
        self.total_samples_train = 0
        self.total_correct_valid = 0
        self.total_samples_valid = 0

    def reset(self) -> None:
        """Resets all metrics and counters."""
        self.metrics = {}
        self.total_correct_train = 0
        self.total_samples_train = 0
        self.total_correct_valid = 0
        self.total_samples_valid = 0

    def update(self, name: str, value: float) -> None:
        """Updates a specific metric.

        Args:
            name (str): Metric name.
            value (float): Value of a metric.
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def update_accuracy(self, correct: int, total: int, split: str) -> None:
        """Updates accuracy counters.

        Args:
            correct (int): Number of correct predictions.
            total (int): Number of total examples.
            split (str): Train or valid split indicator.

        Raises:
            ValueError: Error in case some other split name is passed.
        """
        if split == "train":
            self.total_correct_train += correct
            self.total_samples_train += total
        elif split == "valid":
            self.total_correct_valid += correct
            self.total_samples_valid += total
        else:
            raise ValueError

    def compute_accuracy(self, split: str) -> float:
        """Computes accuracy across all batches.

        Args:
            split (str): Train or valid split indicator.

        Raises:
            ValueError: Error in case some other split name is passed.

        Returns:
            float: Accuracy score.
        """
        if self.total_samples_train == 0 or self.total_samples_valid == 0:
            return 0.0
        if split == "train":
            return self.total_correct_train / self.total_samples_train
        elif split == "valid":
            return self.total_correct_valid / self.total_samples_valid
        else:
            raise ValueError

    def get_all_metrics(self) -> dict[str, float]:
        """Computes all metrics as averages.

        Returns:
            dict[str, float]: Epoch-level metrics.
        """
        # Aggregating loss across batches
        metrics = {
            name: sum(values) / len(values) for name, values in self.metrics.items()
        }
        # Computing accuracy scores
        metrics["accuracy/train"] = self.compute_accuracy(split="train")
        metrics["accuracy/valid"] = self.compute_accuracy(split="valid")

        return metrics
