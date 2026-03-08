"""Batch-level accumulation and epoch-level computation of training metrics."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License


from collections import defaultdict
from collections.abc import Callable
from typing import Any


class MetricsTracker:
    """Accumulates per-batch scalars and predictions, then computes epoch stats.

    Typical flow per epoch:

    1. Call :meth:`register_metric` once per custom metric (or at init time).
    2. Call :meth:`update` each batch to record scalar values (e.g. loss).
    3. Call :meth:`update_preds` each batch to accumulate predictions and
       targets for custom metric functions.
    4. Call :meth:`compute_metrics` at epoch end to get averaged scalars and
       custom metric scores keyed as ``'<name>/<split>'``.
    5. Call :meth:`reset` to clear accumulators before the next epoch
       (``metric_fns`` are preserved).
    """

    def __init__(self) -> None:
        """Initialises empty accumulators.

        Attributes set here:
            metrics (defaultdict[str, list[float]]): Named scalar values
                accumulated across batches (e.g. per-batch losses).
            metric_fns (dict[str, Callable]): Registered metric functions
                keyed by metric name; persists across :meth:`reset` calls.
            preds (dict[str, list[Any]]): Per-split accumulated predictions.
            targets (dict[str, list[Any]]): Per-split accumulated targets.
        """
        self.metrics: defaultdict[str, list[float]] = defaultdict(list)
        self.metric_fns: dict[str, Callable] = {}
        self.preds: dict[str, list[Any]] = {}
        self.targets: dict[str, list[Any]] = {}

    def register_metric(self, name: str, fn: Callable) -> None:
        """Registers a custom metric function to be evaluated at epoch end.

        Args:
            name (str): Metric name used as the key prefix in
                :meth:`compute_metrics` output (e.g. ``'accuracy'``).
            fn (Callable): Metric function with signature
                ``(y_true, y_pred) -> float`` (sklearn-compatible).
        """
        self.metric_fns[name] = fn

    def update(self, name: str, value: float) -> None:
        """Appends a scalar value to the named accumulator.

        Args:
            name (str): Accumulator key (e.g. ``'loss/train'``).
            value (float): Scalar value from the current batch.
        """
        self.metrics[name].append(value)

    def update_preds(self, split: str, preds, targets) -> None:
        """Extends the accumulated predictions and targets for a given split.

        Args:
            split (str): Data split identifier — ``'train'``, ``'valid'``,
                or ``'test'``.
            preds: Batch predictions to extend the split's list with.
            targets: Batch targets to extend the split's list with.
        """
        if split not in self.preds:
            self.preds[split] = []
            self.targets[split] = []
        self.preds[split].extend(preds)
        self.targets[split].extend(targets)

    def compute_metrics(self, test: bool = False) -> dict[str, float]:
        """Averages accumulated scalars and evaluates registered metric functions.

        Scalar accumulators in ``self.metrics`` are averaged across batches.
        Registered ``metric_fns`` are applied to accumulated preds/targets for
        each split; results are keyed as ``'<name>/<split>'``. When
        ``test=True``, only the ``'test'`` split is evaluated; otherwise
        ``'train'`` and ``'valid'`` are used.

        Args:
            test (bool, optional): If ``True``, evaluate on the ``'test'``
                split instead of ``'train'``/``'valid'``. Defaults to
                ``False``.

        Returns:
            dict[str, float]: Epoch metrics, e.g. ``{'loss/train': 0.42,
                'loss/valid': 0.51, 'accuracy/train': 0.91, ...}``.
        """
        results = {}
        # Average batch metrics
        for name, values in self.metrics.items():
            results[name] = sum(values) / len(values)
        # Registered metrics
        if self.metric_fns:
            splits = ["test"] if test else ["train", "valid"]
            for split in splits:
                if split in self.preds:
                    for name, fn in self.metric_fns.items():
                        results[f"{name}/{split}"] = fn(
                            self.targets[split], self.preds[split]
                        )
        return results

    def reset(self) -> None:
        """Clears ``metrics``, ``preds``, and ``targets`` for the next epoch.

        ``metric_fns`` is intentionally preserved so registered functions
        do not need to be re-registered each epoch.
        """
        self.metrics = defaultdict(list)
        self.preds = {}
        self.targets = {}
