"""Configuration dataclasses for the training framework."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class MetricConfig:
    """Per-metric display and plot configuration.

    Use this instead of a plain ``dict[str, Callable]`` when you need
    per-metric control over the display name or PNG plot generation.

    Attributes:
        name (str): Metric name shown in logs and plot titles.
        fn (Callable): Metric function with signature
            ``(y_true, y_pred) -> float`` (sklearn-compatible).
        plot (bool): If ``True``, a PNG plot is generated for this metric
            at the end of training. Defaults to ``False``.
    """

    name: str
    fn: Callable[..., float]
    plot: bool = False
