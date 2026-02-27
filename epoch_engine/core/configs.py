"""Module for configurations."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class MetricConfig:
    """Metric configuration.

    This is a dataclass for configuring the metric name which
    will be displayed in the training results, a way to compute it
    and an option to provide a plot of it at the end of training and
    validation.

    Attributes:
        name (str): Name of a metric.
        fn (Callable): Function to use for computing the metric values.
            Must follow the following signature: (y_true, y_pred) -> float.
        plot (bool): Option to provide a plot of this metric at the end of
            training and validation.
    """

    name: str
    fn: Callable
    plot: bool = False
