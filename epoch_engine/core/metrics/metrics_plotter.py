"""Module for plotting metrics."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class MetricsPlotter:
    """Class for plotting metrics.

    Attributes:
        base_dir (str): Base directory where plots are to be stored.
        metrics_path (str): Path where to look for metric values.
        plot_save_dir (str): Directory where to save plots.
    """

    def __init__(
        self,
        base_dir: str = "runs",
        json_file: str = "metrics_history.json",
        plot_save_dir: str = "plots",
    ) -> None:
        """Initializes a class instance.

        Args:
            base_dir (str, optional): Base directory where plots are to be stored. Defaults to "runs".
            json_file (str, optional): Name of a JSON file. Defaults to "metrics_history.json".
            plot_save_dir (str, optional): Directory where to save plots. Defaults to "plots".
        """
        self.base_dir = Path(base_dir)
        self.metrics_path = self.base_dir / json_file
        self.plot_save_dir = Path(plot_save_dir)

    def _get_metric_values(self, metric_name: str, run_id: str) -> list[float]:
        """Retrieves metric values for a specific metric name and run ID."""
        with open(self.metrics_path) as f:
            data = json.load(f)

        return [
            entry[metric_name]
            for run_dict in data["runs"]
            if f"run_id={run_id}" in run_dict.keys()
            for entry in run_dict[f"run_id={run_id}"]
        ]

    def create_plot(self, metric_name: str, run_id: str) -> None:
        """Creates a plot and saves it in set folder for a specific metric and run ID."""

        plot_save_dir = self.base_dir / f"run_id={run_id}" / self.plot_save_dir
        plot_save_filepath = plot_save_dir / f"{metric_name}.png"
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)

        metric_values_train = self._get_metric_values(
            metric_name=f"{metric_name}/train",
            run_id=run_id,
        )
        metric_values_valid = self._get_metric_values(
            metric_name=f"{metric_name}/valid",
            run_id=run_id,
        )

        epochs = np.arange(1, len(metric_values_train) + 1)
        plt.plot(epochs, metric_values_train, marker="o", lw=3)
        plt.plot(epochs, metric_values_valid, marker="o", lw=3)
        plt.xlabel("Epoch", size=15)
        plt.ylabel("Score", size=15)
        title_name = metric_name.capitalize() + f" (run_id={run_id})"
        plt.title(title_name)
        plt.legend(["Training", "Validation"], fontsize=10)
        plt.tight_layout()
        plt.savefig(plot_save_filepath)
        plt.close()
