"""PNG plot generation from the JSON metrics history produced by MetricsLogger."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class MetricsPlotter:
    """Reads the JSON metrics history and saves per-metric PNG plots.

    Each plot shows training and validation curves over epochs and is saved
    to ``<base_dir>/run_id=<hex>/<plot_save_dir>/<metric_name>.png``.

    Attributes:
        base_dir (Path): Root directory containing the JSON log and run folders.
        metrics_path (Path): Full path to the JSON metrics history file.
        plot_save_dir (Path): Subdirectory name for plots within each run folder.
    """

    def __init__(
        self,
        base_dir: str = "runs",
        json_file: str = "metrics_history.json",
        plot_save_dir: str = "plots",
    ) -> None:
        """
        Args:
            base_dir (str, optional): Root directory for runs and the JSON log.
                Defaults to ``"runs"``.
            json_file (str, optional): Filename of the JSON metrics history
                produced by ``MetricsLogger``. Defaults to
                ``"metrics_history.json"``.
            plot_save_dir (str, optional): Subdirectory name created inside
                each run folder to store PNG files. Defaults to ``"plots"``.
        """
        self.base_dir = Path(base_dir)
        self.metrics_path = self.base_dir / json_file
        self.plot_save_dir = Path(plot_save_dir)

    def _get_metric_values(self, metric_name: str, run_id: str) -> list[float]:
        """Reads ``self.metrics_path`` and extracts per-epoch values for one metric.

        Args:
            metric_name (str): Full metric key including split suffix, e.g.
                ``'loss/train'`` or ``'accuracy/valid'``.
            run_id (str): Run ID whose history is searched.

        Returns:
            list[float]: Ordered list of metric values, one per logged epoch.
        """
        with open(self.metrics_path) as f:
            data = json.load(f)

        return [
            entry[metric_name]
            for run_dict in data["runs"]
            if f"run_id={run_id}" in run_dict.keys()
            for entry in run_dict[f"run_id={run_id}"]
        ]

    def create_plot(self, metric_name: str, run_id: str) -> None:
        """Generates and saves a train/validation curve PNG for one metric.

        Reads ``<metric_name>/train`` and ``<metric_name>/valid`` from the
        JSON history and saves the plot to
        ``<base_dir>/run_id=<hex>/<plot_save_dir>/<metric_name>.png``,
        creating the directory if needed.

        Args:
            metric_name (str): Base metric name without a split suffix, e.g.
                ``'loss'`` or ``'accuracy'``.
            run_id (str): Run ID whose history is plotted.
        """

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
