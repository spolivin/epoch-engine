"""Module for logging the computed metrics to a file."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import json
from pathlib import Path

from ..logger import TrainerLogger


class MetricsLogger:
    """Custom context manager for logging metrics to a file.

    Attributes:
        base_dir (str): Base directory where JSON file is to be stored.
        filename (str): Path to a JSON file where to log metrics.
        run_id (str): Trainer's run ID.
        logger (logging.Logger): Configured logger for logging Trainer events.
        current_run (dict[str, list[dict]]): Training logs belonging to current run ID.
        training_process (dict[str, list[dict]]): Full training logs.
    """

    def __init__(
        self,
        run_id: str,
        base_dir: str = "runs",
        logfile: str = "metrics_history.json",
    ) -> None:
        """Initializes a class instance.

        Args:
            run_id (str): Trainer's run ID.
            base_dir (str, optional): Base directory where JSON file is to be stored. Defaults to "runs".
            logfile (str, optional): Name of a logfile. Defaults to "metrics_history.json".
        """
        self.base_dir = Path(base_dir)
        self.filename = self.base_dir / logfile
        self.run_id = run_id
        self.logger = TrainerLogger().get_logger()

    def __enter__(self) -> "MetricsLogger":
        """Loads or creates a new history (JSON file) for the current run ID upon entry."""
        self.load_history()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Saves the recorded metrics history in JSON file upon error or exit."""
        if exc_type is not None:
            if exc_type == KeyboardInterrupt:
                self.logger.info(
                    f"Trainer run for 'run_id={self.run_id}' manually interrupted"
                )
            else:
                self.logger.info(
                    f"Trainer run for 'run_id={self.run_id}' interrupted due to unexpected error"
                )
        self.save_history()

    def load_history(self) -> None:
        """Loads or creates a new history (JSON file) for the current run ID."""
        try:
            with open(self.filename) as h:
                training_process = json.load(h)
        except FileNotFoundError:
            training_process = {"runs": []}

        # Checking if run_id already exists
        run_key = f"run_id={self.run_id}"
        for run in training_process["runs"]:
            if run_key in run:
                self.training_process = training_process
                # Keeping the reference for logging
                self.current_run = run
                break
        else:
            # Creating a new run if not found
            new_run = {run_key: []}
            training_process["runs"].append(new_run)
            self.training_process = training_process
            self.current_run = new_run

    def log_metrics(self, metrics: dict[str, float | int]) -> None:
        """Logs metrics.

        Args:
            metrics (dict[str, float  |  int]): Collection of metrics to be logged.
        """
        run_key = f"run_id={self.run_id}"
        # Finding the current run (should already be set by 'load_history')
        if hasattr(self, "current_run") and run_key in self.current_run:
            self.current_run[run_key].append(metrics)
        else:
            # Fallback: finding and appending
            for run in self.training_process["runs"]:
                if run_key in run:
                    run[run_key].append(metrics)
                    break

    def save_history(self) -> None:
        """Saves the recorded metrics history in JSON file."""
        with open(self.filename, "w") as output:
            json.dump(self.training_process, output, indent=5)
