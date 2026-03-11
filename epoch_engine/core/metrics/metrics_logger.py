"""Context-manager-based JSON logger for per-epoch metric history."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import json
from pathlib import Path
from types import TracebackType

from ..logger import TrainerLogger


class MetricsLogger:
    """Context manager that appends per-epoch metrics to a shared JSON log file.

    On entry, loads (or creates) ``metrics_history.json`` under ``base_dir``
    and locates the section for ``run_id``. On exit, always writes the
    updated history back to disk, logging an error message if training was
    interrupted.

    Attributes:
        base_dir (Path): Root directory containing the JSON log file.
        filename (Path): Full path to the JSON log file.
        run_id (str): Run ID whose entries are being recorded.
        logger (TrainerLogger): Logger used to report interruption events.
        current_run (dict[str, list[dict]]): Reference to the in-memory
            entry for the current run ID within ``training_process``.
        training_process (dict[str, list[dict]]): Full deserialized contents
            of the JSON log file (all runs).
    """

    def __init__(
        self,
        run_id: str,
        base_dir: str = "runs",
        logfile: str = "metrics_history.json",
        truncate_after_epoch: int | None = None,
    ) -> None:
        """
        Args:
            run_id (str): Run ID to record metrics under.
            base_dir (str, optional): Root directory for the log file.
                Defaults to ``"runs"``.
            logfile (str, optional): JSON log filename inside ``base_dir``.
                Defaults to ``"metrics_history.json"``.
            truncate_after_epoch (int | None, optional): If set, drops any
                previously logged entries with ``epoch > truncate_after_epoch``
                when loading history. Used when resuming from an earlier
                checkpoint (e.g. ``best.pt``). Defaults to ``None``.
        """
        self.base_dir = Path(base_dir)
        self.filename = self.base_dir / logfile
        self.run_id = run_id
        self.truncate_after_epoch = truncate_after_epoch
        self.logger = TrainerLogger()

    def __enter__(self) -> "MetricsLogger":
        """Calls ``load_history()`` and returns ``self``."""
        self.load_history()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Always calls ``save_history()``; logs an error message first if
        training was interrupted by an exception or ``KeyboardInterrupt``."""
        if exc_type is not None:
            if isinstance(exc_val, KeyboardInterrupt):
                self.logger.error(
                    f"Trainer run for 'run_id={self.run_id}' manually interrupted"
                )
            else:
                self.logger.error(
                    f"Trainer run for 'run_id={self.run_id}' interrupted due to unexpected error"
                )
        self.save_history()

    def load_history(self) -> None:
        """Loads ``metrics_history.json`` if it exists, or initialises an empty
        structure. Locates (or creates) the entry for ``run_id`` and applies
        ``truncate_after_epoch`` if set."""
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

        if self.truncate_after_epoch is not None:
            self.current_run[run_key] = [
                e
                for e in self.current_run[run_key]
                if e.get("epoch", 0) <= self.truncate_after_epoch
            ]

    def log_metrics(self, metrics: dict[str, float | int]) -> None:
        """Appends ``metrics`` to the current run's entry in ``training_process``.

        Args:
            metrics (dict[str, float | int]): Epoch metrics dict to record,
                e.g. ``{'epoch': 1, 'loss/train': 0.42, 'loss/valid': 0.51}``.
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
        """Serialises ``training_process`` and writes it to ``self.filename``."""
        with open(self.filename, "w") as output:
            json.dump(self.training_process, output, indent=5)
