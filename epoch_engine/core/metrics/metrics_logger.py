"""Module for logging the computed metrics to a file."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import json


class MetricsLogger:
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def load_history(self, run_id: int) -> None:
        try:
            with open(self.filename) as h:
                training_process = json.load(h)
        except FileNotFoundError:
            training_process = {"runs": []}

        # Checking if run_id already exists
        run_key = f"run_id={run_id}"
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

    def log_metrics(
        self, metrics: dict[str, float | int], run_id: int
    ) -> None:
        run_key = f"run_id={run_id}"
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
        with open(self.filename, "w") as output:
            json.dump(self.training_process, output, indent=5)
