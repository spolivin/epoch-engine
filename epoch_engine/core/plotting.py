"Module for metrics plot generation logic."

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _get_metric_values_from_json(
    metric_name: str,
    run_id: str,
    json_file: str = "./runs/training_history.json",
) -> list[float]:
    """Retrieves the values of a metric from JSON-file for a specific run ID.

    Args:
        metric_name (str): Name of a metric.
        run_id (str): Trainer's run identifier.
        json_file (str): JSON filename containing the metrics results. Defaults to "./runs/training_history.json".

    Returns:
        list[float]: Collection of epoch-level metric values.
    """
    with open(json_file) as f:
        data = json.load(f)
    return [
        entry[metric_name]
        for run_dict in data["runs"]
        if f"run_id={run_id}" in run_dict.keys()
        for entry in run_dict[f"run_id={run_id}"]
    ]


def generate_plot_from_json(
    metric_name: str,
    run_id: str,
    json_file: str = "./runs/training_history.json",
) -> None:
    """Generates and saves a plot of a metric from JSON-file for a specific run ID.

    Args:
        metric_name (str): Name of a metric.
        run_id (str): Trainer's run identifier
        json_file (str, optional): JSON filename containing the metrics results. Defaults to "./runs/training_history.json".
    """
    # Using/creating save directory
    save_dir = f"./runs/run_id={run_id}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Retrieving the metrics values (train and valid) to be plotted from JSON file
    metric_values_train = _get_metric_values_from_json(
        metric_name=f"{metric_name}/train",
        run_id=run_id,
        json_file=json_file,
    )
    metric_values_valid = _get_metric_values_from_json(
        metric_name=f"{metric_name}/valid",
        run_id=run_id,
        json_file=json_file,
    )
    # Making and saving a plot
    epochs = np.arange(1, len(metric_values_train) + 1)
    plt.plot(epochs, metric_values_train, marker="o", lw=3)
    plt.plot(epochs, metric_values_valid, marker="o", lw=3)
    plt.xlabel("Epoch", size=15)
    plt.ylabel("Score", size=15)
    title_name = metric_name.capitalize() + f" (run_id={run_id})"
    plt.title(title_name)
    plt.legend(["Training", "Validation"], fontsize=10)
    plt.tight_layout()
    plt.savefig(save_dir + "/" + f"results_{metric_name}.png")
    plt.close()
