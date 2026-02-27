"""Module for displaying metrics in a clean format."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License


from tqdm import tqdm


def format_metrics(
    epoch: int,
    metrics: dict[str, float],
    use_tqdm: bool = True,
) -> None:
    """Display metrics in a clean table format.

    Args:
        epoch: Current epoch number
        metrics: Metrics dict (contains train and valid metrics)
        use_tqdm: Whether to use tqdm.write (for compatibility with progress bars)
    """

    def _write(msg: str):
        if use_tqdm:
            tqdm.write(msg)
        else:
            print(msg)

    # Prepare metrics for display
    all_metrics = {}

    # Add train metrics
    for key, value in metrics.items():
        if key.endswith("train"):
            metric_name = key.replace("/train", "")
            all_metrics[f"{metric_name} (train)"] = value
        elif key.endswith("valid"):
            metric_name = key.replace("/valid", "")
            all_metrics[f"{metric_name} (valid)"] = value
        else:
            all_metrics[key] = value

    # Build the display
    _write(f"\n{'='*70}")
    _write(f"Epoch {epoch} Results:")
    _write(f"{'-'*70}")

    # Display metrics in two columns for train/valid comparison
    train_keys = [k for k in all_metrics.keys() if "(train)" in k]
    valid_keys = [k for k in all_metrics.keys() if "(valid)" in k]

    # Get base metric names
    metric_names = set()
    for k in train_keys:
        metric_names.add(k.replace(" (train)", ""))
    for k in valid_keys:
        metric_names.add(k.replace(" (valid)", ""))

    # Display in organized format
    for metric in sorted(metric_names):
        train_key = f"{metric} (train)"
        valid_key = f"{metric} (valid)"

        train_val = all_metrics.get(train_key, None)
        valid_val = all_metrics.get(valid_key, None)

        _write(
            f"  {metric:20s} | Train: {train_val:8.4f} | Valid: {valid_val:8.4f}"
        )

    _write(f"{'='*70}\n")
