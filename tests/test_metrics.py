"""Tests for epoch_engine.core.metrics.metrics_tracker and display modules."""

from unittest.mock import patch

import pytest

from epoch_engine.core.metrics.display import format_metrics
from epoch_engine.core.metrics.metrics_tracker import MetricsTracker

# ---------------------------------------------------------------------------
# MetricsTracker — init
# ---------------------------------------------------------------------------


class TestMetricsTrackerInit:
    def test_initial_state_empty(self):
        """All internal collections start empty on construction."""
        tracker = MetricsTracker()
        assert tracker.metrics == {}
        assert tracker.metric_fns == {}
        assert tracker.preds == {}
        assert tracker.targets == {}


# ---------------------------------------------------------------------------
# MetricsTracker — register_metric
# ---------------------------------------------------------------------------


class TestMetricsTrackerRegisterMetric:
    def test_registers_fn_under_name(self):
        """Registered function is stored under its name."""
        tracker = MetricsTracker()

        def fn(y_true, y_pred):
            return 1.0

        tracker.register_metric("accuracy", fn)
        assert "accuracy" in tracker.metric_fns
        assert tracker.metric_fns["accuracy"] is fn

    def test_registers_multiple_metrics(self):
        """Multiple metrics can be registered independently."""
        tracker = MetricsTracker()
        tracker.register_metric("acc", lambda a, b: 0.0)
        tracker.register_metric("f1", lambda a, b: 0.0)
        assert "acc" in tracker.metric_fns
        assert "f1" in tracker.metric_fns

    def test_overwrite_existing_metric(self):
        """Re-registering a name replaces the previous function."""
        tracker = MetricsTracker()

        def fn1(a, b):
            return 0.0

        def fn2(a, b):
            return 1.0

        tracker.register_metric("acc", fn1)
        tracker.register_metric("acc", fn2)
        assert tracker.metric_fns["acc"] is fn2


# ---------------------------------------------------------------------------
# MetricsTracker — update
# ---------------------------------------------------------------------------


class TestMetricsTrackerUpdate:
    def test_creates_list_on_first_update(self):
        """First update creates a list with the value."""
        tracker = MetricsTracker()
        tracker.update("loss/train", 0.5)
        assert tracker.metrics["loss/train"] == [0.5]

    def test_appends_on_subsequent_updates(self):
        """Subsequent updates append to the existing list."""
        tracker = MetricsTracker()
        tracker.update("loss/train", 0.5)
        tracker.update("loss/train", 0.3)
        assert tracker.metrics["loss/train"] == [0.5, 0.3]

    def test_multiple_metric_names(self):
        """Different metric keys are tracked separately."""
        tracker = MetricsTracker()
        tracker.update("loss/train", 0.5)
        tracker.update("loss/valid", 0.4)
        assert "loss/train" in tracker.metrics
        assert "loss/valid" in tracker.metrics


# ---------------------------------------------------------------------------
# MetricsTracker — update_preds
# ---------------------------------------------------------------------------


class TestMetricsTrackerUpdatePreds:
    def test_creates_split_on_first_call(self):
        """First call creates preds/targets lists for the split."""
        tracker = MetricsTracker()
        tracker.update_preds("train", [1, 0], [1, 1])
        assert tracker.preds["train"] == [1, 0]
        assert tracker.targets["train"] == [1, 1]

    def test_extends_existing_split(self):
        """Subsequent calls extend the existing lists."""
        tracker = MetricsTracker()
        tracker.update_preds("train", [1, 0], [1, 1])
        tracker.update_preds("train", [0], [0])
        assert tracker.preds["train"] == [1, 0, 0]
        assert tracker.targets["train"] == [1, 1, 0]

    def test_multiple_splits_are_independent(self):
        """Train and valid splits are stored separately."""
        tracker = MetricsTracker()
        tracker.update_preds("train", [1], [1])
        tracker.update_preds("valid", [0], [1])
        assert tracker.preds["train"] == [1]
        assert tracker.preds["valid"] == [0]


# ---------------------------------------------------------------------------
# MetricsTracker — compute_metrics
# ---------------------------------------------------------------------------


class TestMetricsTrackerComputeMetrics:
    def test_averages_batch_metrics(self):
        """Batch losses are averaged across updates."""
        tracker = MetricsTracker()
        tracker.update("loss/train", 0.4)
        tracker.update("loss/train", 0.6)
        results = tracker.compute_metrics()
        assert results["loss/train"] == pytest.approx(0.5)

    def test_single_value_metric(self):
        """A single-value metric is returned as-is."""
        tracker = MetricsTracker()
        tracker.update("loss/valid", 0.3)
        results = tracker.compute_metrics()
        assert results["loss/valid"] == pytest.approx(0.3)

    def test_registered_metric_computed_for_train_split(self):
        """Registered metric is computed for the train split."""
        tracker = MetricsTracker()
        tracker.register_metric("accuracy", lambda y_true, y_pred: 0.9)
        tracker.update_preds("train", [1, 0], [1, 0])
        results = tracker.compute_metrics()
        assert "accuracy/train" in results
        assert results["accuracy/train"] == pytest.approx(0.9)

    def test_registered_metric_computed_for_valid_split(self):
        """Registered metric is computed for the valid split."""
        tracker = MetricsTracker()
        tracker.register_metric("accuracy", lambda y_true, y_pred: 0.8)
        tracker.update_preds("valid", [1, 0], [1, 0])
        results = tracker.compute_metrics()
        assert "accuracy/valid" in results
        assert results["accuracy/valid"] == pytest.approx(0.8)

    def test_registered_metric_receives_correct_arguments(self):
        """Metric function receives (y_true, y_pred) from stored targets/preds."""
        tracker = MetricsTracker()
        received = {}

        def capture_fn(y_true, y_pred):
            received["y_true"] = y_true
            received["y_pred"] = y_pred
            return 1.0

        tracker.register_metric("acc", capture_fn)
        tracker.update_preds("train", [0, 1], [1, 1])
        tracker.compute_metrics()
        assert received["y_true"] == [1, 1]
        assert received["y_pred"] == [0, 1]

    def test_test_mode_computes_test_split(self):
        """test=True computes metrics for the test split."""
        tracker = MetricsTracker()
        tracker.register_metric("accuracy", lambda y_true, y_pred: 0.75)
        tracker.update_preds("test", [1, 0, 1], [1, 0, 1])
        results = tracker.compute_metrics(test=True)
        assert "accuracy/test" in results
        assert results["accuracy/test"] == pytest.approx(0.75)

    def test_test_mode_does_not_include_train_valid_keys(self):
        """test=True excludes train/valid metric keys from the result."""
        tracker = MetricsTracker()
        tracker.register_metric("accuracy", lambda y_true, y_pred: 1.0)
        tracker.update_preds("test", [1], [1])
        results = tracker.compute_metrics(test=True)
        assert "accuracy/train" not in results
        assert "accuracy/valid" not in results

    def test_no_metric_fns_returns_only_batch_metrics(self):
        """Without registered functions only batch averages are returned."""
        tracker = MetricsTracker()
        tracker.update("loss/train", 1.0)
        results = tracker.compute_metrics()
        assert list(results.keys()) == ["loss/train"]

    def test_empty_tracker_returns_empty_dict(self):
        """Empty tracker returns an empty dict."""
        tracker = MetricsTracker()
        assert tracker.compute_metrics() == {}


# ---------------------------------------------------------------------------
# MetricsTracker — reset
# ---------------------------------------------------------------------------


class TestMetricsTrackerReset:
    def test_reset_clears_metrics(self):
        """reset() empties the metrics dict."""
        tracker = MetricsTracker()
        tracker.update("loss/train", 0.5)
        tracker.reset()
        assert tracker.metrics == {}

    def test_reset_clears_preds_and_targets(self):
        """reset() empties preds and targets."""
        tracker = MetricsTracker()
        tracker.update_preds("train", [1], [1])
        tracker.reset()
        assert tracker.preds == {}
        assert tracker.targets == {}

    def test_reset_preserves_metric_fns(self):
        """reset() keeps registered metric functions intact."""
        tracker = MetricsTracker()

        def fn(a, b):
            return 1.0

        tracker.register_metric("acc", fn)
        tracker.reset()
        assert "acc" in tracker.metric_fns

    def test_tracker_usable_after_reset(self):
        """Tracker can accumulate new values after reset."""
        tracker = MetricsTracker()
        tracker.update("loss/train", 0.5)
        tracker.reset()
        tracker.update("loss/train", 0.2)
        results = tracker.compute_metrics()
        assert results["loss/train"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# format_metrics — output routing
# ---------------------------------------------------------------------------


class TestFormatMetrics:
    def _metrics(self):
        return {"loss/train": 0.4, "loss/valid": 0.3}

    def test_uses_print_when_tqdm_false(self, capsys):
        """Output goes to stdout via print when use_tqdm=False."""
        format_metrics(epoch=1, metrics=self._metrics(), use_tqdm=False)
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_uses_tqdm_write_when_tqdm_true(self):
        """Output routes through tqdm.write when use_tqdm=True."""
        with patch("tqdm.tqdm.write") as mock_write:
            format_metrics(epoch=1, metrics=self._metrics(), use_tqdm=True)
            assert mock_write.call_count > 0

    def test_does_not_call_tqdm_when_tqdm_false(self):
        """tqdm.write is not called when use_tqdm=False."""
        with patch("tqdm.tqdm.write") as mock_write:
            format_metrics(epoch=1, metrics=self._metrics(), use_tqdm=False)
            mock_write.assert_not_called()


# ---------------------------------------------------------------------------
# format_metrics — content
# ---------------------------------------------------------------------------


class TestFormatMetricsContent:
    def _capture(self, epoch, metrics):
        lines = []
        with patch(
            "tqdm.tqdm.write", side_effect=lambda msg: lines.append(msg)
        ):
            format_metrics(epoch=epoch, metrics=metrics, use_tqdm=True)
        return lines

    def test_epoch_number_in_output(self):
        """Epoch number appears in the formatted output."""
        lines = self._capture(5, {"loss/train": 0.5, "loss/valid": 0.4})
        combined = "\n".join(lines)
        assert "5" in combined

    def test_metric_name_in_output(self):
        """Metric name appears in the formatted output."""
        lines = self._capture(1, {"loss/train": 0.5, "loss/valid": 0.4})
        combined = "\n".join(lines)
        assert "loss" in combined

    def test_train_value_formatted(self):
        """Train split value is formatted in the output."""
        lines = self._capture(1, {"loss/train": 0.1234, "loss/valid": 0.5678})
        combined = "\n".join(lines)
        assert "0.1234" in combined

    def test_valid_value_formatted(self):
        """Validation split value is formatted in the output."""
        lines = self._capture(1, {"loss/train": 0.1234, "loss/valid": 0.5678})
        combined = "\n".join(lines)
        assert "0.5678" in combined

    def test_separator_lines_present(self):
        """Separator lines are included in the output."""
        lines = self._capture(1, {"loss/train": 0.5, "loss/valid": 0.4})
        combined = "\n".join(lines)
        assert "=" * 10 in combined

    def test_train_label_in_output(self):
        """'Train' label appears in the output."""
        lines = self._capture(1, {"loss/train": 0.5, "loss/valid": 0.4})
        combined = "\n".join(lines)
        assert "Train" in combined

    def test_valid_label_in_output(self):
        """'Valid' label appears in the output."""
        lines = self._capture(1, {"loss/train": 0.5, "loss/valid": 0.4})
        combined = "\n".join(lines)
        assert "Valid" in combined

    def test_multiple_metrics_all_displayed(self):
        """All registered metric names appear in the output."""
        metrics = {
            "loss/train": 0.5,
            "loss/valid": 0.4,
            "accuracy/train": 0.8,
            "accuracy/valid": 0.75,
        }
        lines = self._capture(3, metrics)
        combined = "\n".join(lines)
        assert "loss" in combined
        assert "accuracy" in combined
