"""Tests for MetricsLogger and MetricsPlotter."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from epoch_engine.core.metrics.metrics_logger import MetricsLogger
from epoch_engine.core.metrics.metrics_plotter import MetricsPlotter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_history(path: Path, run_id: str, entries: list[dict]) -> None:
    """Write a minimal metrics_history.json to *path*."""
    data = {"runs": [{f"run_id={run_id}": entries}]}
    path.write_text(json.dumps(data))


def _make_entries(epochs: list[int]) -> list[dict]:
    """Return dummy metric dicts for the given epoch numbers."""
    return [
        {
            "loss/train": round(0.5 / e, 4),
            "loss/valid": round(0.4 / e, 4),
            "epoch": e,
        }
        for e in epochs
    ]


# ---------------------------------------------------------------------------
# MetricsLogger — init
# ---------------------------------------------------------------------------


class TestMetricsLoggerInit:
    def test_stores_run_id(self, tmp_path):
        """run_id is accessible after construction."""
        ml = MetricsLogger(run_id="abc", base_dir=str(tmp_path))
        assert ml.run_id == "abc"

    def test_default_truncate_is_none(self, tmp_path):
        """truncate_after_epoch defaults to None."""
        ml = MetricsLogger(run_id="abc", base_dir=str(tmp_path))
        assert ml.truncate_after_epoch is None

    def test_truncate_after_epoch_stored(self, tmp_path):
        """Explicit truncate_after_epoch is stored on the instance."""
        ml = MetricsLogger(
            run_id="abc", base_dir=str(tmp_path), truncate_after_epoch=5
        )
        assert ml.truncate_after_epoch == 5


# ---------------------------------------------------------------------------
# MetricsLogger — load_history
# ---------------------------------------------------------------------------


class TestMetricsLoggerLoadHistory:
    def test_creates_empty_structure_when_no_file(self, tmp_path):
        """Missing JSON file produces an empty run entry."""
        ml = MetricsLogger(run_id="abc", base_dir=str(tmp_path))
        ml.load_history()
        assert ml.current_run == {"run_id=abc": []}

    def test_loads_existing_run(self, tmp_path):
        """Existing entries for the run_id are loaded into current_run."""
        entries = _make_entries([1, 2, 3])
        _write_history(tmp_path / "metrics_history.json", "abc", entries)
        ml = MetricsLogger(run_id="abc", base_dir=str(tmp_path))
        ml.load_history()
        assert ml.current_run["run_id=abc"] == entries

    def test_creates_new_entry_for_unknown_run_id(self, tmp_path):
        """Unknown run_id gets a fresh empty entry appended."""
        _write_history(
            tmp_path / "metrics_history.json", "other", _make_entries([1])
        )
        ml = MetricsLogger(run_id="new", base_dir=str(tmp_path))
        ml.load_history()
        assert ml.current_run == {"run_id=new": []}

    def test_does_not_discard_other_runs_on_new_entry(self, tmp_path):
        """Creating a new run entry preserves all pre-existing runs."""
        _write_history(
            tmp_path / "metrics_history.json", "existing", _make_entries([1])
        )
        ml = MetricsLogger(run_id="new", base_dir=str(tmp_path))
        ml.load_history()
        all_keys = [list(r)[0] for r in ml.training_process["runs"]]
        assert "run_id=existing" in all_keys
        assert "run_id=new" in all_keys


# ---------------------------------------------------------------------------
# MetricsLogger — truncation
# ---------------------------------------------------------------------------


class TestMetricsLoggerTruncation:
    def test_no_truncation_when_param_is_none(self, tmp_path):
        """truncate_after_epoch=None leaves all entries intact."""
        entries = _make_entries([1, 2, 3, 4, 5])
        _write_history(tmp_path / "metrics_history.json", "abc", entries)
        ml = MetricsLogger(run_id="abc", base_dir=str(tmp_path))
        ml.load_history()
        assert [e["epoch"] for e in ml.current_run["run_id=abc"]] == [
            1,
            2,
            3,
            4,
            5,
        ]

    def test_drops_entries_beyond_cutoff(self, tmp_path):
        """Entries with epoch > truncate_after_epoch are removed."""
        entries = _make_entries([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        _write_history(tmp_path / "metrics_history.json", "abc", entries)
        ml = MetricsLogger(
            run_id="abc", base_dir=str(tmp_path), truncate_after_epoch=5
        )
        ml.load_history()
        assert [e["epoch"] for e in ml.current_run["run_id=abc"]] == [
            1,
            2,
            3,
            4,
            5,
        ]

    def test_cutoff_at_last_epoch_is_noop(self, tmp_path):
        """truncate_after_epoch equal to the last entry's epoch keeps everything."""
        entries = _make_entries([1, 2, 3])
        _write_history(tmp_path / "metrics_history.json", "abc", entries)
        ml = MetricsLogger(
            run_id="abc", base_dir=str(tmp_path), truncate_after_epoch=3
        )
        ml.load_history()
        assert len(ml.current_run["run_id=abc"]) == 3

    def test_cutoff_below_all_entries_yields_empty(self, tmp_path):
        """truncate_after_epoch=0 removes every entry (epochs are 1-indexed)."""
        entries = _make_entries([1, 2, 3])
        _write_history(tmp_path / "metrics_history.json", "abc", entries)
        ml = MetricsLogger(
            run_id="abc", base_dir=str(tmp_path), truncate_after_epoch=0
        )
        ml.load_history()
        assert ml.current_run["run_id=abc"] == []

    def test_truncation_on_brand_new_run_is_safe(self, tmp_path):
        """Truncating a run_id with no existing history causes no error."""
        ml = MetricsLogger(
            run_id="new", base_dir=str(tmp_path), truncate_after_epoch=5
        )
        ml.load_history()
        assert ml.current_run["run_id=new"] == []

    def test_removes_stale_entry_matching_resume_from_best_scenario(
        self, tmp_path
    ):
        """Simulates the bug: 6 epochs logged, resume from best.pt at epoch 5.

        Without truncation the resumed run would re-log epoch 6, duplicating it.
        With truncate_after_epoch=5 the stale epoch 6 is removed before the new
        training session appends its own epoch 6.
        """
        entries = _make_entries([1, 2, 3, 4, 5, 6])
        _write_history(tmp_path / "metrics_history.json", "abc", entries)
        ml = MetricsLogger(
            run_id="abc", base_dir=str(tmp_path), truncate_after_epoch=5
        )
        ml.load_history()
        epochs = [e["epoch"] for e in ml.current_run["run_id=abc"]]
        assert epochs == [1, 2, 3, 4, 5]
        assert 6 not in epochs

    def test_truncation_does_not_affect_other_runs(self, tmp_path):
        """Truncating one run leaves sibling runs untouched."""
        data = {
            "runs": [
                {"run_id=abc": _make_entries([1, 2, 3, 4, 5])},
                {"run_id=other": _make_entries([1, 2, 3, 4, 5])},
            ]
        }
        (tmp_path / "metrics_history.json").write_text(json.dumps(data))
        ml = MetricsLogger(
            run_id="abc", base_dir=str(tmp_path), truncate_after_epoch=3
        )
        ml.load_history()
        assert len(ml.current_run["run_id=abc"]) == 3
        other_entries = next(
            r["run_id=other"]
            for r in ml.training_process["runs"]
            if "run_id=other" in r
        )
        assert len(other_entries) == 5


# ---------------------------------------------------------------------------
# MetricsLogger — log_metrics
# ---------------------------------------------------------------------------


class TestMetricsLoggerLogMetrics:
    def test_appends_entry_to_current_run(self, tmp_path):
        """log_metrics adds the dict to the current run list."""
        ml = MetricsLogger(run_id="abc", base_dir=str(tmp_path))
        ml.load_history()
        ml.log_metrics({"loss/train": 0.5, "epoch": 1})
        assert ml.current_run["run_id=abc"] == [
            {"loss/train": 0.5, "epoch": 1}
        ]

    def test_multiple_calls_append_in_order(self, tmp_path):
        """Successive log_metrics calls preserve insertion order."""
        ml = MetricsLogger(run_id="abc", base_dir=str(tmp_path))
        ml.load_history()
        ml.log_metrics({"epoch": 1})
        ml.log_metrics({"epoch": 2})
        assert [e["epoch"] for e in ml.current_run["run_id=abc"]] == [1, 2]

    def test_appends_after_truncation(self, tmp_path):
        """New entries attach after the truncated history, not at the original end."""
        _write_history(
            tmp_path / "metrics_history.json",
            "abc",
            _make_entries([1, 2, 3, 4, 5]),
        )
        ml = MetricsLogger(
            run_id="abc", base_dir=str(tmp_path), truncate_after_epoch=3
        )
        ml.load_history()
        ml.log_metrics({"epoch": 4})
        assert [e["epoch"] for e in ml.current_run["run_id=abc"]] == [
            1,
            2,
            3,
            4,
        ]


# ---------------------------------------------------------------------------
# MetricsLogger — save_history
# ---------------------------------------------------------------------------


class TestMetricsLoggerSaveHistory:
    def test_writes_json_to_disk(self, tmp_path):
        """save_history persists the in-memory state to the JSON file."""
        ml = MetricsLogger(run_id="abc", base_dir=str(tmp_path))
        ml.load_history()
        ml.log_metrics({"epoch": 1})
        ml.save_history()
        saved = json.loads((tmp_path / "metrics_history.json").read_text())
        assert saved["runs"][0]["run_id=abc"][0]["epoch"] == 1

    def test_truncation_is_persisted_on_save(self, tmp_path):
        """After truncation + save, reloading the file shows no stale entries."""
        _write_history(
            tmp_path / "metrics_history.json",
            "abc",
            _make_entries([1, 2, 3, 4, 5]),
        )
        ml = MetricsLogger(
            run_id="abc", base_dir=str(tmp_path), truncate_after_epoch=3
        )
        ml.load_history()
        ml.save_history()
        saved = json.loads((tmp_path / "metrics_history.json").read_text())
        epochs = [e["epoch"] for e in saved["runs"][0]["run_id=abc"]]
        assert epochs == [1, 2, 3]


# ---------------------------------------------------------------------------
# MetricsLogger — context manager
# ---------------------------------------------------------------------------


class TestMetricsLoggerContextManager:
    def test_enter_returns_self(self, tmp_path):
        """__enter__ returns the MetricsLogger instance."""
        ml = MetricsLogger(run_id="abc", base_dir=str(tmp_path))
        with ml as ctx:
            assert ctx is ml

    def test_history_loaded_on_enter(self, tmp_path):
        """__enter__ calls load_history so current_run is populated."""
        _write_history(
            tmp_path / "metrics_history.json", "abc", _make_entries([1, 2])
        )
        ml = MetricsLogger(run_id="abc", base_dir=str(tmp_path))
        with ml:
            assert len(ml.current_run["run_id=abc"]) == 2

    def test_history_saved_on_clean_exit(self, tmp_path):
        """__exit__ saves history even when no exception occurred."""
        ml = MetricsLogger(run_id="abc", base_dir=str(tmp_path))
        with ml:
            ml.log_metrics({"epoch": 1})
        saved = json.loads((tmp_path / "metrics_history.json").read_text())
        assert saved["runs"][0]["run_id=abc"][0]["epoch"] == 1

    def test_history_saved_on_exception(self, tmp_path):
        """__exit__ saves history even when an exception propagates."""
        ml = MetricsLogger(run_id="abc", base_dir=str(tmp_path))
        try:
            with ml:
                ml.log_metrics({"epoch": 1})
                raise RuntimeError("simulated error")
        except RuntimeError:
            pass
        saved = json.loads((tmp_path / "metrics_history.json").read_text())
        assert saved["runs"][0]["run_id=abc"][0]["epoch"] == 1

    def test_truncation_applied_inside_context(self, tmp_path):
        """Truncation takes effect during the context so logged entries follow cleanly."""
        _write_history(
            tmp_path / "metrics_history.json",
            "abc",
            _make_entries([1, 2, 3, 4, 5]),
        )
        ml = MetricsLogger(
            run_id="abc", base_dir=str(tmp_path), truncate_after_epoch=3
        )
        with ml:
            ml.log_metrics({"epoch": 4})
            ml.log_metrics({"epoch": 5})
        saved = json.loads((tmp_path / "metrics_history.json").read_text())
        epochs = [e["epoch"] for e in saved["runs"][0]["run_id=abc"]]
        assert epochs == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# MetricsPlotter — _get_metric_values
# ---------------------------------------------------------------------------


class TestMetricsPlotterGetMetricValues:
    def _plotter(self, tmp_path: Path) -> MetricsPlotter:
        return MetricsPlotter(base_dir=str(tmp_path))

    def test_returns_values_for_run(self, tmp_path):
        """Correct float values are extracted for a given metric and run."""
        entries = [
            {"loss/train": 0.5, "loss/valid": 0.4, "epoch": 1},
            {"loss/train": 0.3, "loss/valid": 0.2, "epoch": 2},
        ]
        _write_history(tmp_path / "metrics_history.json", "abc", entries)
        values = self._plotter(tmp_path)._get_metric_values(
            "loss/train", "abc"
        )
        assert values == [0.5, 0.3]

    def test_returns_empty_list_for_unknown_run_id(self, tmp_path):
        """Non-existent run_id returns an empty list."""
        _write_history(
            tmp_path / "metrics_history.json", "abc", _make_entries([1, 2])
        )
        values = self._plotter(tmp_path)._get_metric_values(
            "loss/train", "unknown"
        )
        assert values == []

    def test_ignores_sibling_run_ids(self, tmp_path):
        """Values from other runs are excluded."""
        data = {
            "runs": [
                {"run_id=abc": [{"loss/train": 0.5, "epoch": 1}]},
                {"run_id=other": [{"loss/train": 0.9, "epoch": 1}]},
            ]
        }
        (tmp_path / "metrics_history.json").write_text(json.dumps(data))
        values = self._plotter(tmp_path)._get_metric_values(
            "loss/train", "abc"
        )
        assert values == [0.5]

    def test_returns_all_entries_in_insertion_order(self, tmp_path):
        """All entries are returned in order with no gaps."""
        entries = _make_entries([1, 2, 3, 4, 5])
        _write_history(tmp_path / "metrics_history.json", "abc", entries)
        values = self._plotter(tmp_path)._get_metric_values(
            "loss/train", "abc"
        )
        assert len(values) == 5
        assert values == [e["loss/train"] for e in entries]

    def test_duplicate_epoch_yields_extra_value(self, tmp_path):
        """Without truncation a duplicate epoch entry inflates the value list.

        This documents the pre-fix behaviour: 6 real epochs but 7 values returned,
        causing the plotter to draw a 7-point x-axis instead of 6.
        """
        entries = _make_entries([1, 2, 3, 4, 5, 6]) + [_make_entries([6])[0]]
        _write_history(tmp_path / "metrics_history.json", "abc", entries)
        values = self._plotter(tmp_path)._get_metric_values(
            "loss/train", "abc"
        )
        assert len(values) == 7


# ---------------------------------------------------------------------------
# MetricsPlotter — create_plot
# ---------------------------------------------------------------------------


def _all_plt_patches():
    """Context manager that stubs out every matplotlib.pyplot call in create_plot."""
    return (
        patch("matplotlib.pyplot.plot"),
        patch("matplotlib.pyplot.xlabel"),
        patch("matplotlib.pyplot.ylabel"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.legend"),
        patch("matplotlib.pyplot.tight_layout"),
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.close"),
    )


class TestMetricsPlotterCreatePlot:
    def _write_simple_history(
        self, path: Path, run_id: str, n_epochs: int
    ) -> None:
        entries = _make_entries(list(range(1, n_epochs + 1)))
        _write_history(path, run_id, entries)

    def test_creates_plot_directory(self, tmp_path):
        """create_plot creates the plots subdirectory when it doesn't exist."""
        self._write_simple_history(tmp_path / "metrics_history.json", "abc", 3)
        plotter = MetricsPlotter(base_dir=str(tmp_path))
        patches = _all_plt_patches()
        cms = [p.__enter__() for p in patches]
        try:
            plotter.create_plot("loss", "abc")
        finally:
            for p, cm in zip(patches, cms):
                p.__exit__(None, None, None)
        assert (tmp_path / "run_id=abc" / "plots").is_dir()

    def test_saves_png_at_expected_path(self, tmp_path):
        """savefig is called with a path ending in '<metric_name>.png'."""
        self._write_simple_history(tmp_path / "metrics_history.json", "abc", 3)
        plotter = MetricsPlotter(base_dir=str(tmp_path))
        mock_savefig = MagicMock()
        with (
            patch("matplotlib.pyplot.plot"),
            patch("matplotlib.pyplot.xlabel"),
            patch("matplotlib.pyplot.ylabel"),
            patch("matplotlib.pyplot.title"),
            patch("matplotlib.pyplot.legend"),
            patch("matplotlib.pyplot.tight_layout"),
            patch("matplotlib.pyplot.savefig", mock_savefig),
            patch("matplotlib.pyplot.close"),
        ):
            plotter.create_plot("loss", "abc")
        saved_path = str(mock_savefig.call_args[0][0])
        assert saved_path.endswith("loss.png")

    def test_x_axis_length_matches_epoch_count(self, tmp_path):
        """plt.plot is called with an x-axis of length equal to the number of logged epochs."""
        n_epochs = 5
        self._write_simple_history(
            tmp_path / "metrics_history.json", "abc", n_epochs
        )
        plotter = MetricsPlotter(base_dir=str(tmp_path))
        captured_x = []
        with (
            patch(
                "matplotlib.pyplot.plot",
                side_effect=lambda x, y, **kw: captured_x.append(list(x)),
            ),
            patch("matplotlib.pyplot.xlabel"),
            patch("matplotlib.pyplot.ylabel"),
            patch("matplotlib.pyplot.title"),
            patch("matplotlib.pyplot.legend"),
            patch("matplotlib.pyplot.tight_layout"),
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
        ):
            plotter.create_plot("loss", "abc")
        # First plt.plot call is the train curve
        assert captured_x[0] == list(np.arange(1, n_epochs + 1))

    def test_duplicate_epoch_inflates_x_axis(self, tmp_path):
        """Documents the pre-fix bug: a duplicate epoch 6 makes x-axis go to 7.

        After the MetricsLogger truncation fix this scenario no longer arises at
        log time, but the plotter itself has no guard — if stale data exists the
        plot will still be wrong.
        """
        entries = _make_entries([1, 2, 3, 4, 5, 6]) + [_make_entries([6])[0]]
        _write_history(tmp_path / "metrics_history.json", "abc", entries)
        plotter = MetricsPlotter(base_dir=str(tmp_path))
        captured_x = []
        with (
            patch(
                "matplotlib.pyplot.plot",
                side_effect=lambda x, y, **kw: captured_x.append(list(x)),
            ),
            patch("matplotlib.pyplot.xlabel"),
            patch("matplotlib.pyplot.ylabel"),
            patch("matplotlib.pyplot.title"),
            patch("matplotlib.pyplot.legend"),
            patch("matplotlib.pyplot.tight_layout"),
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
        ):
            plotter.create_plot("loss", "abc")
        assert len(captured_x[0]) == 7  # 7-point axis for 6 real epochs

    def test_plot_closed_after_save(self, tmp_path):
        """plt.close() is called after plt.savefig() to free memory."""
        self._write_simple_history(tmp_path / "metrics_history.json", "abc", 2)
        plotter = MetricsPlotter(base_dir=str(tmp_path))
        call_order = []
        with (
            patch("matplotlib.pyplot.plot"),
            patch("matplotlib.pyplot.xlabel"),
            patch("matplotlib.pyplot.ylabel"),
            patch("matplotlib.pyplot.title"),
            patch("matplotlib.pyplot.legend"),
            patch("matplotlib.pyplot.tight_layout"),
            patch(
                "matplotlib.pyplot.savefig",
                side_effect=lambda _: call_order.append("save"),
            ),
            patch(
                "matplotlib.pyplot.close",
                side_effect=lambda: call_order.append("close"),
            ),
        ):
            plotter.create_plot("loss", "abc")
        assert call_order == ["save", "close"]
