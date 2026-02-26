"""Tests for epoch_engine.core.callbacks module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from epoch_engine.core.callbacks import (
    BestCheckpoint,
    Callback,
    CheckpointPruner,
    EarlyStopping,
    NanCallback,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer():
    """Creates a simple MagicMock trainer with a logger attribute."""
    trainer = MagicMock()
    trainer.logger = MagicMock()
    return trainer


# ---------------------------------------------------------------------------
# Callback (base class)
# ---------------------------------------------------------------------------


class TestCallback:
    def test_cannot_instantiate_directly(self):
        """Callback is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Callback()

    def test_subclass_can_be_instantiated(self):
        """A concrete subclass of Callback can be instantiated."""

        class DummyCallback(Callback):
            pass

        cb = DummyCallback()
        assert isinstance(cb, Callback)

    def test_all_hooks_return_none(self):
        """All default hook implementations return None."""

        class DummyCallback(Callback):
            pass

        cb = DummyCallback()
        trainer = _make_trainer()

        assert cb.on_train_start(trainer) is None
        assert cb.on_train_end(trainer) is None
        assert cb.on_epoch_start(trainer, epoch=1) is None
        assert cb.on_epoch_end(trainer, epoch=1, metrics={}) is None
        assert cb.on_batch_start(trainer, batch_idx=0) is None
        assert cb.on_batch_end(trainer, batch_idx=0, loss=0.5) is None
        assert cb.on_validation_start(trainer) is None
        assert cb.on_validation_end(trainer, valid_metrics={}) is None
        assert (
            cb.on_checkpoint_save(trainer, checkpoint_path=Path(".")) is None
        )


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------


class TestEarlyStoppingInit:
    def test_default_values(self):
        """Default attribute values match the documented defaults."""
        es = EarlyStopping()
        assert es.monitor == "loss/valid"
        assert es.patience == 5
        assert es.mode == "min"
        assert es.min_delta == 0.0
        assert es.verbose is True
        assert es.best_score is None
        assert es.counter == 0
        assert es.should_stop is False

    def test_custom_values(self):
        """Custom arguments are stored on the instance."""
        es = EarlyStopping(
            monitor="accuracy/valid",
            patience=3,
            mode="max",
            min_delta=0.01,
            verbose=False,
        )
        assert es.monitor == "accuracy/valid"
        assert es.patience == 3
        assert es.mode == "max"
        assert es.min_delta == 0.01
        assert es.verbose is False

    def test_invalid_mode_raises_value_error(self):
        """Passing an invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode"):
            EarlyStopping(mode="invalid")


class TestEarlyStoppingOnEpochEnd:
    def test_sets_best_score_on_first_call(self):
        """best_score is initialised on the first call."""
        es = EarlyStopping(monitor="loss/valid")
        es.on_epoch_end(_make_trainer(), epoch=1, metrics={"loss/valid": 0.5})
        assert es.best_score == 0.5

    def test_missing_metric_does_nothing(self):
        """Missing monitor metric leaves state unchanged."""
        es = EarlyStopping(monitor="loss/valid")
        es.on_epoch_end(_make_trainer(), epoch=1, metrics={"accuracy": 0.9})
        assert es.best_score is None
        assert es.counter == 0

    def test_improvement_in_min_mode_resets_counter(self):
        """Lower value in min mode updates best_score and resets counter."""
        es = EarlyStopping(monitor="loss/valid", mode="min")
        trainer = _make_trainer()
        es.on_epoch_end(trainer, epoch=1, metrics={"loss/valid": 0.5})
        es.on_epoch_end(trainer, epoch=2, metrics={"loss/valid": 0.4})
        assert es.best_score == 0.4
        assert es.counter == 0

    def test_improvement_in_max_mode_resets_counter(self):
        """Higher value in max mode updates best_score and resets counter."""
        es = EarlyStopping(monitor="acc/valid", mode="max")
        trainer = _make_trainer()
        es.on_epoch_end(trainer, epoch=1, metrics={"acc/valid": 0.7})
        es.on_epoch_end(trainer, epoch=2, metrics={"acc/valid": 0.8})
        assert es.best_score == 0.8
        assert es.counter == 0

    def test_no_improvement_increments_counter(self):
        """Non-improving value increments the patience counter."""
        es = EarlyStopping(monitor="loss/valid", patience=3)
        trainer = _make_trainer()
        es.on_epoch_end(trainer, epoch=1, metrics={"loss/valid": 0.5})
        es.on_epoch_end(trainer, epoch=2, metrics={"loss/valid": 0.6})
        assert es.counter == 1
        assert es.should_stop is False

    def test_should_stop_set_when_patience_exceeded(self):
        """should_stop becomes True after patience epochs without improvement."""
        es = EarlyStopping(monitor="loss/valid", patience=2)
        trainer = _make_trainer()
        es.on_epoch_end(trainer, epoch=1, metrics={"loss/valid": 0.5})
        es.on_epoch_end(trainer, epoch=2, metrics={"loss/valid": 0.6})
        es.on_epoch_end(trainer, epoch=3, metrics={"loss/valid": 0.6})
        assert es.should_stop is True

    def test_counter_resets_after_improvement(self):
        """Counter resets to zero after a subsequent improvement."""
        es = EarlyStopping(monitor="loss/valid", patience=3)
        trainer = _make_trainer()
        es.on_epoch_end(trainer, epoch=1, metrics={"loss/valid": 0.5})
        es.on_epoch_end(
            trainer, epoch=2, metrics={"loss/valid": 0.6}
        )  # no improvement
        es.on_epoch_end(
            trainer, epoch=3, metrics={"loss/valid": 0.3}
        )  # improvement
        assert es.counter == 0
        assert es.best_score == 0.3

    def test_verbose_false_does_not_log(self):
        """No warning is logged when verbose=False."""
        es = EarlyStopping(monitor="loss/valid", patience=1, verbose=False)
        trainer = _make_trainer()
        es.on_epoch_end(trainer, epoch=1, metrics={"loss/valid": 0.5})
        es.on_epoch_end(trainer, epoch=2, metrics={"loss/valid": 0.6})
        es.on_epoch_end(trainer, epoch=3, metrics={"loss/valid": 0.7})
        trainer.logger.warning.assert_not_called()

    def test_verbose_true_logs_on_no_improvement(self):
        """Warning is logged on no-improvement when verbose=True."""
        es = EarlyStopping(monitor="loss/valid", patience=3, verbose=True)
        trainer = _make_trainer()
        es.on_epoch_end(trainer, epoch=1, metrics={"loss/valid": 0.5})
        es.on_epoch_end(trainer, epoch=2, metrics={"loss/valid": 0.6})
        trainer.logger.warning.assert_called()

    def test_min_delta_respected_in_min_mode(self):
        """Change smaller than min_delta is not counted as improvement (min mode)."""
        es = EarlyStopping(monitor="loss/valid", mode="min", min_delta=0.1)
        trainer = _make_trainer()
        es.on_epoch_end(trainer, epoch=1, metrics={"loss/valid": 0.5})
        # 0.45 is better than 0.5 but not by more than min_delta=0.1
        es.on_epoch_end(trainer, epoch=2, metrics={"loss/valid": 0.45})
        assert es.counter == 1  # not counted as improvement

    def test_min_delta_respected_in_max_mode(self):
        """Change smaller than min_delta is not counted as improvement (max mode)."""
        es = EarlyStopping(monitor="acc/valid", mode="max", min_delta=0.1)
        trainer = _make_trainer()
        es.on_epoch_end(trainer, epoch=1, metrics={"acc/valid": 0.7})
        # 0.75 is better than 0.7 but not by more than min_delta=0.1
        es.on_epoch_end(trainer, epoch=2, metrics={"acc/valid": 0.75})
        assert es.counter == 1  # not counted as improvement


class TestEarlyStoppingIsImprovement:
    def test_min_mode_strictly_less(self):
        """Returns True only when current is strictly less than best minus delta."""
        es = EarlyStopping(mode="min", min_delta=0.0)
        es.best_score = 0.5
        assert es._is_improvement(0.4) is True
        assert es._is_improvement(0.5) is False
        assert es._is_improvement(0.6) is False

    def test_max_mode_strictly_greater(self):
        """Returns True only when current is strictly greater than best plus delta."""
        es = EarlyStopping(mode="max", min_delta=0.0)
        es.best_score = 0.7
        assert es._is_improvement(0.8) is True
        assert es._is_improvement(0.7) is False
        assert es._is_improvement(0.6) is False

    def test_min_mode_with_min_delta(self):
        """min_delta creates a threshold below best_score."""
        es = EarlyStopping(mode="min", min_delta=0.05)
        es.best_score = 0.5
        assert es._is_improvement(0.44) is True  # 0.44 < 0.5 - 0.05 = 0.45
        assert es._is_improvement(0.46) is False  # 0.46 >= 0.45

    def test_max_mode_with_min_delta(self):
        """min_delta creates a threshold above best_score."""
        es = EarlyStopping(mode="max", min_delta=0.05)
        es.best_score = 0.7
        assert es._is_improvement(0.76) is True  # 0.76 > 0.7 + 0.05 = 0.75
        assert es._is_improvement(0.74) is False  # 0.74 <= 0.75


# ---------------------------------------------------------------------------
# NanCallback
# ---------------------------------------------------------------------------


class TestNanCallbackInit:
    def test_default_values(self):
        """Default attribute values match documented defaults."""
        cb = NanCallback()
        assert cb.monitor is None
        assert cb.verbose is True
        assert cb.should_stop is False

    def test_custom_values(self):
        """Custom monitor list and verbose flag are stored."""
        cb = NanCallback(monitor=["loss/train"], verbose=False)
        assert cb.monitor == ["loss/train"]
        assert cb.verbose is False


class TestNanCallbackOnEpochEnd:
    def test_no_nan_does_not_stop(self):
        """Clean metrics do not trigger should_stop."""
        cb = NanCallback()
        cb.on_epoch_end(
            _make_trainer(),
            epoch=0,
            metrics={"loss/train": 0.5, "loss/valid": 0.6},
        )
        assert cb.should_stop is False

    def test_nan_in_train_loss_sets_should_stop(self):
        """NaN in loss/train sets should_stop to True."""
        cb = NanCallback()
        cb.on_epoch_end(
            _make_trainer(),
            epoch=0,
            metrics={"loss/train": float("nan"), "loss/valid": 0.6},
        )
        assert cb.should_stop is True

    def test_nan_in_valid_loss_sets_should_stop(self):
        """NaN in loss/valid sets should_stop to True."""
        cb = NanCallback()
        cb.on_epoch_end(
            _make_trainer(),
            epoch=0,
            metrics={"loss/train": 0.5, "loss/valid": float("nan")},
        )
        assert cb.should_stop is True

    def test_nan_in_custom_metric_sets_should_stop(self):
        """NaN in any float metric sets should_stop to True."""
        cb = NanCallback()
        cb.on_epoch_end(
            _make_trainer(),
            epoch=0,
            metrics={"loss/train": 0.5, "accuracy/train": float("nan")},
        )
        assert cb.should_stop is True

    def test_epoch_int_key_not_flagged(self):
        """Integer 'epoch' key is ignored when scanning for NaN."""
        cb = NanCallback()
        cb.on_epoch_end(
            _make_trainer(), epoch=0, metrics={"loss/train": 0.5, "epoch": 1}
        )
        assert cb.should_stop is False

    def test_monitor_restricts_to_specified_keys(self):
        """When monitor list is set, NaN in unmonitored keys is ignored."""
        cb = NanCallback(monitor=["loss/valid"])
        # NaN only in an unmonitored key — should not stop
        cb.on_epoch_end(
            _make_trainer(),
            epoch=0,
            metrics={"loss/train": float("nan"), "loss/valid": 0.6},
        )
        assert cb.should_stop is False

    def test_monitor_detects_nan_in_specified_key(self):
        """NaN in a monitored key sets should_stop to True."""
        cb = NanCallback(monitor=["loss/valid"])
        cb.on_epoch_end(
            _make_trainer(),
            epoch=0,
            metrics={"loss/train": 0.5, "loss/valid": float("nan")},
        )
        assert cb.should_stop is True

    def test_monitor_key_missing_does_not_error(self):
        """Missing monitored key is silently skipped."""
        cb = NanCallback(monitor=["loss/valid"])
        # monitored key absent — should silently skip
        cb.on_epoch_end(_make_trainer(), epoch=0, metrics={"loss/train": 0.5})
        assert cb.should_stop is False

    def test_verbose_true_calls_logger_warning(self):
        """A warning is logged when NaN is detected and verbose=True."""
        cb = NanCallback(verbose=True)
        trainer = _make_trainer()
        cb.on_epoch_end(trainer, epoch=2, metrics={"loss/train": float("nan")})
        trainer.logger.warning.assert_called_once()

    def test_verbose_false_does_not_log(self):
        """No warning is logged when verbose=False."""
        cb = NanCallback(verbose=False)
        trainer = _make_trainer()
        cb.on_epoch_end(trainer, epoch=2, metrics={"loss/train": float("nan")})
        trainer.logger.warning.assert_not_called()

    def test_no_nan_does_not_log(self):
        """No error is logged when no NaN is present."""
        cb = NanCallback(verbose=True)
        trainer = _make_trainer()
        cb.on_epoch_end(trainer, epoch=0, metrics={"loss/train": 0.5})
        trainer.logger.error.assert_not_called()


# ---------------------------------------------------------------------------
# BestCheckpoint
# ---------------------------------------------------------------------------


class TestBestCheckpointInit:
    def test_default_values(self):
        """Default attribute values match documented defaults."""
        cb = BestCheckpoint()
        assert cb.monitor == "loss/valid"
        assert cb.mode == "min"
        assert cb.min_delta == 0.0
        assert cb.verbose is True
        assert cb.best_score is None

    def test_custom_values(self):
        """Custom arguments are stored on the instance."""
        cb = BestCheckpoint(
            monitor="accuracy/valid", mode="max", min_delta=0.01, verbose=False
        )
        assert cb.monitor == "accuracy/valid"
        assert cb.mode == "max"
        assert cb.min_delta == 0.01
        assert cb.verbose is False

    def test_invalid_mode_raises_value_error(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode"):
            BestCheckpoint(mode="invalid")


class TestBestCheckpointOnEpochEnd:
    def test_sets_best_score_on_first_call(self):
        """best_score is initialised on the first call."""
        cb = BestCheckpoint(monitor="loss/valid")
        cb.on_epoch_end(_make_trainer(), epoch=0, metrics={"loss/valid": 0.5})
        assert cb.best_score == 0.5

    def test_sets_new_best_true_on_first_call(self):
        """_new_best is True after the first call."""
        cb = BestCheckpoint(monitor="loss/valid")
        cb.on_epoch_end(_make_trainer(), epoch=0, metrics={"loss/valid": 0.5})
        assert cb._new_best is True

    def test_improvement_min_mode_updates_best_and_sets_flag(self):
        """Improvement in min mode updates best_score and sets _new_best."""
        cb = BestCheckpoint(monitor="loss/valid", mode="min")
        trainer = _make_trainer()
        cb.on_epoch_end(trainer, epoch=0, metrics={"loss/valid": 0.5})
        cb.on_epoch_end(trainer, epoch=1, metrics={"loss/valid": 0.3})
        assert cb.best_score == 0.3
        assert cb._new_best is True

    def test_improvement_max_mode_updates_best_and_sets_flag(self):
        """Improvement in max mode updates best_score and sets _new_best."""
        cb = BestCheckpoint(monitor="accuracy/valid", mode="max")
        trainer = _make_trainer()
        cb.on_epoch_end(trainer, epoch=0, metrics={"accuracy/valid": 0.7})
        cb.on_epoch_end(trainer, epoch=1, metrics={"accuracy/valid": 0.9})
        assert cb.best_score == 0.9
        assert cb._new_best is True

    def test_no_improvement_does_not_update_best_score(self):
        """Non-improving value leaves best_score unchanged."""
        cb = BestCheckpoint(monitor="loss/valid", mode="min")
        trainer = _make_trainer()
        cb.on_epoch_end(trainer, epoch=0, metrics={"loss/valid": 0.5})
        cb.on_epoch_end(trainer, epoch=1, metrics={"loss/valid": 0.7})
        assert cb.best_score == 0.5

    def test_no_improvement_sets_new_best_false(self):
        """Non-improving value sets _new_best to False."""
        cb = BestCheckpoint(monitor="loss/valid", mode="min")
        trainer = _make_trainer()
        cb.on_epoch_end(trainer, epoch=0, metrics={"loss/valid": 0.5})
        cb.on_epoch_end(trainer, epoch=1, metrics={"loss/valid": 0.7})
        assert cb._new_best is False

    def test_missing_metric_does_not_set_flag(self):
        """Missing monitor metric leaves _new_best unset."""
        cb = BestCheckpoint(monitor="loss/valid")
        cb.on_epoch_end(
            _make_trainer(), epoch=1, metrics={"accuracy/valid": 0.9}
        )
        assert not hasattr(cb, "_new_best")

    def test_missing_metric_warns_on_epoch_1_when_verbose(self):
        """Warning is logged on epoch 1 when metric is missing and verbose=True."""
        cb = BestCheckpoint(monitor="loss/valid", verbose=True)
        trainer = _make_trainer()
        cb.on_epoch_end(trainer, epoch=1, metrics={"accuracy/valid": 0.9})
        trainer.logger.warning.assert_called_once()

    def test_missing_metric_no_warning_after_epoch_1(self):
        """No warning logged after epoch 1 for a missing metric."""
        cb = BestCheckpoint(monitor="loss/valid", verbose=True)
        trainer = _make_trainer()
        cb.on_epoch_end(trainer, epoch=2, metrics={"accuracy/valid": 0.9})
        trainer.logger.warning.assert_not_called()

    def test_missing_metric_no_warning_when_verbose_false(self):
        """No warning logged when verbose=False even if metric is missing."""
        cb = BestCheckpoint(monitor="loss/valid", verbose=False)
        trainer = _make_trainer()
        cb.on_epoch_end(trainer, epoch=1, metrics={"accuracy/valid": 0.9})
        trainer.logger.warning.assert_not_called()

    def test_min_delta_respected(self):
        """Improvement smaller than min_delta does not update best_score."""
        cb = BestCheckpoint(monitor="loss/valid", mode="min", min_delta=0.1)
        trainer = _make_trainer()
        cb.on_epoch_end(trainer, epoch=0, metrics={"loss/valid": 0.5})
        # 0.45 is better than 0.5 but not by more than min_delta=0.1
        cb.on_epoch_end(trainer, epoch=1, metrics={"loss/valid": 0.45})
        assert cb.best_score == 0.5
        assert cb._new_best is False


class TestBestCheckpointOnCheckpointSave:
    def test_copies_to_best_pt_when_new_best(self, tmp_path):
        """best.pt is created when _new_best is True."""
        ckpt_path = tmp_path / "ckpt_epoch_1.pt"
        ckpt_path.touch()
        cb = BestCheckpoint(monitor="loss/valid", verbose=False)
        cb._new_best = True
        cb.best_score = 0.3
        cb.on_checkpoint_save(_make_trainer(), checkpoint_path=ckpt_path)
        assert (tmp_path / "best.pt").exists()

    def test_skips_when_not_new_best(self, tmp_path):
        """best.pt is not created when _new_best is False."""
        ckpt_path = tmp_path / "ckpt_epoch_1.pt"
        ckpt_path.touch()
        cb = BestCheckpoint(monitor="loss/valid")
        cb._new_best = False
        cb.on_checkpoint_save(_make_trainer(), checkpoint_path=ckpt_path)
        assert not (tmp_path / "best.pt").exists()

    def test_skips_when_flag_not_set(self, tmp_path):
        """best.pt is not created when _new_best has never been set."""
        ckpt_path = tmp_path / "ckpt_epoch_1.pt"
        ckpt_path.touch()
        cb = BestCheckpoint(monitor="loss/valid")
        cb.on_checkpoint_save(_make_trainer(), checkpoint_path=ckpt_path)
        assert not (tmp_path / "best.pt").exists()

    def test_verbose_logs_when_new_best(self, tmp_path):
        """A log message is emitted when a new best is saved and verbose=True."""
        ckpt_path = tmp_path / "ckpt_epoch_1.pt"
        ckpt_path.touch()
        cb = BestCheckpoint(monitor="loss/valid", verbose=True)
        cb._new_best = True
        cb.best_score = 0.3
        trainer = _make_trainer()
        cb.on_checkpoint_save(trainer, checkpoint_path=ckpt_path)
        trainer.logger.info.assert_called_once()

    def test_verbose_false_does_not_log(self, tmp_path):
        """No log message is emitted when verbose=False."""
        ckpt_path = tmp_path / "ckpt_epoch_1.pt"
        ckpt_path.touch()
        cb = BestCheckpoint(monitor="loss/valid", verbose=False)
        cb._new_best = True
        cb.best_score = 0.3
        trainer = _make_trainer()
        cb.on_checkpoint_save(trainer, checkpoint_path=ckpt_path)
        trainer.logger.info.assert_not_called()

    def test_overwrites_previous_best(self, tmp_path):
        """best.pt is overwritten with the latest checkpoint content."""
        cb = BestCheckpoint(monitor="loss/valid", verbose=False)
        cb.best_score = 0.3
        cb._new_best = True
        trainer = _make_trainer()
        for epoch in (1, 2):
            ckpt_path = tmp_path / f"ckpt_epoch_{epoch}.pt"
            ckpt_path.write_bytes(b"epoch_%d" % epoch)
            cb.on_checkpoint_save(trainer, checkpoint_path=ckpt_path)
        assert (tmp_path / "best.pt").read_bytes() == b"epoch_2"


class TestBestCheckpointIsImprovement:
    def test_min_mode_strictly_less(self):
        """Returns True only when current is strictly less than best minus delta."""
        cb = BestCheckpoint(mode="min", min_delta=0.0)
        cb.best_score = 0.5
        assert cb._is_improvement(0.4) is True
        assert cb._is_improvement(0.5) is False
        assert cb._is_improvement(0.6) is False

    def test_max_mode_strictly_greater(self):
        """Returns True only when current is strictly greater than best plus delta."""
        cb = BestCheckpoint(mode="max", min_delta=0.0)
        cb.best_score = 0.7
        assert cb._is_improvement(0.8) is True
        assert cb._is_improvement(0.7) is False
        assert cb._is_improvement(0.6) is False

    def test_min_mode_with_min_delta(self):
        """min_delta creates a threshold below best_score."""
        cb = BestCheckpoint(mode="min", min_delta=0.05)
        cb.best_score = 0.5
        assert cb._is_improvement(0.44) is True  # 0.44 < 0.5 - 0.05
        assert cb._is_improvement(0.46) is False  # 0.46 >= 0.45

    def test_max_mode_with_min_delta(self):
        """min_delta creates a threshold above best_score."""
        cb = BestCheckpoint(mode="max", min_delta=0.05)
        cb.best_score = 0.7
        assert cb._is_improvement(0.76) is True  # 0.76 > 0.7 + 0.05
        assert cb._is_improvement(0.74) is False  # 0.74 <= 0.75


# ---------------------------------------------------------------------------
# CheckpointPruner
# ---------------------------------------------------------------------------


class TestCheckpointPrunerInit:
    def test_default_keep_last_n(self):
        """keep_last_n defaults to 1."""
        cb = CheckpointPruner()
        assert cb.keep_last_n == 1

    def test_custom_keep_last_n(self):
        """Custom keep_last_n is stored."""
        cb = CheckpointPruner(keep_last_n=3)
        assert cb.keep_last_n == 3

    def test_zero_raises_value_error(self):
        """keep_last_n=0 raises ValueError."""
        with pytest.raises(ValueError, match="keep_last_n"):
            CheckpointPruner(keep_last_n=0)


class TestCheckpointPrunerOnCheckpointSave:
    def _make_epoch_ckpts(self, directory: Path, n: int) -> list[Path]:
        paths = []
        for i in range(1, n + 1):
            p = directory / f"ckpt_epoch_{i}.pt"
            p.touch()
            paths.append(p)
        return paths

    def test_deletes_old_checkpoints_beyond_keep_last_1(self, tmp_path):
        """Only the most recent checkpoint is kept with keep_last_n=1."""
        paths = self._make_epoch_ckpts(tmp_path, 3)
        cb = CheckpointPruner(keep_last_n=1)
        cb.on_checkpoint_save(_make_trainer(), checkpoint_path=paths[-1])
        assert not paths[0].exists()
        assert not paths[1].exists()
        assert paths[2].exists()

    def test_keeps_last_n_when_n_is_2(self, tmp_path):
        """Only the two most recent checkpoints are kept with keep_last_n=2."""
        paths = self._make_epoch_ckpts(tmp_path, 4)
        cb = CheckpointPruner(keep_last_n=2)
        cb.on_checkpoint_save(_make_trainer(), checkpoint_path=paths[-1])
        assert not paths[0].exists()
        assert not paths[1].exists()
        assert paths[2].exists()
        assert paths[3].exists()

    def test_does_nothing_when_fewer_checkpoints_than_keep_last_n(
        self, tmp_path
    ):
        """No deletion occurs when checkpoint count is below keep_last_n."""
        paths = self._make_epoch_ckpts(tmp_path, 2)
        cb = CheckpointPruner(keep_last_n=3)
        cb.on_checkpoint_save(_make_trainer(), checkpoint_path=paths[-1])
        assert all(p.exists() for p in paths)

    def test_does_not_touch_best_pt(self, tmp_path):
        """best.pt is never deleted by the pruner."""
        paths = self._make_epoch_ckpts(tmp_path, 3)
        best = tmp_path / "best.pt"
        best.touch()
        cb = CheckpointPruner(keep_last_n=1)
        cb.on_checkpoint_save(_make_trainer(), checkpoint_path=paths[-1])
        assert best.exists()

    def test_single_checkpoint_is_never_deleted(self, tmp_path):
        """A single checkpoint is always kept regardless of keep_last_n."""
        paths = self._make_epoch_ckpts(tmp_path, 1)
        cb = CheckpointPruner(keep_last_n=1)
        cb.on_checkpoint_save(_make_trainer(), checkpoint_path=paths[-1])
        assert paths[0].exists()
