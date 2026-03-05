"""Tests for epoch_engine.core.trainer module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from epoch_engine.core.callbacks import Callback, EarlyStopping
from epoch_engine.core.configs import MetricConfig
from epoch_engine.core.trainer import Trainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model():
    """Creates a simple linear model for testing."""
    return nn.Linear(4, 2)


def _make_dataloader(n=8, features=4, num_classes=2, batch_size=4):
    """Creates a DataLoader with random data for testing."""
    dataset = TensorDataset(
        torch.randn(n, features), torch.randint(0, num_classes, (n,))
    )
    return DataLoader(dataset, batch_size=batch_size)


def _make_optimizer(model=None):
    """Creates a simple SGD optimizer for testing."""
    if model is None:
        model = _make_model()
    return optim.SGD(model.parameters(), lr=0.01)


def _make_trainer(**kwargs):
    """Creates a Trainer instance with default components, allowing overrides via kwargs."""
    model = kwargs.pop("model", _make_model())
    defaults = dict(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=kwargs.pop("optimizer", _make_optimizer(model)),
        train_loader=_make_dataloader(),
        valid_loader=_make_dataloader(),
    )
    defaults.update(kwargs)
    return Trainer(**defaults)


# ---------------------------------------------------------------------------
# Trainer.__init__
# ---------------------------------------------------------------------------


class TestTrainerInit:
    def test_valid_construction_stores_attributes(self):
        """Core attributes are stored correctly on construction."""
        model = _make_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = _make_optimizer(model)
        train_loader = _make_dataloader()
        valid_loader = _make_dataloader()
        trainer = Trainer(
            model, criterion, optimizer, train_loader, valid_loader
        )
        assert trainer.criterion is criterion
        assert trainer.optimizer is optimizer
        assert trainer.train_loader is train_loader
        assert trainer.valid_loader is valid_loader

    def test_model_is_nn_module(self):
        """model is an nn.Module instance."""
        trainer = _make_trainer()
        assert isinstance(trainer.model, nn.Module)

    def test_scheduler_defaults_to_none(self):
        """scheduler and scheduler_level default to None when not provided."""
        trainer = _make_trainer()
        assert trainer.scheduler is None
        assert trainer.scheduler_level is None

    def test_scheduler_stored_when_provided(self):
        """Provided scheduler is stored with default level 'epoch'."""
        model = _make_model()
        optimizer = _make_optimizer(model)
        scheduler = StepLR(optimizer, step_size=1)
        trainer = _make_trainer(
            model=model, optimizer=optimizer, scheduler=scheduler
        )
        assert trainer.scheduler is scheduler
        assert trainer.scheduler_level == "epoch"

    def test_scheduler_level_batch(self):
        """scheduler_level='batch' is stored correctly."""
        model = _make_model()
        optimizer = _make_optimizer(model)
        scheduler = StepLR(optimizer, step_size=1)
        trainer = _make_trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_level="batch",
        )
        assert trainer.scheduler_level == "batch"

    def test_invalid_scheduler_level_raises_value_error(self):
        """Invalid scheduler_level raises ValueError."""
        model = _make_model()
        optimizer = _make_optimizer(model)
        scheduler = StepLR(optimizer, step_size=1)
        with pytest.raises(ValueError, match="scheduler_level"):
            _make_trainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scheduler_level="step",
            )

    def test_last_epoch_defaults_to_zero(self):
        """last_epoch starts at 0."""
        trainer = _make_trainer()
        assert trainer.last_epoch == 0

    def test_extra_metrics_defaults_to_false(self):
        """extra_metrics flag defaults to False."""
        trainer = _make_trainer()
        assert trainer.extra_metrics is False

    def test_metrics_to_plot_contains_loss_by_default(self):
        """metrics_to_plot always includes 'loss'."""
        trainer = _make_trainer()
        assert trainer.metrics_to_plot == ["loss"]

    def test_metrics_registered_from_dict(self):
        """Dict metrics set extra_metrics and populate metrics_to_plot."""
        trainer = _make_trainer(metrics={"accuracy": lambda a, b: 0.0})
        assert trainer.extra_metrics is True
        assert "accuracy" in trainer.metrics_to_plot

    def test_metrics_registered_from_list_with_plot(self):
        """MetricConfig with plot=True is added to metrics_to_plot."""
        metric_cfg = MetricConfig(name="f1", fn=lambda a, b: 0.0, plot=True)
        trainer = _make_trainer(metrics=[metric_cfg])
        assert trainer.extra_metrics is True
        assert "f1" in trainer.metrics_to_plot

    def test_metrics_list_plot_false_not_in_metrics_to_plot(self):
        """MetricConfig with plot=False is excluded from metrics_to_plot."""
        metric_cfg = MetricConfig(name="f1", fn=lambda a, b: 0.0, plot=False)
        trainer = _make_trainer(metrics=[metric_cfg])
        assert "f1" not in trainer.metrics_to_plot

    def test_callbacks_defaults_to_empty_list(self):
        """callbacks defaults to an empty list."""
        trainer = _make_trainer()
        assert trainer.callbacks == []

    def test_interrupted_defaults_to_false(self):
        """interrupted defaults to False."""
        trainer = _make_trainer()
        assert trainer.interrupted is False

    def test_logger_defaults_to_none(self):
        """logger defaults to None before run() is called."""
        trainer = _make_trainer()
        assert trainer.logger is None

    def test_test_loader_stored_when_provided(self):
        """Optional test_loader is stored when provided."""
        test_loader = _make_dataloader()
        trainer = _make_trainer(test_loader=test_loader)
        assert trainer.test_loader is test_loader

    def test_callbacks_stored_when_provided(self):
        """Provided callbacks list is stored on the instance."""

        class DummyCb(Callback):
            pass

        cb = DummyCb()
        trainer = _make_trainer(callbacks=[cb])
        assert cb in trainer.callbacks

    def test_invalid_model_raises_type_error(self):
        """Non-Module model raises TypeError."""
        with pytest.raises(TypeError, match="model"):
            Trainer(
                model="not_a_model",
                criterion=nn.CrossEntropyLoss(),
                optimizer=_make_optimizer(),
                train_loader=_make_dataloader(),
                valid_loader=_make_dataloader(),
            )

    def test_non_callable_criterion_raises_type_error(self):
        """Non-callable criterion raises TypeError."""
        with pytest.raises(TypeError, match="criterion"):
            Trainer(
                model=_make_model(),
                criterion=42,
                optimizer=_make_optimizer(),
                train_loader=_make_dataloader(),
                valid_loader=_make_dataloader(),
            )

    def test_invalid_optimizer_raises_type_error(self):
        """Non-optimizer raises TypeError."""
        with pytest.raises(TypeError, match="optimizer"):
            Trainer(
                model=_make_model(),
                criterion=nn.CrossEntropyLoss(),
                optimizer="not_an_optimizer",
                train_loader=_make_dataloader(),
                valid_loader=_make_dataloader(),
            )

    def test_invalid_train_loader_raises_type_error(self):
        """Non-DataLoader train_loader raises TypeError."""
        with pytest.raises(TypeError):
            Trainer(
                model=_make_model(),
                criterion=nn.CrossEntropyLoss(),
                optimizer=_make_optimizer(),
                train_loader="not_a_loader",
                valid_loader=_make_dataloader(),
            )

    def test_invalid_valid_loader_raises_type_error(self):
        """Non-DataLoader valid_loader raises TypeError."""
        with pytest.raises(TypeError):
            Trainer(
                model=_make_model(),
                criterion=nn.CrossEntropyLoss(),
                optimizer=_make_optimizer(),
                train_loader=_make_dataloader(),
                valid_loader="not_a_loader",
            )

    def test_invalid_test_loader_raises_type_error(self):
        """Non-DataLoader test_loader raises TypeError."""
        with pytest.raises(TypeError, match="test_loader"):
            Trainer(
                model=_make_model(),
                criterion=nn.CrossEntropyLoss(),
                optimizer=_make_optimizer(),
                train_loader=_make_dataloader(),
                valid_loader=_make_dataloader(),
                test_loader="not_a_loader",
            )

    def test_amp_false_does_not_create_scaler(self):
        """No GradScaler is created when enable_amp=False."""
        trainer = _make_trainer(enable_amp=False)
        assert not hasattr(trainer, "scaler")


# ---------------------------------------------------------------------------
# Trainer.reset_scheduler
# ---------------------------------------------------------------------------


class TestResetScheduler:
    def test_sets_scheduler_to_none(self):
        """reset_scheduler() sets scheduler to None."""
        model = _make_model()
        optimizer = _make_optimizer(model)
        scheduler = StepLR(optimizer, step_size=1)
        trainer = _make_trainer(
            model=model, optimizer=optimizer, scheduler=scheduler
        )
        assert trainer.scheduler is not None
        trainer.reset_scheduler()
        assert trainer.scheduler is None

    def test_sets_scheduler_level_to_none(self):
        """reset_scheduler() sets scheduler_level to None."""
        model = _make_model()
        optimizer = _make_optimizer(model)
        scheduler = StepLR(optimizer, step_size=1)
        trainer = _make_trainer(
            model=model, optimizer=optimizer, scheduler=scheduler
        )
        trainer.reset_scheduler()
        assert trainer.scheduler_level is None

    def test_reset_when_already_none_is_safe(self):
        """reset_scheduler() is safe when scheduler was never set."""
        trainer = _make_trainer()
        trainer.reset_scheduler()  # scheduler was never set
        assert trainer.scheduler is None
        assert trainer.scheduler_level is None


# ---------------------------------------------------------------------------
# Trainer.__call__
# ---------------------------------------------------------------------------


class TestTrainerCall:
    def test_returns_tuple_of_loss_and_outputs(self):
        """__call__ returns a (loss, outputs) tuple."""
        trainer = _make_trainer()
        x = torch.randn(4, 4)
        y = torch.randint(0, 2, (4,))
        result = trainer(x, y)
        assert len(result) == 2

    def test_loss_is_scalar_tensor(self):
        """Loss is a scalar (0-dimensional) tensor."""
        trainer = _make_trainer()
        x = torch.randn(4, 4)
        y = torch.randint(0, 2, (4,))
        loss, _ = trainer(x, y)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_outputs_shape_matches_model(self):
        """Outputs shape matches the model's output dimensions."""
        trainer = _make_trainer()  # Linear(4, 2)
        x = torch.randn(4, 4)
        y = torch.randint(0, 2, (4,))
        _, outputs = trainer(x, y)
        assert outputs.shape == (4, 2)

    def test_loss_value_is_finite(self):
        """Loss value is finite."""
        trainer = _make_trainer()
        x = torch.randn(4, 4)
        y = torch.randint(0, 2, (4,))
        loss, _ = trainer(x, y)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Trainer._register_metrics
# ---------------------------------------------------------------------------


class TestRegisterMetrics:
    def test_dict_input_sets_extra_metrics_flag(self):
        """Dict input sets extra_metrics to True."""
        trainer = _make_trainer()
        trainer._register_metrics({"accuracy": lambda a, b: 0.0})
        assert trainer.extra_metrics is True

    def test_list_input_sets_extra_metrics_flag(self):
        """List input sets extra_metrics to True."""
        trainer = _make_trainer()
        trainer._register_metrics([MetricConfig("acc", lambda a, b: 0.0)])
        assert trainer.extra_metrics is True

    def test_dict_metrics_are_all_added_to_metrics_to_plot(self):
        """All dict keys are added to metrics_to_plot."""
        trainer = _make_trainer()
        trainer._register_metrics(
            {"acc": lambda a, b: 0.0, "f1": lambda a, b: 0.0}
        )
        assert "acc" in trainer.metrics_to_plot
        assert "f1" in trainer.metrics_to_plot

    def test_list_metric_with_plot_true_is_added_to_metrics_to_plot(self):
        """MetricConfig with plot=True is added to metrics_to_plot."""
        trainer = _make_trainer()
        trainer._register_metrics(
            [MetricConfig("f1", lambda a, b: 0.0, plot=True)]
        )
        assert "f1" in trainer.metrics_to_plot

    def test_list_metric_with_plot_false_not_added_to_metrics_to_plot(self):
        """MetricConfig with plot=False is excluded from metrics_to_plot."""
        trainer = _make_trainer()
        trainer._register_metrics(
            [MetricConfig("f1", lambda a, b: 0.0, plot=False)]
        )
        assert "f1" not in trainer.metrics_to_plot

    def test_non_callable_metric_raises_type_error(self):
        """Non-callable metric function raises TypeError."""
        trainer = _make_trainer()
        with pytest.raises(TypeError):
            trainer._register_metrics([MetricConfig("bad", "not_callable")])

    def test_metric_registered_in_tracker(self):
        """Metric function is stored in the metrics tracker."""
        trainer = _make_trainer()

        def fn(a, b):
            return 0.0

        trainer._register_metrics({"accuracy": fn})
        assert "accuracy" in trainer.metrics_tracker.metric_fns
        assert trainer.metrics_tracker.metric_fns["accuracy"] is fn

    def test_loss_still_in_metrics_to_plot_after_registration(self):
        """'loss' remains in metrics_to_plot after adding extra metrics."""
        trainer = _make_trainer()
        trainer._register_metrics({"accuracy": lambda a, b: 0.0})
        assert "loss" in trainer.metrics_to_plot


# ---------------------------------------------------------------------------
# Trainer._call_callbacks
# ---------------------------------------------------------------------------


class TestCallCallbacks:
    def test_calls_hook_on_all_callbacks(self):
        """Named hook is called on every registered callback."""

        class DummyCb(Callback):
            def on_epoch_end(self, trainer, epoch, metrics):
                pass

        cb1 = MagicMock(spec=DummyCb)
        cb2 = MagicMock(spec=DummyCb)
        trainer = _make_trainer(callbacks=[cb1, cb2])
        trainer._call_callbacks("on_epoch_end", epoch=0, metrics={})
        cb1.on_epoch_end.assert_called_once_with(
            trainer=trainer, epoch=0, metrics={}
        )
        cb2.on_epoch_end.assert_called_once_with(
            trainer=trainer, epoch=0, metrics={}
        )

    def test_no_callbacks_does_not_raise(self):
        """Empty callback list does not raise."""
        trainer = _make_trainer(callbacks=[])
        trainer._call_callbacks("on_epoch_end", epoch=0, metrics={})

    def test_passes_trainer_reference_to_hook(self):
        """The trainer instance is passed as first argument to the hook."""

        class DummyCb(Callback):
            def on_train_start(self, trainer):
                pass

        cb = MagicMock(spec=DummyCb)
        trainer = _make_trainer(callbacks=[cb])
        trainer._call_callbacks("on_train_start")
        cb.on_train_start.assert_called_once_with(trainer=trainer)

    def test_missing_hook_attribute_is_skipped(self):
        """Objects without the requested hook attribute are silently skipped."""

        # Object without the hook: getattr returns None, should be skipped
        class NoHookObj:
            pass

        trainer = _make_trainer()
        trainer.callbacks = [NoHookObj()]
        trainer._call_callbacks(
            "on_epoch_end", epoch=0, metrics={}
        )  # must not raise


# ---------------------------------------------------------------------------
# Trainer._collect_metrics
# ---------------------------------------------------------------------------


class TestCollectMetrics:
    def test_updates_loss_in_tracker(self):
        """Batch loss is recorded in the metrics tracker."""
        trainer = _make_trainer()
        loss = torch.tensor(0.5)
        outputs = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,))
        trainer._collect_metrics(
            loss=loss, outputs=outputs, labels=labels, split="train"
        )
        assert "loss/train" in trainer.metrics_tracker.metrics
        assert (
            abs(trainer.metrics_tracker.metrics["loss/train"][0] - 0.5) < 1e-6
        )

    def test_loss_split_key_matches_split_argument(self):
        """Loss key includes the correct split suffix."""
        trainer = _make_trainer()
        loss = torch.tensor(0.3)
        outputs = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,))
        trainer._collect_metrics(
            loss=loss, outputs=outputs, labels=labels, split="valid"
        )
        assert "loss/valid" in trainer.metrics_tracker.metrics

    def test_preds_not_updated_when_no_extra_metrics(self):
        """preds dict is not updated when extra_metrics is False."""
        trainer = _make_trainer()
        loss = torch.tensor(0.5)
        outputs = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,))
        trainer._collect_metrics(
            loss=loss, outputs=outputs, labels=labels, split="train"
        )
        assert trainer.metrics_tracker.preds == {}

    def test_preds_updated_when_extra_metrics_enabled(self):
        """preds dict is updated for the split when extra_metrics is True."""
        trainer = _make_trainer()
        trainer.extra_metrics = True
        loss = torch.tensor(0.5)
        outputs = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,))
        trainer._collect_metrics(
            loss=loss, outputs=outputs, labels=labels, split="train"
        )
        assert "train" in trainer.metrics_tracker.preds


# ---------------------------------------------------------------------------
# Trainer.evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_raises_value_error_if_no_test_loader_set(self):
        """ValueError raised when no test loader is available."""
        trainer = _make_trainer()
        with pytest.raises(ValueError, match="test loader"):
            trainer.evaluate()

    def test_raises_type_error_for_invalid_loader_arg(self):
        """TypeError raised when a non-DataLoader is passed."""
        trainer = _make_trainer()
        with pytest.raises(TypeError, match="loader"):
            trainer.evaluate(loader="not_a_loader")

    def test_uses_stored_test_loader_when_no_arg_given(self):
        """Stored test_loader is used when evaluate() is called with no args."""
        test_loader = _make_dataloader()
        trainer = _make_trainer(test_loader=test_loader)
        trainer._process_batch = MagicMock()
        trainer.metrics_tracker = MagicMock()
        trainer.metrics_tracker.compute_metrics.return_value = {
            "loss/test": 0.5
        }
        result = trainer.evaluate()
        assert "loss/test" in result

    def test_uses_provided_loader_when_given(self):
        """Explicitly passed loader takes precedence over stored test_loader."""
        trainer = _make_trainer()
        loader = _make_dataloader()
        trainer._process_batch = MagicMock()
        trainer.metrics_tracker = MagicMock()
        trainer.metrics_tracker.compute_metrics.return_value = {
            "loss/test": 0.3
        }
        result = trainer.evaluate(loader=loader)
        assert isinstance(result, dict)

    def test_calls_process_batch_for_every_batch(self):
        """_process_batch is called once per batch."""
        trainer = _make_trainer()
        loader = _make_dataloader(n=8, batch_size=4)  # 2 batches
        trainer._process_batch = MagicMock()
        trainer.metrics_tracker = MagicMock()
        trainer.metrics_tracker.compute_metrics.return_value = {}
        trainer.evaluate(loader=loader)
        assert trainer._process_batch.call_count == 2

    def test_resets_tracker_after_evaluation(self):
        """metrics_tracker.reset() is called after evaluation completes."""
        trainer = _make_trainer()
        loader = _make_dataloader()
        trainer._process_batch = MagicMock()
        trainer.metrics_tracker = MagicMock()
        trainer.metrics_tracker.compute_metrics.return_value = {}
        trainer.evaluate(loader=loader)
        trainer.metrics_tracker.reset.assert_called_once()

    def test_returns_dict(self):
        """evaluate() returns a dict."""
        trainer = _make_trainer()
        loader = _make_dataloader()
        trainer._process_batch = MagicMock()
        trainer.metrics_tracker = MagicMock()
        trainer.metrics_tracker.compute_metrics.return_value = {
            "loss/test": 0.5
        }
        result = trainer.evaluate(loader=loader)
        assert isinstance(result, dict)

    def test_process_batch_called_with_test_split(self):
        """_process_batch is called with split='test'."""
        trainer = _make_trainer()
        loader = _make_dataloader(n=4, batch_size=4)  # 1 batch
        trainer._process_batch = MagicMock()
        trainer.metrics_tracker = MagicMock()
        trainer.metrics_tracker.compute_metrics.return_value = {}
        trainer.evaluate(loader=loader)
        _, call_kwargs = trainer._process_batch.call_args
        assert (
            call_kwargs.get("split") == "test"
            or trainer._process_batch.call_args[0][1] == "test"
        )


# ---------------------------------------------------------------------------
# Trainer.run  (all I/O mocked)
# ---------------------------------------------------------------------------


class TestTrainerRun:
    """Tests for the `run` method with external dependencies mocked."""

    def _make_run_trainer(self, callbacks=None):
        return _make_trainer(callbacks=callbacks or [])

    def _make_run_info(self, last_logged_epoch=0):
        run_info = MagicMock()
        run_info.run_id = "test123"
        run_info.ckpts_path = MagicMock()
        run_info.last_logged_epoch = last_logged_epoch
        return run_info

    def _setup_mocks(self, trainer, run_info):
        trainer.run_mgr = MagicMock()
        trainer.run_mgr.init_run.return_value = run_info
        trainer.metrics_tracker = MagicMock()
        trainer.metrics_tracker.compute_metrics.return_value = {
            "loss/train": 0.5,
            "loss/valid": 0.4,
        }
        trainer.metrics_plotter = MagicMock()
        trainer._run_one_epoch = MagicMock()

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_calls_train_and_validate_per_epoch(
        self, MockLogger, MockMetricsLogger, mock_format_metrics
    ):
        """_run_one_epoch is called twice (train + valid) for each epoch."""
        trainer = self._make_run_trainer()
        run_info = self._make_run_info()
        self._setup_mocks(trainer, run_info)

        trainer.run(epochs=3, enable_tqdm=False)

        assert trainer._run_one_epoch.call_count == 6

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_last_epoch_updated_after_run(
        self, MockLogger, MockMetricsLogger, mock_format_metrics
    ):
        """last_epoch equals the number of epochs after training completes."""
        trainer = self._make_run_trainer()
        run_info = self._make_run_info()
        self._setup_mocks(trainer, run_info)

        trainer.run(epochs=2, enable_tqdm=False)

        assert trainer.last_epoch == 2

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_saves_checkpoint_each_epoch(
        self, MockLogger, MockMetricsLogger, mock_format_metrics
    ):
        """A checkpoint is saved after every epoch."""
        trainer = self._make_run_trainer()
        run_info = self._make_run_info()
        self._setup_mocks(trainer, run_info)

        trainer.run(epochs=3, enable_tqdm=False)

        assert trainer.run_mgr.save_checkpoint.call_count == 3

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_run_id_set_from_run_info(
        self, MockLogger, MockMetricsLogger, mock_format_metrics
    ):
        """run_id is taken from the RunInfo returned by init_run."""
        trainer = self._make_run_trainer()
        run_info = self._make_run_info()
        self._setup_mocks(trainer, run_info)

        trainer.run(epochs=1, enable_tqdm=False)

        assert trainer.run_id == "test123"

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_epoch_end_callbacks_called_each_epoch(
        self, MockLogger, MockMetricsLogger, mock_format_metrics
    ):
        """on_epoch_end hook is called once per epoch."""

        class DummyCb(Callback):
            def on_epoch_end(self, trainer, epoch, metrics):
                pass

        cb = MagicMock(spec=DummyCb)
        trainer = self._make_run_trainer(callbacks=[cb])
        run_info = self._make_run_info()
        self._setup_mocks(trainer, run_info)

        trainer.run(epochs=2, enable_tqdm=False)

        assert cb.on_epoch_end.call_count == 2

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_early_stopping_halts_training(
        self, MockLogger, MockMetricsLogger, mock_format_metrics
    ):
        """EarlyStopping callback halts training before all epochs complete."""
        es = EarlyStopping(monitor="loss/valid", patience=1)
        trainer = self._make_run_trainer(callbacks=[es])
        run_info = self._make_run_info()
        self._setup_mocks(trainer, run_info)

        call_count = 0

        def side_effect():
            nonlocal call_count
            call_count += 1
            # First epoch: good, second epoch: worse (triggers early stopping)
            return {
                "loss/train": 0.5,
                "loss/valid": 0.5 if call_count == 1 else 0.6,
            }

        trainer.metrics_tracker.compute_metrics.side_effect = side_effect

        trainer.run(epochs=10, enable_tqdm=False)

        assert trainer.interrupted is True
        assert trainer.last_epoch < 10

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_metric_plots_created_in_finally(
        self, MockLogger, MockMetricsLogger, mock_format_metrics
    ):
        """Metric plots are generated in the finally block after training."""
        trainer = self._make_run_trainer()
        run_info = self._make_run_info()
        self._setup_mocks(trainer, run_info)

        trainer.run(epochs=1, enable_tqdm=False)

        trainer.metrics_plotter.create_plot.assert_called_once_with(
            metric_name="loss", run_id="test123"
        )

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_metrics_tracker_reset_after_each_epoch(
        self, MockLogger, MockMetricsLogger, mock_format_metrics
    ):
        """Metrics tracker is reset after every epoch."""
        trainer = self._make_run_trainer()
        run_info = self._make_run_info()
        self._setup_mocks(trainer, run_info)

        trainer.run(epochs=3, enable_tqdm=False)

        assert trainer.metrics_tracker.reset.call_count == 3

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_run_id_passed_to_init_run(
        self, MockLogger, MockMetricsLogger, mock_format_metrics
    ):
        """Explicit run_id argument is forwarded to init_run."""
        trainer = self._make_run_trainer()
        run_info = self._make_run_info()
        self._setup_mocks(trainer, run_info)

        trainer.run(epochs=1, run_id="my_run", enable_tqdm=False)

        trainer.run_mgr.init_run.assert_called_once_with(run_id="my_run")

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_additional_metric_plots_created_when_registered(
        self, MockLogger, MockMetricsLogger, mock_format_metrics
    ):
        """Plots are created for all registered metrics."""
        trainer = self._make_run_trainer()
        trainer._register_metrics(
            [MetricConfig("accuracy", lambda a, b: 0.0, plot=True)]
        )
        run_info = self._make_run_info()
        self._setup_mocks(trainer, run_info)

        trainer.run(epochs=1, enable_tqdm=False)

        plot_calls = [
            call[1]["metric_name"]
            for call in trainer.metrics_plotter.create_plot.call_args_list
        ]
        assert "loss" in plot_calls
        assert "accuracy" in plot_calls


# ---------------------------------------------------------------------------
# Trainer.run  — resume_from_best
# ---------------------------------------------------------------------------


class TestTrainerRunResumeBest:
    """Tests for trainer.run(resume_from_best=True) with I/O mocked."""

    def _make_run_info(self, tmp_path, last_logged_epoch=6):
        """Return a RunInfo mock whose ckpts_path is a real temp directory."""
        run_info = MagicMock()
        run_info.run_id = "test123"
        ckpts_path = tmp_path / "checkpoints"
        ckpts_path.mkdir()
        run_info.ckpts_path = ckpts_path
        run_info.last_logged_epoch = last_logged_epoch
        return run_info

    def _setup_mocks(self, trainer, run_info, best_epoch):
        """Wire up all mocks; load_checkpoint returns a real-structure checkpoint."""
        trainer.run_mgr = MagicMock()
        trainer.run_mgr.init_run.return_value = run_info
        trainer.run_mgr.load_checkpoint.return_value = {
            "epoch": best_epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
        }
        trainer.metrics_tracker = MagicMock()
        trainer.metrics_tracker.compute_metrics.return_value = {
            "loss/train": 0.5,
            "loss/valid": 0.4,
        }
        trainer.metrics_plotter = MagicMock()
        trainer._run_one_epoch = MagicMock()

    # --- checkpoint path selection ------------------------------------------

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_loads_best_pt_when_it_exists(
        self, MockLogger, MockMetricsLogger, mock_format_metrics, tmp_path
    ):
        """load_checkpoint is called with best.pt when it exists and resume_from_best=True."""
        trainer = _make_trainer()
        run_info = self._make_run_info(tmp_path, last_logged_epoch=6)
        self._setup_mocks(trainer, run_info, best_epoch=4)
        best_pt = run_info.ckpts_path / "best.pt"
        best_pt.touch()

        trainer.run(epochs=2, resume_from_best=True, enable_tqdm=False)

        trainer.run_mgr.load_checkpoint.assert_called_once_with(path=best_pt)

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_falls_back_to_last_epoch_ckpt_when_best_pt_missing(
        self, MockLogger, MockMetricsLogger, mock_format_metrics, tmp_path
    ):
        """Falls back to ckpt_epoch_N.pt when best.pt does not exist."""
        trainer = _make_trainer()
        run_info = self._make_run_info(tmp_path, last_logged_epoch=6)
        self._setup_mocks(trainer, run_info, best_epoch=6)
        # best.pt intentionally not created

        trainer.run(epochs=2, resume_from_best=True, enable_tqdm=False)

        expected = run_info.ckpts_path / "ckpt_epoch_6.pt"
        trainer.run_mgr.load_checkpoint.assert_called_once_with(path=expected)

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_normal_resume_loads_last_epoch_checkpoint(
        self, MockLogger, MockMetricsLogger, mock_format_metrics, tmp_path
    ):
        """Without resume_from_best the last epoch checkpoint is always loaded."""
        trainer = _make_trainer()
        run_info = self._make_run_info(tmp_path, last_logged_epoch=6)
        self._setup_mocks(trainer, run_info, best_epoch=6)
        # even if best.pt is present it must be ignored
        (run_info.ckpts_path / "best.pt").touch()

        trainer.run(epochs=2, resume_from_best=False, enable_tqdm=False)

        expected = run_info.ckpts_path / "ckpt_epoch_6.pt"
        trainer.run_mgr.load_checkpoint.assert_called_once_with(path=expected)

    # --- last_epoch / training continuation ---------------------------------

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_last_epoch_set_from_best_checkpoint(
        self, MockLogger, MockMetricsLogger, mock_format_metrics, tmp_path
    ):
        """self.last_epoch is set to checkpoint['epoch'] when resuming from best.pt."""
        trainer = _make_trainer()
        run_info = self._make_run_info(tmp_path, last_logged_epoch=6)
        self._setup_mocks(trainer, run_info, best_epoch=4)
        (run_info.ckpts_path / "best.pt").touch()

        trainer.run(epochs=2, resume_from_best=True, enable_tqdm=False)

        assert trainer.last_epoch == 6  # 4 (best) + 2 epochs

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_exactly_requested_epochs_trained_from_best(
        self, MockLogger, MockMetricsLogger, mock_format_metrics, tmp_path
    ):
        """Exactly `epochs` training iterations run after resuming from best.pt."""
        trainer = _make_trainer()
        run_info = self._make_run_info(tmp_path, last_logged_epoch=6)
        self._setup_mocks(trainer, run_info, best_epoch=4)
        (run_info.ckpts_path / "best.pt").touch()

        trainer.run(epochs=3, resume_from_best=True, enable_tqdm=False)

        assert trainer._run_one_epoch.call_count == 6

    # --- MetricsLogger truncation -------------------------------------------

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_metrics_logger_truncate_set_to_best_epoch(
        self, MockLogger, MockMetricsLogger, mock_format_metrics, tmp_path
    ):
        """MetricsLogger receives truncate_after_epoch equal to the best.pt epoch."""
        trainer = _make_trainer()
        run_info = self._make_run_info(tmp_path, last_logged_epoch=6)
        self._setup_mocks(trainer, run_info, best_epoch=4)
        (run_info.ckpts_path / "best.pt").touch()

        trainer.run(epochs=2, resume_from_best=True, enable_tqdm=False)

        MockMetricsLogger.assert_called_once_with(
            run_id="test123", truncate_after_epoch=4
        )

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_metrics_logger_truncate_none_for_new_run(
        self, MockLogger, MockMetricsLogger, mock_format_metrics
    ):
        """MetricsLogger receives truncate_after_epoch=None on a brand-new run."""
        trainer = _make_trainer()
        run_info = MagicMock()
        run_info.run_id = "test123"
        run_info.ckpts_path = MagicMock()
        run_info.last_logged_epoch = 0
        trainer.run_mgr = MagicMock()
        trainer.run_mgr.init_run.return_value = run_info
        trainer.metrics_tracker = MagicMock()
        trainer.metrics_tracker.compute_metrics.return_value = {
            "loss/train": 0.5,
            "loss/valid": 0.4,
        }
        trainer.metrics_plotter = MagicMock()
        trainer._run_one_epoch = MagicMock()

        trainer.run(epochs=1, enable_tqdm=False)

        MockMetricsLogger.assert_called_once_with(
            run_id="test123", truncate_after_epoch=None
        )

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_metrics_logger_truncate_is_noop_on_normal_resume(
        self, MockLogger, MockMetricsLogger, mock_format_metrics, tmp_path
    ):
        """On normal resume truncate_after_epoch equals last_epoch — effectively a no-op."""
        trainer = _make_trainer()
        run_info = self._make_run_info(tmp_path, last_logged_epoch=5)
        self._setup_mocks(trainer, run_info, best_epoch=5)

        trainer.run(epochs=2, resume_from_best=False, enable_tqdm=False)

        MockMetricsLogger.assert_called_once_with(
            run_id="test123", truncate_after_epoch=5
        )

    # --- warning logging ----------------------------------------------------

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_warning_when_best_pt_missing(
        self, MockLogger, MockMetricsLogger, mock_format_metrics, tmp_path
    ):
        """A warning is logged when resume_from_best=True but best.pt does not exist."""
        trainer = _make_trainer()
        run_info = self._make_run_info(tmp_path, last_logged_epoch=6)
        self._setup_mocks(trainer, run_info, best_epoch=6)
        # best.pt intentionally not created

        trainer.run(epochs=1, resume_from_best=True, enable_tqdm=False)

        trainer.logger.warning.assert_called_once()

    @patch("epoch_engine.core.trainer.format_metrics")
    @patch("epoch_engine.core.trainer.MetricsLogger")
    @patch("epoch_engine.core.trainer.TrainerLogger")
    def test_no_warning_when_best_pt_exists(
        self, MockLogger, MockMetricsLogger, mock_format_metrics, tmp_path
    ):
        """No warning is logged when best.pt is present and resume_from_best=True."""
        trainer = _make_trainer()
        run_info = self._make_run_info(tmp_path, last_logged_epoch=6)
        self._setup_mocks(trainer, run_info, best_epoch=4)
        (run_info.ckpts_path / "best.pt").touch()

        trainer.run(epochs=1, resume_from_best=True, enable_tqdm=False)

        trainer.logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# task parameter
# ---------------------------------------------------------------------------


def _make_regression_dataloader(n=8, features=4, batch_size=4):
    """Creates a DataLoader with continuous targets for regression testing."""
    dataset = TensorDataset(torch.randn(n, features), torch.randn(n))
    return DataLoader(dataset, batch_size=batch_size)


class TestTaskParameter:
    def test_default_task_is_classification(self):
        """task defaults to 'classification'."""
        trainer = _make_trainer()
        assert trainer.task == "classification"

    def test_regression_task_stored(self):
        """task='regression' is stored on the instance."""
        model = nn.Linear(4, 1)
        trainer = _make_trainer(
            model=model,
            task="regression",
            train_loader=_make_regression_dataloader(),
            valid_loader=_make_regression_dataloader(),
        )
        assert trainer.task == "regression"

    def test_invalid_task_raises_value_error(self):
        """Invalid task string raises ValueError."""
        with pytest.raises(ValueError, match="task"):
            _make_trainer(task="segmentation")

    def test_classification_call_casts_targets_to_long(self):
        """Classification __call__ produces finite loss with integer targets."""
        trainer = _make_trainer()
        x = torch.randn(4, 4)
        y = torch.randint(0, 2, (4,))
        loss, _ = trainer(x, y)
        assert torch.isfinite(loss)

    def test_regression_call_produces_finite_loss(self):
        """Regression __call__ produces finite loss with float targets."""
        model = nn.Linear(4, 1)
        trainer = _make_trainer(
            model=model,
            criterion=nn.MSELoss(),
            task="regression",
            train_loader=_make_regression_dataloader(),
            valid_loader=_make_regression_dataloader(),
        )
        x = torch.randn(4, 4)
        y = torch.randn(4)
        loss, _ = trainer(x, y)
        assert torch.isfinite(loss)

    def test_regression_collect_metrics_stores_raw_outputs(self):
        """Regression _collect_metrics stores squeezed outputs, not argmax."""
        model = nn.Linear(4, 1)
        trainer = _make_trainer(
            model=model,
            criterion=nn.MSELoss(),
            task="regression",
            train_loader=_make_regression_dataloader(),
            valid_loader=_make_regression_dataloader(),
        )
        trainer.extra_metrics = True
        outputs = torch.tensor([[0.1], [0.5], [0.9], [0.3]])
        labels = torch.randn(4)
        trainer._collect_metrics(
            loss=torch.tensor(0.1),
            outputs=outputs,
            labels=labels,
            split="train",
        )
        preds = trainer.metrics_tracker.preds["train"]
        # Should be raw values, not argmax (which would be all 0s)
        assert any(abs(p - 0.0) > 1e-3 for p in preds)

    def test_regression_call_batch_size_one_no_broadcast_error(self):
        """squeeze(-1) prevents scalar collapse when batch_size=1."""
        model = nn.Linear(4, 1)
        trainer = _make_trainer(
            model=model,
            criterion=nn.MSELoss(),
            task="regression",
            train_loader=_make_regression_dataloader(),
            valid_loader=_make_regression_dataloader(),
        )
        x = torch.randn(1, 4)  # batch_size=1 → output shape (1, 1)
        y = torch.randn(1)  # target shape (1,)
        loss, outputs = trainer(x, y)
        assert outputs.shape == (1, 1)
        assert torch.isfinite(loss)

    def test_classification_collect_metrics_stores_argmax(self):
        """Classification _collect_metrics stores argmax of outputs."""
        trainer = _make_trainer()
        trainer.extra_metrics = True
        # High confidence for class 1
        outputs = torch.tensor(
            [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.05, 0.95]]
        )
        labels = torch.randint(0, 2, (4,))
        trainer._collect_metrics(
            loss=torch.tensor(0.1),
            outputs=outputs,
            labels=labels,
            split="train",
        )
        preds = trainer.metrics_tracker.preds["train"]
        assert all(p == 1 for p in preds)
