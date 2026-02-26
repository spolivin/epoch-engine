"""Tests for epoch_engine.core.configs module."""

from epoch_engine.core.configs import MetricConfig

# ---------------------------------------------------------------------------
# MetricConfig
# ---------------------------------------------------------------------------


class TestMetricConfig:
    def test_construction_with_required_fields(self):
        """name and fn are stored correctly on construction."""

        def fn(y_true, y_pred):
            return 0.0

        cfg = MetricConfig(name="accuracy", fn=fn)
        assert cfg.name == "accuracy"
        assert cfg.fn is fn

    def test_default_plot_is_false(self):
        """plot defaults to False when not explicitly provided."""
        cfg = MetricConfig(name="loss", fn=lambda a, b: 0.0)
        assert cfg.plot is False

    def test_plot_can_be_set_to_true(self):
        """plot can be explicitly set to True."""
        cfg = MetricConfig(name="f1", fn=lambda a, b: 0.0, plot=True)
        assert cfg.plot is True

    def test_fn_is_callable(self):
        """Stored fn is callable."""
        cfg = MetricConfig(name="acc", fn=lambda a, b: 1.0)
        assert callable(cfg.fn)
