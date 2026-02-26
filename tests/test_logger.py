"""Tests for epoch_engine.core.logger module."""

from unittest.mock import patch

from epoch_engine.core.logger import TrainerLogger

# ---------------------------------------------------------------------------
# TrainerLogger — init
# ---------------------------------------------------------------------------


class TestTrainerLoggerInit:
    def test_default_enable_tqdm_is_true(self):
        """enable_tqdm defaults to True."""
        logger = TrainerLogger()
        assert logger.enable_tqdm is True

    def test_enable_tqdm_false(self):
        """enable_tqdm can be set to False."""
        logger = TrainerLogger(enable_tqdm=False)
        assert logger.enable_tqdm is False


# ---------------------------------------------------------------------------
# TrainerLogger — tqdm=False (uses print / sys.stderr)
# ---------------------------------------------------------------------------


class TestTrainerLoggerNoTqdm:
    def setup_method(self):
        self.logger = TrainerLogger(enable_tqdm=False)

    def test_info_prints_message(self, capsys):
        """info() writes the message to stdout."""
        self.logger.info("hello")
        assert capsys.readouterr().out.strip() == "hello"

    def test_warning_prints_to_stderr(self, capsys):
        """warning() writes with ⚠️ prefix to stderr."""
        self.logger.warning("something wrong")
        captured = capsys.readouterr()
        assert "something wrong" in captured.err
        assert "⚠️" in captured.err

    def test_error_prints_to_stderr(self, capsys):
        """error() writes with ❌ prefix to stderr."""
        self.logger.error("fatal issue")
        captured = capsys.readouterr()
        assert "fatal issue" in captured.err
        assert "❌" in captured.err

    def test_success_prints_to_stdout(self, capsys):
        """success() writes with ✅ prefix to stdout."""
        self.logger.success("all good")
        captured = capsys.readouterr()
        assert "all good" in captured.out
        assert "✅" in captured.out

    def test_info_does_not_write_to_stderr(self, capsys):
        """info() does not write to stderr."""
        self.logger.info("info msg")
        assert capsys.readouterr().err == ""

    def test_success_does_not_write_to_stderr(self, capsys):
        """success() does not write to stderr."""
        self.logger.success("ok")
        assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# TrainerLogger — tqdm=True (uses tqdm.write)
# ---------------------------------------------------------------------------


class TestTrainerLoggerWithTqdm:
    def setup_method(self):
        self.logger = TrainerLogger(enable_tqdm=True)

    def test_info_uses_tqdm_write(self):
        """info() routes through tqdm.write."""
        with patch("tqdm.tqdm.write") as mock_write:
            self.logger.info("tqdm info")
            mock_write.assert_called_once_with("tqdm info")

    def test_warning_uses_tqdm_write_with_prefix(self):
        """warning() routes through tqdm.write with ⚠️ prefix."""
        with patch("tqdm.tqdm.write") as mock_write:
            self.logger.warning("tqdm warn")
            args = mock_write.call_args[0][0]
            assert "tqdm warn" in args
            assert "⚠️" in args

    def test_error_uses_tqdm_write_with_prefix(self):
        """error() routes through tqdm.write with ❌ prefix."""
        with patch("tqdm.tqdm.write") as mock_write:
            self.logger.error("tqdm err")
            args = mock_write.call_args[0][0]
            assert "tqdm err" in args
            assert "❌" in args

    def test_success_uses_tqdm_write_with_prefix(self):
        """success() routes through tqdm.write with ✅ prefix."""
        with patch("tqdm.tqdm.write") as mock_write:
            self.logger.success("tqdm ok")
            args = mock_write.call_args[0][0]
            assert "tqdm ok" in args
            assert "✅" in args

    def test_tqdm_write_called_once_per_method(self):
        """Each log method calls tqdm.write exactly once."""
        for method in (
            self.logger.info,
            self.logger.warning,
            self.logger.error,
            self.logger.success,
        ):
            with patch("tqdm.tqdm.write") as mock_write:
                method("msg")
                assert mock_write.call_count == 1
