"""Tests for epoch_engine.core.managers module."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from epoch_engine.core.managers import DeviceManager, RunManager
from epoch_engine.core.managers.run_manager import RunInfo

# ---------------------------------------------------------------------------
# RunInfo (dataclass)
# ---------------------------------------------------------------------------


class TestRunInfo:
    def test_fields_stored(self, tmp_path):
        """All three fields are stored correctly."""
        info = RunInfo(
            run_id="abc123", ckpts_path=tmp_path, last_logged_epoch=3
        )
        assert info.run_id == "abc123"
        assert info.ckpts_path == tmp_path
        assert info.last_logged_epoch == 3


# ---------------------------------------------------------------------------
# DeviceManager — init
# ---------------------------------------------------------------------------


class TestDeviceManagerInit:
    def test_default_enable_amp_false(self):
        """enable_amp defaults to False."""
        dm = DeviceManager()
        assert dm.enable_amp is False

    def test_device_is_torch_device(self):
        """Detected device is a torch.device instance."""
        dm = DeviceManager()
        assert isinstance(dm.device, torch.device)

    def test_enable_amp_false_on_cpu(self):
        """enable_amp=False is stored as-is on CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            dm = DeviceManager(enable_amp=False)
        assert dm.enable_amp is False

    def test_enable_amp_true_raises_on_cpu(self):
        """enable_amp=True raises RuntimeError when CUDA is unavailable."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="CUDA"):
                DeviceManager(enable_amp=True)

    def test_enable_amp_true_on_cuda(self):
        """enable_amp=True succeeds and configures a CUDA device."""
        with patch("torch.cuda.is_available", return_value=True):
            dm = DeviceManager(enable_amp=True)
        assert dm.enable_amp is True
        assert dm.device.type == "cuda"


# ---------------------------------------------------------------------------
# DeviceManager — _auto_detect_device
# ---------------------------------------------------------------------------


class TestDeviceManagerAutoDetect:
    def test_returns_cuda_when_available(self):
        """CUDA device returned when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            dm = DeviceManager()
        assert dm.device.type == "cuda"

    def test_returns_cpu_when_nothing_available(self):
        """CPU device returned as final fallback."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            dm = DeviceManager()
        assert dm.device.type == "cpu"

    def test_returns_mps_when_cuda_unavailable(self):
        """MPS device returned when CUDA is absent but MPS is available."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            dm = DeviceManager()
        assert dm.device.type == "mps"


# ---------------------------------------------------------------------------
# DeviceManager — set_seed
# ---------------------------------------------------------------------------


class TestDeviceManagerSetSeed:
    def test_set_seed_cpu_calls_manual_seed(self):
        """torch.manual_seed called with the given seed on CPU."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            dm = DeviceManager()
        with patch("torch.manual_seed") as mock_seed:
            dm.set_seed(42)
            mock_seed.assert_called_once_with(42)

    def test_set_seed_cuda_calls_cuda_manual_seed(self):
        """torch.cuda.manual_seed called with the given seed on CUDA."""
        with patch("torch.cuda.is_available", return_value=True):
            dm = DeviceManager()
        with patch("torch.cuda.manual_seed") as mock_seed:
            dm.set_seed(7)
            mock_seed.assert_called_once_with(7)

    def test_set_seed_default_value(self):
        """Default seed of 42 is used when no argument is provided."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            dm = DeviceManager()
        with patch("torch.manual_seed") as mock_seed:
            dm.set_seed()
            mock_seed.assert_called_once_with(42)


# ---------------------------------------------------------------------------
# DeviceManager — move_to_device
# ---------------------------------------------------------------------------


class TestDeviceManagerMoveToDevice:
    def setup_method(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            self.dm = DeviceManager()  # cpu device

    def test_moves_tensor_to_device(self):
        """Tensor is moved to the configured device."""
        t = torch.randn(3, 3)
        result = self.dm.move_to_device(t)
        assert result.device.type == "cpu"

    def test_moves_model_to_device(self):
        """Model parameters are moved to the configured device."""
        model = nn.Linear(4, 2)
        result = self.dm.move_to_device(model)
        param_device = next(result.parameters()).device.type
        assert param_device == "cpu"

    def test_returns_same_object_type_for_tensor(self):
        """Return type is still torch.Tensor after move."""
        t = torch.randn(2, 2)
        result = self.dm.move_to_device(t)
        assert isinstance(result, torch.Tensor)

    def test_returns_same_object_type_for_module(self):
        """Return type is still nn.Module after move."""
        model = nn.Linear(2, 2)
        result = self.dm.move_to_device(model)
        assert isinstance(result, nn.Module)


# ---------------------------------------------------------------------------
# RunManager — init
# ---------------------------------------------------------------------------


class TestRunManagerInit:
    def test_default_dirs(self):
        """Default base_dir is 'runs' and ckpt_dir is 'checkpoints'."""
        rm = RunManager()
        assert rm.base_dir == Path("runs")
        assert rm.ckpt_dir == Path("checkpoints")

    def test_custom_dirs(self):
        """Custom directory paths are stored correctly."""
        rm = RunManager(base_dir="output", ckpt_dir="ckpts")
        assert rm.base_dir == Path("output")
        assert rm.ckpt_dir == Path("ckpts")


# ---------------------------------------------------------------------------
# RunManager — init_run
# ---------------------------------------------------------------------------


class TestRunManagerInitRun:
    def test_new_run_creates_directory(self, tmp_path):
        """Starting a new run creates the checkpoint directory on disk."""
        rm = RunManager(base_dir=str(tmp_path))
        info = rm.init_run(run_id=None)
        assert info.ckpts_path.exists()

    def test_new_run_returns_run_info(self, tmp_path):
        """New run returns a RunInfo with last_logged_epoch=0 and 6-char run_id."""
        rm = RunManager(base_dir=str(tmp_path))
        info = rm.init_run(run_id=None)
        assert isinstance(info, RunInfo)
        assert info.last_logged_epoch == 0
        assert len(info.run_id) == 6

    def test_new_run_id_is_hex(self, tmp_path):
        """Generated run_id consists only of hexadecimal characters."""
        rm = RunManager(base_dir=str(tmp_path))
        info = rm.init_run(run_id=None)
        assert all(c in "0123456789abcdef" for c in info.run_id)

    def test_resume_run_returns_correct_epoch_count(self, tmp_path):
        """Resuming a run reports the highest saved epoch number."""
        rm = RunManager(base_dir=str(tmp_path), ckpt_dir="checkpoints")
        run_id = "abc123"
        ckpts_path = tmp_path / f"run_id={run_id}" / "checkpoints"
        ckpts_path.mkdir(parents=True)
        # Simulate 3 saved checkpoints
        for i in range(1, 4):
            (ckpts_path / f"ckpt_epoch_{i}.pt").touch()

        info = rm.init_run(run_id=run_id)
        assert info.run_id == run_id
        assert info.last_logged_epoch == 3

    def test_resume_run_empty_ckpts_raises_runtime_error(self, tmp_path):
        """Resuming a run with an empty checkpoint directory raises RuntimeError."""
        rm = RunManager(base_dir=str(tmp_path), ckpt_dir="checkpoints")
        run_id = "abc123"
        ckpts_path = tmp_path / f"run_id={run_id}" / "checkpoints"
        ckpts_path.mkdir(parents=True)

        with pytest.raises(RuntimeError, match="empty"):
            rm.init_run(run_id=run_id)

    def test_resume_nonexistent_run_raises_value_error(self, tmp_path):
        """Resuming a non-existent run_id raises ValueError."""
        rm = RunManager(base_dir=str(tmp_path))
        with pytest.raises(ValueError, match="does not exist"):
            rm.init_run(run_id="nonexistent")


# ---------------------------------------------------------------------------
# RunManager — save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------


class TestRunManagerCheckpoints:
    def setup_method(self):
        self.model = nn.Linear(4, 2)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def test_save_creates_file(self, tmp_path):
        """Saving a checkpoint creates a file at the specified path."""
        rm = RunManager()
        ckpt_path = tmp_path / "epoch_1.pt"
        rm.save_checkpoint(ckpt_path, self.model, self.optimizer, epoch=1)
        assert ckpt_path.exists()

    def test_save_and_load_roundtrip(self, tmp_path):
        """Loaded checkpoint contains epoch, model_state_dict, and optimizer_state_dict."""
        rm = RunManager()
        ckpt_path = tmp_path / "epoch_1.pt"
        rm.save_checkpoint(ckpt_path, self.model, self.optimizer, epoch=1)
        checkpoint = rm.load_checkpoint(ckpt_path)
        assert checkpoint["epoch"] == 1
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_loaded_model_state_dict_matches(self, tmp_path):
        """Model weights are identical after a save/load roundtrip."""
        rm = RunManager()
        ckpt_path = tmp_path / "epoch_2.pt"
        rm.save_checkpoint(ckpt_path, self.model, self.optimizer, epoch=2)
        checkpoint = rm.load_checkpoint(ckpt_path)

        new_model = nn.Linear(4, 2)
        new_model.load_state_dict(checkpoint["model_state_dict"])

        for p_orig, p_loaded in zip(
            self.model.parameters(), new_model.parameters()
        ):
            assert torch.equal(p_orig, p_loaded)
