"""Device detection, model placement, seed setting, and AMP configuration."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

from typing import TypeVar

import torch
import torch.nn as nn

_T = TypeVar("_T", nn.Module, torch.Tensor)


class DeviceManager:
    """Handles device selection, model/tensor placement, seeds, and AMP.

    Device priority: CUDA -> MPS -> CPU. AMP (automatic mixed precision) is
    only supported on CUDA; passing ``enable_amp=True`` on any other device
    raises a ``RuntimeError``.

    Attributes:
        device (torch.device): Auto-detected device used for all placements.
        enable_amp (bool): Whether AMP is active (always ``False`` on non-CUDA).
    """

    def __init__(self, enable_amp: bool = False) -> None:
        """
        Args:
            enable_amp (bool, optional): Enable AMP for mixed-precision
                training. Only valid on CUDA. Defaults to ``False``.

        Raises:
            RuntimeError: If ``enable_amp=True`` but the detected device is
                not CUDA.
        """
        self.device = self._auto_detect_device()
        self.enable_amp = enable_amp and self.device.type == "cuda"
        if enable_amp and not self.device.type == "cuda":
            raise RuntimeError(
                "CUDA device is not available but 'enable_amp' is set to True. "
                "Mixed precision can only be used on CUDA."
            )

    def _auto_detect_device(self) -> torch.device:
        """Returns the best available device in CUDA -> MPS -> CPU priority.

        Returns:
            torch.device: The selected device.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def set_seed(self, seed: int = 42) -> None:
        """Sets the random seed for reproducibility.

        Uses ``torch.cuda.manual_seed`` on CUDA, ``torch.manual_seed``
        otherwise.

        Args:
            seed (int, optional): Random seed value. Defaults to ``42``.
        """
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)

    def move_to_device(self, obj: _T) -> _T:
        """Moves ``obj`` to ``self.device`` and returns it.

        Args:
            obj (nn.Module | torch.Tensor): Model or tensor to move.

        Returns:
            nn.Module | torch.Tensor: The same object, now on ``self.device``,
                preserving the input type.
        """
        return obj.to(self.device)
