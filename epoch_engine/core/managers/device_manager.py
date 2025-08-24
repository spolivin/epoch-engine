import torch
import torch.nn as nn


class DeviceManager:
    """Manager of device placement and AMP.

    Attributes:
        device (torch.device): The device detected.
        enable_amp (bool): Flag enabling mixed precision training.
    """

    def __init__(self, enable_amp: bool = False) -> None:
        """Initializes a class instance.

        Args:
            enable_amp (bool, optional): Flag enabling mixed precision training. Defaults to False.

        Raises:
            RuntimeError: Error raised in case of trying to enable AMP for non-CUDA device.
        """
        self.device = self._auto_detect_device()
        self.enable_amp = enable_amp and self.device.type == "cuda"
        if enable_amp and not self.device.type == "cuda":
            raise RuntimeError(
                "CUDA device is not available but 'enable_amp' is set to True. "
                "Mixed precision can only be used on CUDA."
            )

    def _auto_detect_device(self) -> torch.device:
        """Automatically detects the device type to use for training.

        Returns:
            str: The detected device type (CPU, CUDA, or MPS).
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

    def set_seed(self, seed: int = 42):
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)

    def move_to_device(
        self, obj: nn.Module | torch.Tensor
    ) -> nn.Module | torch.Tensor:
        """Moves an object (nn.Module or torch.Tensor) to detected device."""

        return obj.to(self.device)
