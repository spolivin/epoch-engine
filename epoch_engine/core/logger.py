"""Tqdm-aware logger used by the Trainer for console output."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import sys


class TrainerLogger:
    """Console logger that avoids corrupting tqdm progress bars.

    When ``enable_tqdm=True``, all output is routed through
    ``tqdm.write`` so messages appear above the progress bar without
    breaking it. When disabled, output falls back to ``print`` (stdout)
    or ``print(..., file=sys.stderr)`` depending on severity.
    """

    def __init__(self, enable_tqdm: bool = True) -> None:
        """
        Args:
            enable_tqdm (bool): If ``True``, use ``tqdm.write`` for all
                output. Set to ``False`` when tqdm is not in use.
                Defaults to ``True``.
        """
        self.enable_tqdm = enable_tqdm

    def info(self, message: str) -> None:
        """Write ``message`` to stdout (no prefix)."""
        if self.enable_tqdm:
            from tqdm import tqdm

            tqdm.write(message)
        else:
            print(message)

    def warning(self, message: str) -> None:
        """Write ``message`` to stderr prefixed with ⚠️."""
        message = f"⚠️  {message}"
        if self.enable_tqdm:
            from tqdm import tqdm

            tqdm.write(message)
        else:
            print(message, file=sys.stderr)

    def error(self, message: str) -> None:
        """Write ``message`` to stderr prefixed with ❌."""
        message = f"❌ {message}"
        if self.enable_tqdm:
            from tqdm import tqdm

            tqdm.write(message)
        else:
            print(message, file=sys.stderr)

    def success(self, message: str) -> None:
        """Write ``message`` to stdout prefixed with ✅."""
        message = f"✅ {message}"
        if self.enable_tqdm:
            from tqdm import tqdm

            tqdm.write(message)
        else:
            print(message)
