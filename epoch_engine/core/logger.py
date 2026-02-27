"""Module for unified logging in Trainer that works with tqdm progress bars."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import sys


class TrainerLogger:
    """Unified logging for Trainer that works with tqdm progress bars."""

    def __init__(self, enable_tqdm: bool = True):
        self.enable_tqdm = enable_tqdm

    def info(self, message: str) -> None:
        """Print info message (compatible with tqdm)."""
        if self.enable_tqdm:
            from tqdm import tqdm

            tqdm.write(message)
        else:
            print(message)

    def warning(self, message: str) -> None:
        """Print warning message (compatible with tqdm)."""
        message = f"⚠️  {message}"
        if self.enable_tqdm:
            from tqdm import tqdm

            tqdm.write(message)
        else:
            print(message, file=sys.stderr)

    def error(self, message: str) -> None:
        """Print error message (compatible with tqdm)."""
        message = f"❌ {message}"
        if self.enable_tqdm:
            from tqdm import tqdm

            tqdm.write(message)
        else:
            print(message, file=sys.stderr)

    def success(self, message: str) -> None:
        """Print success message (compatible with tqdm)."""
        message = f"✅ {message}"
        if self.enable_tqdm:
            from tqdm import tqdm

            tqdm.write(message)
        else:
            print(message)
