"""Module for logging Trainer events."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import logging
from pathlib import Path


class TrainerLogger:
    """Logger of Trainer events.

    Attributes:
        base_dir (str): Base directory where logfile is to be created.
        logfile_path (str): Full path to a logfile.
        logger_name (str): Name of a logger.
    """

    def __init__(
        self,
        base_dir: str = "runs",
        log_file: str = "trainer_events.log",
        logger_name: str = "trainer.global",
    ) -> None:
        """Initializes a class instance.

        Args:
            base_dir (str, optional): Base directory where logfile is to be created. Defaults to "runs".
            log_file (str, optional): Name of a logfile. Defaults to "trainer_events.log".
            logger_name (str, optional): Name of a logger. Defaults to "trainer.global".
        """
        self.base_dir = Path(base_dir)
        self.logfile_path = self.base_dir / log_file
        self.logger_name = logger_name

    def get_logger(self) -> logging.Logger:
        """Retrieves the configured logger.

        Returns:
            logging.Logger: Configured logger.
        """
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            file_handler = logging.FileHandler(self.logfile_path)
            file_handler.setLevel(logging.INFO)

            formatter = logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s: %(message)s",
                datefmt="%d-%m-%Y %H:%M:%S",
            )
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)

        logger.propagate = False

        return logger
