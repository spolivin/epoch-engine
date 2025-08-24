import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer


@dataclass
class RunInfo:
    """Trainer run information output configuration.

    Attributes:
        run_id (str): Trainer run's ID.
        ckpts_path (Path): Path to run-specific checkpoints subfolder.
        last_logged_epoch (int): Last epoch that Trainer has registered.
    """

    run_id: str
    ckpts_path: Path
    last_logged_epoch: int


class RunManager:
    """Manager of run-specific events (checkpoints handling and run initialization).

    Attributes:
        base_dir (str): Base directory where run information is stored.
        ckpt_dir (str): Directory storing Trainer checkpoints.
    """

    def __init__(
        self, base_dir: str = "runs", ckpt_dir: str = "checkpoints"
    ) -> None:
        """Initializes a class instance.

        Args:
            base_dir (str, optional): Base directory where run information is stored. Defaults to "runs".
            ckpt_dir (str, optional): Directory storing Trainer checkpoints. Defaults to "checkpoints".
        """
        self.base_dir = Path(base_dir)
        self.ckpt_dir = Path(ckpt_dir)

    def init_run(self, run_id: str | None) -> RunInfo:
        """Initializes a Trainer run.

        Args:
            run_id (str | None): Trainer run's ID.

        Raises:
            RuntimeError: Error raised when `run_id` is present but no checkpoint to start with.
            ValueError: Error raised in case passed `run_id` cannot be found in history.

        Returns:
            RunInfo: Trainer run results as a config.
        """
        # Checking the case when `run_id` is not None
        if run_id:
            ckpts_path = self.base_dir / f"run_id={run_id}" / self.ckpt_dir
            if os.path.exists(ckpts_path):
                last_logged_epoch = len(os.listdir(ckpts_path))
                if last_logged_epoch == 0:
                    raise RuntimeError(
                        f"Checkpoint directory for 'run_id={run_id}' appears to be empty. "
                        "Resuming from this checkpoint is not possible."
                    )
                else:
                    return RunInfo(
                        run_id=run_id,
                        ckpts_path=ckpts_path,
                        last_logged_epoch=last_logged_epoch,
                    )
            else:
                raise ValueError(
                    f"Cannot resume training from 'run_id={run_id}', since it does not exist in history. "
                    "Check the correctness of 'run_id'."
                )
        # Checking the case when `run_id` is None
        else:
            # Generating `run_id` for a new run and creating folder for this run
            run_id = uuid.uuid4().hex[:6]
            ckpts_path = self.base_dir / f"run_id={run_id}" / self.ckpt_dir
            os.makedirs(ckpts_path)

            return RunInfo(
                run_id=run_id,
                ckpts_path=ckpts_path,
                last_logged_epoch=0,
            )

    def save_checkpoint(
        self, path: Path, model: nn.Module, optimizer: Optimizer, epoch: int
    ) -> None:
        """Saves checkpoint.

        Args:
            path (Path): Path to checkpoint.
            model (nn.Module): Torch model instance.
            optimizer (Optimizer): Torch optimizer instance.
            epoch (int): Current epoch.
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> Any:
        """Loads checkpoint.

        Args:
            path (Path): Path to checkpoint.

        Returns:
            Any: Loaded checkpoint.
        """
        checkpoint = torch.load(path, weights_only=True)

        return checkpoint
