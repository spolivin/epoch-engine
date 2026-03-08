"""Run lifecycle management: directory creation, checkpoint I/O, and resume logic."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

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
    """Metadata about an initialized training run.

    Attributes:
        run_id (str): 6-character hex identifier for the run
            (e.g. ``'a3f91c'``).
        ckpts_path (Path): Absolute path to the run's checkpoint directory
            (``runs/run_id=<hex>/checkpoints/``).
        last_logged_epoch (int): Highest epoch number found in saved
            checkpoints; ``0`` for a freshly created run.
    """

    run_id: str
    ckpts_path: Path
    last_logged_epoch: int


class RunManager:
    """Manages the on-disk run structure, checkpoint saving, and resume logic.

    Each run lives under ``<base_dir>/run_id=<hex>/<ckpt_dir>/`` and contains
    ``ckpt_epoch_N.pt`` files plus an optional ``best.pt`` written by
    :class:`callbacks.BestCheckpoint`.

    Attributes:
        base_dir (Path): Root directory that holds all run subdirectories.
        ckpt_dir (Path): Name of the checkpoint subdirectory within each run.
    """

    def __init__(
        self, base_dir: str = "runs", ckpt_dir: str = "checkpoints"
    ) -> None:
        """
        Args:
            base_dir (str, optional): Root directory for all runs.
                Defaults to ``"runs"``.
            ckpt_dir (str, optional): Checkpoint subdirectory name inside
                each run folder. Defaults to ``"checkpoints"``.
        """
        self.base_dir = Path(base_dir)
        self.ckpt_dir = Path(ckpt_dir)

    def init_run(self, run_id: str | None) -> RunInfo:
        """Initializes a training run, either new or resumed.

        If ``run_id`` is ``None``, a new 6-char hex ID is generated and the
        run directory is created. If ``run_id`` is provided, the existing run
        directory is located and the highest saved epoch is resolved for
        resuming.

        Args:
            run_id (str | None): Existing run ID to resume, or ``None`` to
                start a new run.

        Raises:
            RuntimeError: If ``run_id`` exists on disk but its checkpoint
                directory contains no ``ckpt_epoch_*.pt`` files.
            ValueError: If ``run_id`` is provided but the corresponding run
                directory does not exist.

        Returns:
            RunInfo: Resolved run metadata (ID, checkpoint path, last epoch).
        """
        # Checking the case when `run_id` is not None
        if run_id:
            ckpts_path = self.base_dir / f"run_id={run_id}" / self.ckpt_dir
            if os.path.exists(ckpts_path):
                epoch_ckpts = list(ckpts_path.glob("ckpt_epoch_*.pt"))
                if not epoch_ckpts:
                    raise RuntimeError(
                        f"Checkpoint directory for 'run_id={run_id}' appears to be empty. "
                        "Resuming from this checkpoint is not possible."
                    )
                last_logged_epoch = max(
                    int(f.stem.split("_")[-1]) for f in epoch_ckpts
                )
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
        """Saves a checkpoint dict to ``path``.

        The file contains ``{'epoch': int, 'model_state_dict': ...,
        'optimizer_state_dict': ...}``.

        Args:
            path (Path): Destination file path (e.g. ``ckpt_epoch_5.pt``).
            model (nn.Module): Model whose ``state_dict`` is saved.
            optimizer (Optimizer): Optimizer whose ``state_dict`` is saved.
            epoch (int): Current epoch number stored alongside the states.
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        """Loads a checkpoint file and returns its contents.

        Args:
            path (Path): Path to the ``.pt`` checkpoint file.

        Returns:
            dict[str, Any]: Dict with keys ``'epoch'``, ``'model_state_dict'``,
                and ``'optimizer_state_dict'``.
        """
        checkpoint = torch.load(path, weights_only=True)

        return checkpoint
