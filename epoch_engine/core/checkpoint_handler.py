"""Module for handling saving/loading of checkpoints."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import torch


class CheckpointHandler:
    """Class for saving and loading trainer checkpoints."""

    def save_checkpoint(self, trainer, path: str) -> None:
        """Saves the checkpoint: last trained epoch, model state and optimizer state.

        Args:
            trainer: Trainer instance.
            path (str): Path of the checkpoint file to which data are to be saved.
        """
        torch.save(
            {
                "epoch": trainer.last_epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, trainer, path: str) -> None:
        """Loads the checkpoint: last trained epoch, model state and optimizer state.

        Args:
            trainer: Trainer instance.
            path (str): Path of the checkpoint file from which data are to be loaded.
        """
        # Loading the checkpoint file
        checkpoint = torch.load(path, weights_only=True)
        # Loading the data into the Trainer instance attributes
        trainer.last_epoch = checkpoint["epoch"]
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
