import argparse

import torch
import torch.nn as nn
from sklearn.metrics import precision_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from epoch_engine.core.trainer import Trainer
from epoch_engine.models.architectures import BasicBlock, EDNet, ResNet


def prepare_datasets(
    download_dir: str = "./data", transform: transforms.Compose = None
) -> tuple:
    """Prepare the MNIST datasets for training and validation.

    Args:
        download_dir (str, optional): Directory to download the dataset. Defaults to "./data".
        transform (transforms.Compose, optional): Transformations to apply to the dataset. Defaults to None.

    Returns:
        tuple: A tuple containing the training and validation datasets.
    """

    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root=download_dir, train=True, download=True, transform=transform
    )
    valid_dataset = datasets.MNIST(
        root=download_dir, train=False, download=True, transform=transform
    )

    return train_dataset, valid_dataset


def prepare_dataloaders(
    train_dataset: datasets.MNIST,
    valid_dataset: datasets.MNIST,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple:
    """Prepare data loaders for training and validation.

    Args:
        train_dataset (datasets.MNIST): The training dataset.
        valid_dataset (datasets.MNIST): The validation dataset.
        batch_size (int, optional): Batch size for the data loaders. Defaults to 32.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 0.

    Returns:
        tuple: A tuple containing the training and validation data loaders.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, valid_loader


parser = argparse.ArgumentParser(description="Run Trainer Example")
parser.add_argument(
    "--model", type=str, default=None, help="Model name to use"
)
parser.add_argument(
    "--epochs", type=int, default=3, help="Number of epochs to train"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for training and validation",
)
parser.add_argument(
    "--resume-training",
    type=bool,
    default=False,
    help="Resume training from checkpoint",
)
parser.add_argument(
    "--ckpt-path", type=str, default=None, help="Path to the checkpoint file"
)
args = parser.parse_args()

if __name__ == "__main__":
    train_dataset, valid_dataset = prepare_datasets()
    train_loader, valid_loader = prepare_dataloaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=args.batch_size,
        num_workers=0,
    )

    if args.model == "resnet":
        net = ResNet(
            in_channels=1,
            block=BasicBlock,
            num_blocks=[1, 1, 1],
            num_classes=10,
        )
    elif args.model == "ednet":
        net = EDNet(
            in_channels=1,
            encoder_channels=(32, 64),
            decoder_features=(576, 250),
            num_labels=10,
        )
    else:
        raise ValueError("Unsupported model type. Use 'resnet' or 'ednet'.")

    loss_func = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=net,
        criterion=loss_func,
        train_loader=train_loader,
        valid_loader=valid_loader,
        train_on="auto",
    )
    # Registering an additional metric for evaluation
    trainer.register_metric(
        "precision",
        lambda y_true, y_pred: precision_score(
            y_true, y_pred, average="macro"
        ),
    )

    print(f"Using device: {trainer.device.type}")

    # Setting up the optimizer and scheduler
    trainer.configure_optimizers(
        optimizer_class=torch.optim.SGD,
        optimizer_params={
            "lr": 0.25,
            "momentum": 0.75,
        },
        scheduler_class=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 2, "gamma": 0.1},
    )

    # If resuming training, load the checkpoint if provided
    if args.resume_training and args.ckpt_path:
        trainer.load_checkpoint(args.ckpt_path)
    elif args.resume_training:
        raise ValueError(
            "Checkpoint path must be provided to resume training."
        )

    # Run the training process
    trainer.run(
        epochs=args.epochs,
        seed=42,
        enable_tqdm=True,
    )

    # Save the final model checkpoint
    trainer.save_checkpoint(f"checkpoints/ckpt_{trainer.last_epoch}.pt")
