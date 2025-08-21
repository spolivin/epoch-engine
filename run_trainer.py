import argparse

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from epoch_engine.core import Trainer
from epoch_engine.core.configs import (
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
)
from epoch_engine.models import EDNet, ResNet


def prepare_datasets(
    download_dir: str = "./data", transform: transforms.Compose = None
) -> tuple:
    """Prepares the MNIST datasets for training and validation.

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
    test_dataset = datasets.MNIST(
        root=download_dir, train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset


def prepare_dataloaders(
    train_dataset: datasets.MNIST,
    test_dataset: datasets.MNIST,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple:
    """Prepares data loaders for training, validation and testing.

    Args:
        train_dataset (datasets.MNIST): The training dataset.
        test_dataset (datasets.MNIST): The dataset to be divided into validation and testing sets.
        batch_size (int, optional): Batch size for the data loaders. Defaults to 32.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 0.

    Returns:
        tuple: A tuple containing the training, validation and testing data loaders.
    """
    # Taking a subset to form a validation set
    valid_dataset = Subset(test_dataset, torch.arange(1000, len(test_dataset)))
    # Taking a subset to form the final testing set (1000 objects)
    test_dataset_2 = Subset(test_dataset, torch.arange(1000))

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
    test_loader = DataLoader(
        test_dataset_2,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, test_loader


parser = argparse.ArgumentParser(description="Run Trainer Example")
parser.add_argument(
    "--model", type=str, default=None, help="Model name to use"
)
parser.add_argument(
    "--epochs", type=int, default=3, help="Number of epochs to train"
)
parser.add_argument(
    "--enable-amp",
    type=bool,
    default=False,
    help="Flag to enable mixed precision",
)
args = parser.parse_args()

if __name__ == "__main__":
    # Downloading and preparing the data
    train_dataset, test_dataset = prepare_datasets()
    # Preparing dataloaders for training, validation and testing sets
    train_loader, valid_loader, test_loader = prepare_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=32,
        num_workers=0,
    )

    # Selecting the model for running the Trainer
    if args.model == "resnet":
        net = ResNet(
            in_channels=1,
            num_blocks=[1, 3, 1],
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

    trainer_config = TrainerConfig(
        # Base settings
        model=net,
        criterion=nn.CrossEntropyLoss(),
        train_loader=train_loader,
        valid_loader=valid_loader,
        enable_amp=args.enable_amp,
        # Optimizer + Scheduler
        optimizer_config=OptimizerConfig(
            optimizer_class=torch.optim.SGD,
            optimizer_params={"lr": 0.25, "momentum": 0.75},
        ),
        scheduler_config=SchedulerConfig(
            scheduler_class=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"gamma": 0.1, "step_size": 2},
        ),
        # Metrics for evaluation
        metrics={
            "accuracy": accuracy_score,
            "precision": lambda y_true, y_pred: precision_score(
                y_true, y_pred, average="macro"
            ),
            "f1": lambda y_true, y_pred: f1_score(
                y_true, y_pred, average="macro"
            ),
        },
    )
    trainer = Trainer.from_config(config=trainer_config)

    # Run the training process (no run_id specified -> new run)
    trainer.run(
        epochs=args.epochs,
        seed=42,
        enable_tqdm=True,
    )

    # Evaluating the model on the test set
    print("Computing metrics on test set. Standby...")
    test_metrics = trainer.evaluate(loader=test_loader)
    print(f"Test metrics: {test_metrics}")

    # Resuming a training run for the already saved checkpoint (run_id specified)
    print(f"Resuming training for 'run_id={trainer.run_id}'...")
    trainer.run(
        epochs=2,
        run_id=trainer.run_id,
        seed=42,
        enable_tqdm=True,
    )

    # Evaluating the model on the test set after finishing another training cycle
    print("Computing metrics on test set after resuming training. Standby...")
    test_metrics = trainer.evaluate(loader=test_loader)
    print(f"Test metrics: {test_metrics}")
