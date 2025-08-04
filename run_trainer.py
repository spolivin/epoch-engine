import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from epoch_engine.core.trainer import Trainer
from epoch_engine.models.architectures import BasicBlock, EDNet, ResNet


def prepare_datasets(download_dir="./data", transform=None):
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root=download_dir, train=True, download=True, transform=transform
    )
    valid_dataset = datasets.MNIST(
        root=download_dir, train=False, download=True, transform=transform
    )

    return train_dataset, valid_dataset


def prepare_dataloaders(train_dataset, valid_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def configure_loss_function():
    return nn.CrossEntropyLoss()


def configure_optimizer(model, learning_rate=0.25, momentum=0.75):
    optimizer_params = {
        "lr": learning_rate,
        "momentum": momentum,
    }
    return torch.optim.SGD(model.parameters(), **optimizer_params)


parser = argparse.ArgumentParser(description="Run Trainer Example")
parser.add_argument("--model", type=str, default=None, help="Model name to use")
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
parser.add_argument(
    "--batch-size", type=int, default=32, help="Batch size for training and validation"
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
        train_dataset, valid_dataset, batch_size=args.batch_size
    )

    if args.model == "resnet":
        net = ResNet(
            in_channels=1, block=BasicBlock, num_blocks=[1, 2, 1], num_classes=10
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

    loss_func = configure_loss_function()
    optimizer = configure_optimizer(model=net)

    trainer = Trainer(
        model=net,
        criterion=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        train_on="auto",
    )

    if args.resume_training and args.ckpt_path:
        trainer.load_checkpoint(args.ckpt_path)
    elif args.resume_training:
        raise ValueError("Checkpoint path must be provided to resume training.")

    trainer.run(
        epochs=args.epochs,
        seed=42,
        enable_tqdm=True,
    )

    trainer.save_checkpoint(f"checkpoints/ckpt_{trainer.last_epoch}.pt")
