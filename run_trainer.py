import argparse
from functools import partial

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
)
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from epoch_engine.core import Trainer
from epoch_engine.core.callbacks import (
    BestCheckpoint,
    CheckpointPruner,
    EarlyStopping,
    NanCallback,
)
from epoch_engine.core.configs import MetricConfig
from epoch_engine.models import EDNet, ResNet


def prepare_classification_data(
    download_dir="./data", batch_size=32, num_workers=0
):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(
        root=download_dir, train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root=download_dir, train=False, download=True, transform=transform
    )

    def loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    return (
        loader(train_ds, shuffle=True),
        loader(
            Subset(test_ds, torch.arange(1000, len(test_ds))), shuffle=False
        ),
        loader(Subset(test_ds, torch.arange(1000)), shuffle=False),
    )


def prepare_regression_data(
    n_train=2000, n_valid=400, n_test=200, n_features=16, batch_size=32
):
    """Synthetic regression dataset: y = X @ w + noise."""
    torch.manual_seed(0)
    w = torch.randn(n_features)
    total = n_train + n_valid + n_test
    X = torch.randn(total, n_features)
    y = X @ w + 0.5 * torch.randn(total)

    def make_loader(xs, ys, shuffle):
        return DataLoader(
            TensorDataset(xs, ys), batch_size=batch_size, shuffle=shuffle
        )

    return (
        make_loader(X[:n_train], y[:n_train], shuffle=True),
        make_loader(
            X[n_train : n_train + n_valid],
            y[n_train : n_train + n_valid],
            shuffle=False,
        ),
        make_loader(
            X[n_train + n_valid :], y[n_train + n_valid :], shuffle=False
        ),
    )


CLASSIFICATION_MODELS = {
    "resnet": lambda: ResNet(
        in_channels=1, num_blocks=[1, 3, 1], num_classes=10
    ),
    "ednet": lambda: EDNet(
        in_channels=1,
        encoder_channels=(32, 64),
        decoder_features=(576, 250),
        num_labels=10,
    ),
}


def build_regression_model(n_features=16):
    return nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def run_classification(args):
    if args.model not in CLASSIFICATION_MODELS:
        raise ValueError(
            f"Unsupported model. Choose from: {list(CLASSIFICATION_MODELS)}"
        )
    train_loader, valid_loader, test_loader = prepare_classification_data()
    net = CLASSIFICATION_MODELS[args.model]()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.25, momentum=0.75)

    raw_metrics = [
        ("accuracy", accuracy_score),
        ("precision", partial(precision_score, average="macro")),
        ("f1", partial(f1_score, average="macro")),
    ]
    metrics = (
        {name: fn for name, fn in raw_metrics}
        if args.plot_metrics
        else [
            MetricConfig(name=name, fn=fn, plot=False)
            for name, fn in raw_metrics
        ]
    )

    trainer = Trainer(
        model=net,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        metrics=metrics,
        callbacks=[
            EarlyStopping(
                monitor="loss/valid",
                patience=2,
                mode="min",
                min_delta=0.0,
                verbose=True,
            ),
            NanCallback(monitor="loss/train", verbose=True),
            BestCheckpoint(monitor="loss/valid", mode="min", verbose=True),
            CheckpointPruner(keep_last_n=1),
        ],
        enable_amp=args.enable_amp,
        task="classification",
    )

    trainer.run(epochs=args.epochs)
    print(f"Test metrics: {trainer.evaluate()}")

    if not trainer.interrupted:
        print(f"Resuming training for 'run_id={trainer.run_id}'...")
        trainer.run(epochs=1, resume_from_best=True)
        print(f"Test metrics after resume: {trainer.evaluate()}")


def run_regression(args):
    n_features = 16
    train_loader, valid_loader, test_loader = prepare_regression_data(
        n_features=n_features
    )
    net = build_regression_model(n_features=n_features)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    raw_metrics = [
        ("mse", mean_squared_error),
        ("mae", mean_absolute_error),
    ]
    metrics = (
        {name: fn for name, fn in raw_metrics}
        if args.plot_metrics
        else [
            MetricConfig(name=name, fn=fn, plot=False)
            for name, fn in raw_metrics
        ]
    )

    trainer = Trainer(
        model=net,
        criterion=nn.MSELoss(),
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        metrics=metrics,
        callbacks=[
            EarlyStopping(
                monitor="loss/valid",
                patience=3,
                mode="min",
                verbose=True,
            ),
            NanCallback(verbose=True),
            BestCheckpoint(monitor="loss/valid", mode="min", verbose=True),
            CheckpointPruner(keep_last_n=1),
        ],
        task="regression",
    )

    trainer.run(epochs=args.epochs)
    print(f"Test metrics: {trainer.evaluate()}")

    if not trainer.interrupted:
        print(f"Resuming training for 'run_id={trainer.run_id}'...")
        trainer.run(epochs=1, resume_from_best=True)
        print(f"Test metrics after resume: {trainer.evaluate()}")


def main():
    parser = argparse.ArgumentParser(description="Run Trainer Example")
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "regression"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (classification only): resnet, ednet",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--enable-amp", action="store_true")
    parser.add_argument("--plot-metrics", action="store_true")
    args = parser.parse_args()

    if args.task == "classification":
        run_classification(args)
    else:
        run_regression(args)


if __name__ == "__main__":
    main()
