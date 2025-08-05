# Epoch Engine - Python Library for training PyTorch models

[![PyPI](https://img.shields.io/pypi/v/epoch-engine)](https://pypi.org/project/epoch-engine/)
[![Publish](https://github.com/spolivin/epoch-engine/actions/workflows/publish.yml/badge.svg)](https://github.com/spolivin/epoch-engine/actions/workflows/publish.yml)
[![License](https://img.shields.io/github/license/spolivin/epoch-engine)](https://github.com/spolivin/epoch-engine/blob/master/LICENSE.txt)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

This project represents my attempt to come up with a convenient way to train neural nets coded in Torch. While being aware of already existing libraries for training PyTorch models (e.g. PyTorch Lightning), my idea here is to make training of the models more visual and understandable as to what is going on during training.

The project is currently in its raw form, more changes expected.

## Features

* TQDM-Progress bar support for both training and validation loops
* Intemediate metrics computations after each forward pass (loss is computed by default + added support for flexibly defining metrics from `scikit-learn`)
* Saving/loading checkpoints from/into Trainer directly without having to touch model, optimizer or scheduler separately
* Resuming training from the loaded checkpoint with epoch number being remembered automatically to avoid having to remember from which epoch the training originally started
* Ready-to-use neural net architectures coded from scratch (currently only 4-layer Encoder-Decoder and ResNet with 20 layers architectures are available)

## Installation

The package can be installed as follows:

```bash
# Installing the main package
pip install epoch-engine

# Installing additional optional dependencies
pip install epoch-engine[build,linters]
```

### Development mode

```bash
# Cloning the repo and moving to the repo dir
git clone https://github.com/spolivin/epoch-engine.git
cd epoch-engine

# Installing the package and optional deps (dev mode)
pip install -e .
pip install -e .[build,linters]
```

### Pre-commit support

The repository provides support for running pre-commit checks via hooks defined in `.pre-commit-config.yaml`. These can be loaded in the current git repository by running:

```bash
pre-commit install
```
> `pre-commit` will already be loaded to the venv after running `pip install epoch-engine[linters]` or `pip install -e .[linters]`

## Python API

Let's suppose that we have constructed a model in PyTorch called `model` and set up the loss function we would like to optimize:

```python
import torch.nn as nn

from epoch_engine.models.architectures import BasicBlock, ResNet

# Instantiating a ResNet model for gray-scale images
net = ResNet(
    in_channels=1,
    block=BasicBlock,
    num_blocks=[3, 3, 3],
    num_classes=10,
)
loss_func = nn.CrossEntropyLoss()
```

We have also the already prepared dataloaders: `train_loader` and `valid_loader` for training and validation sets respectively. Then, we can set up the Trainer as follows:

```python
import torch
from sklearn.metrics import accuracy_score

# Instantiating a Trainer (with auto device detection)
trainer = Trainer(
    model=net,
    criterion=loss_func,
    train_loader=train_loader,
    valid_loader=valid_loader,
    train_on="auto",
)

# Setting up an additional metric to track (loss is computed and displayed by default)
trainer.register_metric("accuracy", accuracy_score)

# Setting up optimizer and scheduler
trainer.configure_optimizers(
    optimizer_class=torch.optim.SGD,
    optimizer_params={
        "lr": 0.25,
        "momentum": 0.75,
    },
    scheduler_class=torch.optim.lr_scheduler.StepLR,
    scheduler_params={"step_size": 2, "gamma": 0.1},
)
```
> Parameter `train_on` will detect whether CUDA, MPS or CPU is to be used (when initialized with `"auto"`). We can register an additional metric to track which will be shown for training and validation sets after each epoch. Lastly, we need to configure optimizer and scheduler to use (scheduler-related parameters can be omitted if we do not want to use scheduler).

Now, we launch the training which will show the TQDM-progress bar (if enabled):

```python
# Launching training
trainer.run(
    epochs=5,
    seed=42,
    enable_tqdm=True,
    )
```

### Loading/saving checkpoints

After running training, we can save the state dict for the model and optimizer for later use (resuming training) as follows:

```python
trainer.save_checkpoint("checkpoints/checkpoint.pt")
```
> This method will also save the latest epoch at which the training stopped (useful for displaying the next epoch during new training run).

We can also load the state dict back into the model and optimizer if available before running a new training:

```python
trainer.load_checkpoint(path="checkpoints/checkpoint.pt")
```
> One needs to make sure in this case that the optimizer defined in the Trainer matches with that for which checkpoint was saved.

### Test script

The basics of the developed API are presented in the `run_trainer.py` I built in the root of the repository. It can be run for instance as follows:

```bash
python run_trainer.py --model=resnet --epochs=3 --batch-size=16
```
> The training will be launched on the device automatically derived based on the CUDA availability and the final training checkpoint will be saved in `checkpoints` directory.

One can also resume the training from the saved checkpoint:

```bash
python run_trainer.py --model=resnet --epochs=4 --resume-training=True --ckpt-path=checkpoints/ckpt_3.pt
```
> The training will be resumed from the loaded checkpoint with TQDM-progress bar showing the next training epoch.
