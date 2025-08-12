# Epoch Engine - Python Library for training PyTorch models

[![PyPI](https://img.shields.io/pypi/v/epoch-engine)](https://pypi.org/project/epoch-engine/)
[![Publish](https://github.com/spolivin/epoch-engine/actions/workflows/publish.yml/badge.svg)](https://github.com/spolivin/epoch-engine/actions/workflows/publish.yml)
[![License](https://img.shields.io/github/license/spolivin/epoch-engine)](https://github.com/spolivin/epoch-engine/blob/master/LICENSE.txt)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

This project represents my attempt to come up with a convenient way to train neural nets coded in Torch. While being aware of already existing libraries for training PyTorch models (e.g. PyTorch Lightning), my idea here is to make training of the models more visual and understandable as to what is going on during training.

The project is currently in its raw form, more changes expected.

## Features

* Local automatic checkpoint and metrics tracking system (***New in 0.1.2***)
* Evaluation of test sets via a separate method (***New in 0.1.2***)
* Progress bar support for both training and validation loops where the last trained epoch is remembered for tracking how many epochs a model has been trained for in total
* Intermediate metrics computations after each forward pass: loss is computed by default + added support for flexibly defining metrics from `scikit-learn`
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

### Instantiating Trainer

Let's suppose that we have constructed a model in PyTorch called `net` and set up the loss function we would like to optimize:

```python
import torch.nn as nn

from epoch_engine.models import BasicBlock, ResNet

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
from epoch_engine.core import Trainer

# Instantiating a Trainer (with auto device detection)
trainer = Trainer(
    model=net,
    criterion=loss_func,
    train_loader=train_loader,
    valid_loader=valid_loader,
    train_on="auto",
)
```
> Parameter `train_on` will detect whether CUDA, MPS or CPU is to be used (when initialized with `"auto"`).

### Optimizer and scheduler

The next step is to configure optimizer and (optionally) scheduler which can be done in one of two ways:

#### Configs

```python
from epoch_engine.core import OptimizerConfig, SchedulerConfig

# Setting up configs for optimizer and scheduler
optimizer_config = OptimizerConfig(
    optimizer_class=torch.optim.SGD,
    optimizer_params={"lr": 0.25, "momentum": 0.75},
)
scheduler_config = SchedulerConfig(
    scheduler_class=torch.optim.lr_scheduler.StepLR,
    scheduler_params={"gamma": 0.1, "step_size": 2},
    scheduler_level="epoch",
)

trainer.configure_trainer(
    optimizer_config=optimizer_config,
    scheduler_config=scheduler_config,
)
```

#### Direct

```python
trainer.configure_trainer(
    optimizer_class=torch.optim.SGD,
    optimizer_params={"lr": 0.25, "momentum": 0.75},
    scheduler_class=torch.optim.lr_scheduler.StepLR,
    scheduler_params={"gamma": 0.1, "step_size": 2},
    scheduler_level="epoch",
)
```

### Metrics registration

By default only loss is computed but we can also add extra metrics to track during Trainer run:

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score

trainer.register_metrics({
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="macro"),
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
})
```
> Important thing here is for the passed dict to map metric name we want to see in the logs to the callable objects which in turn map targets and predictions to floats.

If we want to track just one extra metric we can just define it like this:

```python
trainer.register_metric("accuracy", accuracy_score)
```

### Running the Trainer

#### New run

Now, we launch the training for the first time which will show the progress bar (if enabled):

```python
# Launching training
trainer.run(
    epochs=5,
    run_id=None,
    seed=42,
    enable_tqdm=True,
)
```

Running the Trainer in this way will set up in the current directory the following directory structure:

```
current-dir/
├── runs/
    ├── run_id=82cc72/
    │    ├── checkpoints/
    │        ├── ckpt_epoch_1.pt
    │        ├── ckpt_epoch_2.pt
    │        ├── ckpt_epoch_3.pt
    │        ├── ckpt_epoch_4.pt
    │        ├── ckpt_epoch_5.pt
    └── training_process.json
```

At the beginning of the run, a new `run_id` is generated (if `run_id=None`) and in the current folder the method creates `runs` folder with a separate folder for the files related to the current run (in the above example folder named `run_id=82cc72` with the generated run ID). After each epoch the checkpoint (containing last trained epoch, Trainer's new run ID, model parameters and optimizer state) is saved to `checkpoints` subfolder.

Additionally, the losses for each data set as well as the registered metrics for each epoch and training run are saved to `runs/training_process.json` and are written to each run. Such results can look for instance like this:

```json
{
     "runs": [
          {
               "run_id=82cc72": [
                    {
                         "loss/train": 0.7220337147848059,
                         "loss/valid": 0.056390782489593255,
                         "accuracy/train": 0.7354,
                         "accuracy/valid": 0.9828888888888889,
                         "precision/train": 0.745122229756801,
                         "precision/valid": 0.9828108100143986,
                         "f1/train": 0.7367284983928013,
                         "f1/valid": 0.982719354776437,
                         "epoch": 1
                    },
                    ...
               ]
          }
     ]
}
```

#### Resuming training

The training can be easily resumed by specifying the `run_id` from which we would like to continue training (resuming training only from the last logged epoch is supported):

```python
trainer.run(
    epochs=2,
    run_id="82cc72",
    seed=42,
    enable_tqdm=True,
)
```
> The new checkpoints will be saved to the same folder for this run and new metrics will be appended to the same run ID's data in `training_process.json`.

## Test script

The basics of the developed API are presented in the `run_trainer.py` I built in the root of the repository. It can be run for instance as follows:

```bash
# Installing scikit-learn for using metrics
pip install -r requirements.txt

# Running the Trainer
python run_trainer.py --model=resnet --epochs=3
```
> The training will be launched on the device automatically derived based on the CUDA availability and the final training checkpoint will be saved in `checkpoints` directory.
