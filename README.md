# 🏋️‍♂️ EpochEngine: A Lightweight Training Framework for PyTorch

[![PyPI](https://img.shields.io/pypi/v/epoch-engine)](https://pypi.org/project/epoch-engine/)
[![Publish](https://github.com/spolivin/epoch-engine/actions/workflows/publish.yml/badge.svg)](https://github.com/spolivin/epoch-engine/actions/workflows/publish.yml)
[![License](https://img.shields.io/github/license/spolivin/epoch-engine)](https://github.com/spolivin/epoch-engine/blob/master/LICENSE.txt)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

**EpochEngine** is a minimal yet flexible library for training and validating PyTorch models, currently focused on computer vision classification tasks. It provides a simple API to configure models, optimizers, schedulers, and metrics while handling logging, checkpointing, and plotting automatically.

**The project is currently under active development, more changes expected.**

## ✨ Key Features

* 🔧 **Config-driven setup** — pass model, loss, optimizer, scheduler, and metrics in a single config.
* 💾 **Automatic checkpointing** — saves model weights, optimizer state and other important data after each epoch, with easy resuming via run ID.
* 📊 **Metric tracking & plotting** — logs training/validation metrics to JSON and generates plots at the end of training.
* 🚀 **Resuming training** — continue any previous run by providing its run ID.
* ⚡ **Mixed precision training** - supports AMP with `torch.amp` and `GradScaler`.
* 🧩 **Extensible** — easily add custom models, losses, optimizers, schedulers, and metrics.

Designed to be lightweight, transparent, and beginner-friendly, without the overhead of larger frameworks like PyTorch Lightning.

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

Let's suppose that we have constructed a model in PyTorch called `net` (in the example below we can just take already preset blocks for building ResNet model) and set up the loss function we would like to optimize:

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

We have also the already prepared dataloaders: `train_loader` and `valid_loader` for training and validation sets respectively. Then, we can set up the Trainer in two major ways:

* **Multi-step setup**
* **Config-based setup**

### Multi-step setup

#### Instantiating Trainer
```python
from epoch_engine.core import Trainer

# Instantiating a Trainer (with auto device detection)
trainer = Trainer(
    model=net,
    criterion=loss_func,
    train_loader=train_loader,
    valid_loader=valid_loader,
    train_on="auto",
    enable_amp=True,
)
```
> Parameter `train_on` will detect whether CUDA, MPS or CPU is to be used (when initialized with `"auto"`), while `enable_amp` will set the trainer to run mixed precision training (mixed precision works only on CUDA devices so enabling this for CPU will raise `RuntimeError`).

#### Optimizer and scheduler

The next step is to configure optimizer and (optionally) scheduler which can be done in one of two ways:

##### Configs

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

##### Direct

```python
trainer.configure_trainer(
    optimizer_class=torch.optim.SGD,
    optimizer_params={"lr": 0.25, "momentum": 0.75},
    scheduler_class=torch.optim.lr_scheduler.StepLR,
    scheduler_params={"gamma": 0.1, "step_size": 2},
    scheduler_level="epoch",
)
```

#### Metrics registration

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

##### Advanced

One can also register metrics via a custom `MetricConfig` which is similar to passing a dictionary but with an additional `plot` parameter for controlling for which metrics to generate plots (in case of passing a dictionary, `plot=True` is by default):

```python
from epoch_engine.core.configs import MetricConfig

trainer.register_metrics([
    MetricConfig(name="accuracy", fn=accuracy_score, plot=False),
])
```
> In the example above we are registering an additional accuracy metric and specifically state that no plot need to be generated at the end of training run.

### Config-based setup (recommendable)

In version 0.1.3 an additional option of creating a Trainer object has been added where we just specify all required parameters in a `TrainerConfig` and then pass it to `Trainer`. This will set up all things so that training could be launched without any more settings:

```python
from epoch_engine.core.configs import TrainerConfig

trainer_config = TrainerConfig(
    model=net,
    criterion=nn.CrossEntropyLoss(),
    train_loader=train_loader,
    valid_loader=valid_loader,
    enable_amp=True,
    optimizer_config=OptimizerConfig(
        optimizer_class=torch.optim.SGD,
        optimizer_params={"lr": 0.25, "momentum": 0.75},
    ),
    scheduler_config=SchedulerConfig(
        scheduler_class=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"gamma": 0.1, "step_size": 2},
    ),
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
    │    |   ├── ckpt_epoch_1.pt
    │    |   ├── ckpt_epoch_2.pt
    │    |   ├── ckpt_epoch_3.pt
    │    |   ├── ckpt_epoch_4.pt
    │    |   └── ckpt_epoch_5.pt
    |    ├── results_accuracy.png
    |    ├── results_f1.png
    |    ├── results_loss.png
    |    └── results_precision.png
    |
    └── training_history.json
```

At the beginning of the run, a new `run_id` is generated (if `run_id=None`) and in the current folder the method creates `runs` folder with a separate folder for the files related to the current run (in the above example folder named `run_id=82cc72` with the generated run ID). After each epoch the checkpoint (containing last trained epoch, Trainer's new run ID, model parameters and optimizer state) is saved to `checkpoints` subfolder.

At the end of the training, the plots for the registered metrics are saved as well (new in 0.1.3).

Additionally, the losses for each data set as well as the registered metrics for each epoch and training run are saved to `runs/training_history.json` and are written to each run. Such results can look for instance like this:

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
               ]
          }
     ]
}
```
> To make the training results representation readable, the above output shows the training/validation results for only one epoch but in case of training for more epochs, there would be a longer list of such dictionaries distinguishable by `epoch`.

One can in fact quickly generate the plot of a specific metric for each epoch in case of its absence using a specific function for that:

```python
from epoch_engine.core.plotting import generate_plot_from_json

generate_plot_from_json(metric_name="accuracy", run_id="82cc72")
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
> The new checkpoints will be saved to the same folder for this run and new metrics will be appended to the same run ID's data in `training_process.json`. Additionally, the respective plots will also be updated.

## Test script

The basics of the developed API are presented in the `run_trainer.py` I built in the root of the repository. It can be run for instance as follows:

```bash
# Installing scikit-learn for using metrics
pip install -r requirements.txt

# Running the Trainer
python run_trainer.py --model=resnet --epochs=3 --enable-amp=True
```
> The training will be launched on the device automatically derived based on the CUDA availability and using mixed precision training (provided that the device is CUDA).

## TODOs

- [ ] Add gradient clipping
- [ ] Add lightweight tests
- [ ] Re-structure `TrainerConfig` and move some arguments to other methods in order not to overload the config
- [ ] Change the structure of the generated `runs` directory to allow for more convenient structure
- [x] Introduce an option to train using Automatic Mixed Precision (AMP)
- [x] Add plots generation for registered metrics within the tracking/logging system
