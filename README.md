# 🏋️‍♂️ EpochEngine: A Lightweight Training Framework for PyTorch

[![PyPI](https://img.shields.io/pypi/v/epoch-engine)](https://pypi.org/project/epoch-engine/)
[![Publish](https://github.com/spolivin/epoch-engine/actions/workflows/publish.yml/badge.svg)](https://github.com/spolivin/epoch-engine/actions/workflows/publish.yml)
[![License](https://img.shields.io/github/license/spolivin/epoch-engine)](https://github.com/spolivin/epoch-engine/blob/master/LICENSE.txt)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

**EpochEngine** is a minimal yet flexible library for training and validating PyTorch models, currently focused on computer vision classification tasks. It wraps the PyTorch training loop with automatic device detection, checkpointing, metric tracking, AMP support, and a callback system — without hiding your optimizer or model behind extra config layers.

**The project is currently under active development, more changes expected.**

## 🚀 Motivation behind the project

Training deep learning models often requires repeating the same boilerplate code: device management, checkpointing, resuming runs, logging results, and tracking metrics.
Existing frameworks like PyTorch Lightning or Hugging Face Trainer are powerful, but they can feel heavy for smaller projects or quick experiments.

This library started as a **lightweight training framework** I could use in my own computer vision experiments.

It provides:

- A _single constructor_ that accepts your model, optimizer, and loaders directly — no wrapper configs required
- Automatic _run management_ (checkpoints, plots, JSON history)
- Easy _resume support_ for interrupted or already finished runs
- A simple but _extendable interface_ for adding metrics, callbacks, and training options

The goal is not to replace bigger frameworks, but to make my own workflow faster, cleaner, and more reproducible — and to practice designing a training system from scratch.

## ✨ Key Features

- 🔧 **Simple setup** — pass your model, optimizer, scheduler, and metrics directly to `Trainer` — no wrapper configs needed.
- 💾 **Automatic checkpointing** — saves model weights and optimizer state after each epoch, with easy resuming via run ID.
- 📊 **Metric tracking & plotting** — logs training/validation metrics to JSON and generates plots at the end of training.
- 🚀 **Resuming training** — continue any previous run by providing its run ID.
- ⚡ **Mixed precision training** — supports AMP with `torch.amp` and `GradScaler`.
- 🔔 **Callbacks** — built-in early stopping, best checkpoint saving, checkpoint pruning, and NaN detection; fully extensible with custom callbacks.
- 🧩 **Extensible** — easily add custom models, losses, optimizers, schedulers, and metrics.

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

The project uses [uv](https://docs.astral.sh/uv/) for local development. After cloning the repo, install all dependencies (including optional extras) with:

```bash
# Cloning the repo and moving to the repo dir
git clone https://github.com/spolivin/epoch-engine.git
cd epoch-engine

# Installing the package and all optional deps (dev mode)
uv sync --all-extras
```

> `uv` resolves PyTorch from the CPU-only index by default, avoiding the large CUDA download during development. If you need CUDA support locally, install torch manually: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

Alternatively, with plain pip:

```bash
pip install -e .[build,linters,metrics,test]
```

### Pre-commit support

The repository provides support for running pre-commit checks via hooks defined in `.pre-commit-config.yaml`. These can be loaded in the current git repository by running:

```bash
pre-commit install
```

> `pre-commit` will already be available after running `uv sync --all-extras` or `pip install epoch-engine[linters]`

A pre-push hook is also configured to run the test suite automatically before each `git push`, but only when the push includes at least one `.py` file. To activate it:

```bash
pre-commit install --hook-type pre-push
```

## Python API

The examples below assume a model, a loss function, and dataloaders are already prepared. EpochEngine ships two built-in architectures; here we use `ResNet` as a starting point:

```python
from epoch_engine.models import ResNet

net = ResNet(
    in_channels=1,
    num_blocks=[3, 3, 3],
    num_classes=10,
)
```

`train_loader` and `valid_loader` are standard PyTorch `DataLoader` instances for the training and validation sets respectively. `test_loader` is optional and only needed for `trainer.evaluate()`.

### Trainer set-up

Instantiate `Trainer` directly by passing your PyTorch objects — no wrapper configs needed:

```python
import torch
import torch.nn as nn
from epoch_engine.core import Trainer

optimizer = torch.optim.SGD(net.parameters(), lr=0.25, momentum=0.75)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=2)

trainer = Trainer(
    model=net,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,      # optional
    scheduler=scheduler,          # optional
    scheduler_level="epoch",      # optional, "epoch" (default) or "batch"
    metrics=metrics,              # optional, see below
    callbacks=callbacks,          # optional, see below
    enable_amp=True,              # optional, AMP only activates on CUDA
)
```

> `Trainer` will automatically detect whether CUDA, MPS or CPU is to be used. The `scheduler` argument is optional; omitting it disables LR scheduling.

#### Metrics

By default only loss is tracked. Extra metrics can be added by passing a `dict` mapping a metric name to a callable with signature `(y_true, y_pred) -> float`:

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score

metrics = {
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="macro"),
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
}
```

> When using the dict form, plots are generated for all metrics at the end of training.

##### Fine-grained control with `MetricConfig`

Use `MetricConfig` when you want to selectively disable plots for specific metrics:

```python
from epoch_engine.core.configs import MetricConfig

metrics = [
    MetricConfig(name="accuracy", fn=accuracy_score, plot=True),
    MetricConfig(name="f1", fn=f1_metric, plot=False),  # tracked but not plotted
]
```

### Callbacks

Callbacks let you hook into the training lifecycle without modifying the `Trainer`. Pass a list of callback instances to the `callbacks` argument of the constructor:

```python
from epoch_engine.core.callbacks import EarlyStopping, BestCheckpoint, CheckpointPruner, NanCallback

trainer = Trainer(
    ...,
    callbacks=[
        BestCheckpoint(monitor="loss/valid", mode="min"),
        EarlyStopping(monitor="loss/valid", patience=5, mode="min"),
        CheckpointPruner(keep_last_n=2),
    ],
)
```

#### Built-in callbacks

##### `EarlyStopping`

Stops training when a monitored metric stops improving for `patience` consecutive epochs. Sets `trainer.interrupted = True` when triggered.

```python
EarlyStopping(
    monitor="loss/valid",   # metric key to watch
    patience=5,             # epochs without improvement before stopping
    mode="min",             # 'min' or 'max'
    min_delta=0.001,        # minimum change to count as improvement
    verbose=True,           # log warnings when patience counter increments
)
```

##### `BestCheckpoint`

Copies the current epoch checkpoint to `best.pt` whenever the monitored metric reaches a new best. Required before using `resume_from_best=True`.

```python
BestCheckpoint(
    monitor="loss/valid",   # metric key to watch
    mode="min",             # 'min' or 'max'
    min_delta=0.0,
    verbose=True,
)
```

##### `CheckpointPruner`

Deletes old `ckpt_epoch_*.pt` files after each save, keeping only the most recent `keep_last_n`. `best.pt` is never removed.

```python
CheckpointPruner(keep_last_n=2)
```

##### `NanCallback`

Stops training immediately if any metric value is NaN.

```python
NanCallback(
    monitor=["loss/train", "loss/valid"],  # specific keys to check; None checks all
    verbose=True,
)
```

#### Custom callbacks

Subclass `Callback` and override whichever hooks you need. Only `on_epoch_end` and `on_checkpoint_save` are currently dispatched by the trainer; the remaining hooks are available as no-op stubs for future use.

```python
from epoch_engine.core.callbacks import Callback

class PrintLR(Callback):
    def on_epoch_end(self, trainer, epoch, metrics):
        lr = trainer.optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} — lr={lr:.6f}")
```

To halt training from a custom callback, set `self.should_stop = True` inside `on_epoch_end`.

### Running the Trainer

#### New run

Now, we launch the training for the first time which will show the progress bar (if enabled):

```python
# Launching training (with gradient clipping)
trainer.run(
    epochs=5,
    seed=42,
    enable_tqdm=True,
    clip_grad_norm=1.0,
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
    |    └── plots/
    |        ├── accuracy.png
    |        ├── f1.png
    |        ├── loss.png
    |        └── precision.png
    |
    └── metrics_history.json
```

At the beginning of the run, a new `run_id` is generated (if `run_id=None`) and in the current folder the method creates `runs` folder with a separate folder for the files related to the current run (in the above example folder named `run_id=82cc72` with the generated run ID). After each epoch the checkpoint (containing last trained epoch, model parameters and optimizer state) is saved to `checkpoints` subfolder.

At the end of training, plots for the registered metrics are saved in the run-specific `plots` directory.

Additionally, the losses for each data set as well as the registered metrics for each epoch and training run are saved to `runs/metrics_history.json` and are written to each run. Such results can look for instance like this:

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
        }
      ]
    }
  ]
}
```

> To make the training results representation readable, the above output shows the training/validation results for only one epoch but in case of training for more epochs, there would be a longer list of such dictionaries distinguishable by `epoch`.

#### Metrics display

After each epoch, the trainer prints a formatted table with train and valid values for every tracked metric, side by side:

```
======================================================================
Epoch 3 Results:
----------------------------------------------------------------------
  accuracy             | Train:   0.7354 | Valid:   0.9829
  f1                   | Train:   0.7367 | Valid:   0.9827
  loss                 | Train:   0.7220 | Valid:   0.0564
  precision            | Train:   0.7451 | Valid:   0.9828
======================================================================
```

The table is printed via `tqdm.write` when `enable_tqdm=True` (the default), so it doesn't interfere with the progress bars. Pass `enable_tqdm=False` to use plain `print` instead.

#### Resuming training

The training can be easily resumed by specifying the `run_id` from which we would like to continue training:

```python
trainer.run(
    epochs=2,
    run_id="82cc72",
    seed=42,
    enable_tqdm=True,
)
```

> The new checkpoints will be saved to the same folder for this run and new metrics will be appended to the same run ID's data in `runs/metrics_history.json`. Additionally, the respective plots will also be updated.

Alternatively, if resuming training using the same `Trainer` instance, then we can just omit `run_id` whatsoever (equivalent to setting `run_id=None`), in which case `Trainer` will automatically infer that training for the last assigned `run_id` should be continued:

```python
trainer.run(epochs=2)
```

##### Resuming from the best checkpoint

By default, resuming always loads the last epoch checkpoint. If you used the `BestCheckpoint` callback in a prior run, you can instead resume from `best.pt`:

```python
trainer.run(epochs=2, resume_from_best=True)
```

> If `best.pt` does not exist (e.g. `BestCheckpoint` was not used), the trainer logs a warning and falls back to the last epoch checkpoint automatically.

### Testing the trained model

After the model has been trained and validated using `Trainer`, we can quickly test it on test set either in this way:

```python
test_metrics = trainer.evaluate()
```

or by passing a loader directly if `test_loader` was not set at construction time:

```python
test_metrics = trainer.evaluate(loader=test_loader)
```

## Running tests

The test suite uses [pytest](https://docs.pytest.org/). Install the test dependencies first, then run:

```bash
# Install with test extras (uv)
uv sync --extra test

# or with pip
pip install -e ".[test]"

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_trainer.py -v

# Run a single test
pytest tests/test_trainer.py::TestTrainer::test_run -v
```

## Test script

The basics of the developed API are presented in the `run_trainer.py` script in the root of the repository. It can be run via the provided `Makefile` targets:

```bash
# Run with EDNet model (default 5 epochs)
make run-trainer-ednet

# Run with ResNet model (default 5 epochs)
make run-trainer-resnet

# Override the number of epochs
make run-trainer-ednet EPOCHS=3
make run-trainer-resnet EPOCHS=10
```

> The training will be launched on the automatically detected device (CUDA → MPS → CPU). Metric plots are generated at the end of each run.

If `make` is not available (e.g. on Windows), run the script directly:

```bash
python run_trainer.py --model resnet --epochs 5 --plot-metrics
python run_trainer.py --model ednet --epochs 5 --plot-metrics
```

## TODOs

- [x] Add gradient clipping
- [x] Add lightweight tests
- [x] Simplify trainer setup — accept optimizer/scheduler directly, removing `OptimizerConfig`, `SchedulerConfig`, and `TrainerConfig`
- [x] Change the structure of the generated `runs` directory to allow for more convenient structure
- [x] Introduce an option to train using Automatic Mixed Precision (AMP)
- [x] Add plots generation for registered metrics within the tracking/logging system
- [x] Introduce callbacks for early stopping training, saving best checkpoint, etc.
- [x] Come up with a way to track metrics live during training/validation
- [ ] Introduce multi-GPU training
