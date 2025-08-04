# Epoch Engine - Python Library for training PyTorch models

This project represents my attempt to come up with a convenient way to train neural nets coded in Torch. While being aware of already existing libraries for training PyTorch models (e.g. PyTorch Lightning), my idea here is to make training of the models more visual and understandable as to what is going on during training.

The project is currently in its raw form, more changes expected.

## Features

* TQDM-Progress bar support for both training and validation loops
* Intemediate metrics computations after each forward pass (currently it is based on computing loss and accuracy only)
* Saving/loading checkpoints from/into Trainer directly without having to touch model, optimizer or scheduler separately
* Resuming training from the loaded checkpoint with epoch number being remembered automatically to avoid having to remember from which epoch the training originally started
* Ready-to-use neural net architectures coded from scratch (currently only 4-layer Encoder-Decoder and ResNet with 20 layers architectures are available)

## Installation

After cloning this repo, the package can be installed in the development mode as follows:

```bash
# Installing the main package
pip install -e .

# Installing additional optional dependencies
pip install -e .[build,linters]
```

## Python API

The basics of the developed API are presented in the [test script](./run_trainer.py) I built. It can be run for instance as follows:

```bash
python run_trainer.py --model=resnet --epochs=3 --batch-size=16
```
> The training will be launched on the device automatically derived based on the CUDA availability and the final training checkpoint will be saved in `checkpoints` directory.

One can also resume the training from the saved checkpoint:

```bash
python run_trainer.py --model=resnet --epochs=4 --resume-training=True --ckpt-path=checkpoints/ckpt_3.pt
```
> The training will be resumed from the loaded checkpoint with TQDM-progress bar showing the next training epoch.
