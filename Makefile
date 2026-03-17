EPOCHS ?= 5 # Default number of epochs for training

run-trainer-ednet:
	uv run run_trainer.py --model ednet --epochs $(EPOCHS) --plot-metrics
run-trainer-resnet:
	uv run run_trainer.py --model resnet --epochs $(EPOCHS) --plot-metrics
run-trainer-regression:
	uv run run_trainer.py --task regression --epochs $(EPOCHS) --plot-metrics
run-tests:
	uv run pytest
build:
	uv build
build-check:
	uvx twine check dist/*
