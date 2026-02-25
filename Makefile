EPOCHS ?= 5

run-trainer-ednet:
	uv run run_trainer.py --model ednet --epochs $(EPOCHS) --plot-metrics
run-trainer-resnet:
	uv run run_trainer.py --model resnet --epochs $(EPOCHS) --plot-metrics
