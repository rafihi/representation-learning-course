# Part A – implementation notes

## Entry point and repository scope

- Part A now has a local `midterm_a/train.py` entry point, based on the structure of the course training script.
- It accepts `--config`, `--set`, and `--smoke`, applies the YAML configuration and overrides, calls `lightning.seed_everything`, and dispatches to the selected task in `src/tasks.py`.
- All experiment implementations remain in `src/`: models are LightningModules, the Tiny-ImageNet data pipeline is a LightningDataModule, and task functions create the Lightning Trainer.
- The Colab notebook, README, scripts, CLI, configs, tests, and dependency pins are all inside `midterm_a/`. The shared `lec1/`–`lec5/` folders are unchanged.

## Test protocol

- A4 selects the linear-probe learning rate using Validation only, then extracts/evaluates Test features after selection.
- A5 selects the fine-tuning learning rate/model state using Validation only, then evaluates the selected state on Test once.
- Dedicated `*_test.yaml` configs inherit the Validation configs and change the split/output directory.
- Test outputs are written below `results/final_test/`.

## Colab

- `colab_midterm_a.ipynb` sparse-checks out only `midterm_a/` and launches `python train.py ...` from that folder.
- The dependency pins use `pandas==2.2.3` and `pyarrow==23.0.1` to avoid the pandas/pyarrow version conflicts seen in the Colab setup output.

## Validation performed in this edit

- `SSL_FAKE_DATA=1 SSL_TINY_BACKBONE=1 python -m pytest -q tests`: **10 passed**.
- Synthetic-data smoke runs through the new local entry point were completed for the pretraining run, the A1 sweep, frozen evaluation (A4), fine-tuning (A5), robustness (A6), and interpretation (A7).
- The aggregate `scripts/smoke_all.sh` was attempted on the CPU sandbox but did not finish before the execution time limit. This is not recorded as a pass.
- No real Tiny-ImageNet training or final Test evaluation was run in this environment.
