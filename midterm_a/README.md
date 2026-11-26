# Midterm - Part A: Denoising-autoencoder representations on Tiny-ImageNet

Self-contained folder `midterm_a/` for the fork <https://github.com/rafihi/representation-learning-course>
(forked from <https://github.com/SivanSchwartz/representation-learning-course>). The Midterm-specific model, data, and task code lives in `src/`; the Part-A `train.py` is the command-line entry point, implemented locally using the structure and conventions of the course training script.
**Run Part-A commands from inside `midterm_a/`:**
`python train.py --config configs/<name>.yaml`.
The local entry point follows the course training-script pattern: it parses a run configuration, applies overrides, sets the random seed, and dispatches to a Lightning task. The original `lec1/`–`lec5/` files are not modified. This is a Part-A-local adapter following the course entry-point pattern; the assignment wording should be checked with the instructor if it is interpreted as requiring the exact shared lecture file.
The Midterm-A task implementations remain in this folder: models are `LightningModule`s (`src/dae.py`, `src/classifier.py`),
data is managed by the `TinyImageNetDM` `LightningDataModule` (`src/data.py`), and each experiment has its own YAML config.
Seeds are fixed with `L.seed_everything`. TensorBoard logs are written under each run's output directory in `tb/`.

> The lecture-specific MNIST/Fashion-MNIST models and DataModules are not reused unchanged because they do not match Tiny-ImageNet. The Part-A code keeps the course conventions while implementing the task-specific LightningModules, LightningDataModule and Trainer setup in `src/`.

## 1. Setup
```bash
cd midterm_a
pip install -r requirements.txt        # torch/torchvision: use the versions provided by your runtime
python -m pytest -q tests
```
Tiny-ImageNet (HF `zh-plus/tiny-imagenet`) and the transfer dataset download on first use into `data.cache_dir` (default `data_cache/`).
**Splits:** 50 images/class (10k) of `train` are held out as **validation** (all tuning); the official `valid` set (10k) is the **test** set,
used only through the dedicated `*_test.yaml` final-evaluation configs. The split seed is fixed and independent of the run seed.

## 2. Check that everything works (no downloads, no GPU needed)
```bash
SSL_FAKE_DATA=1 SSL_TINY_BACKBONE=1 python -m pytest -q tests  # unit tests
SSL_TINY_BACKBONE=1 bash scripts/smoke_all.sh                 # every task via the local Part-A train.py entry point on synthetic data
python train.py --config configs/a1_tune_dae.yaml --smoke  # any config can be shrunk with --smoke
```
(`SSL_TINY_BACKBONE=1` swaps ResNet-18 for a tiny CNN so it runs on a CPU; tests only. On a GPU machine drop it.)

## 3. Commands that reproduce every table and figure
Runs are **resumable**: re-run the same command after a Colab disconnect (`resume: true`; finished runs/trials are skipped).
Override anything with `--set key.sub=value`, e.g. `--set trainer.max_epochs=40`.

| Item | Command | Output |
|---|---|---|
| **A2** tuning (random search over kind/level/lr/weight decay, selected by validation k-NN) + **figure: reconstruction vs. k-NN** | `python train.py --config configs/a1_tune_dae.yaml` | `results/a1_tune_dae/results.csv`, `recon_vs_knn.png` |
| Final DAE (copy best hyper-parameters into the config first) | `python train.py --config configs/a2_dae_final.yaml` | `runs/dae_final/` |
| Vanilla AE baseline (no corruption) | `python train.py --config configs/a3_vanilla_ae_final.yaml` | `runs/vanilla_ae_final/` |
| **A3** table: random-init / vanilla AE / DAE; k-NN + linear probe (top-1/top-5) on Tiny-ImageNet + transfer | `python train.py --config configs/a4_eval.yaml` | `results/a4_eval/results.csv` |
| **A3/A4** fine-tuning 1/10/100 % labels + supervised-from-scratch; **label-efficiency figure** | `python train.py --config configs/a5_finetune.yaml` | `results/a5_finetune/results.csv`, `label_efficiency.png` |
| **A4** **corruption curves** (gaussian, mask, salt&pepper, contrast, blur) | `python train.py --config configs/a6_robustness.yaml` | `results/a6_robustness/robustness.csv`, `robustness.png` |
| **A5** per-class accuracy, t-SNE, nearest-neighbour grid | `python train.py --config configs/a7_interpret.yaml` | `results/a7_interpret/*` |

**Final numbers for the report:** use the dedicated test-only configs below. In A4/A5, learning-rate selection is performed on Validation only; the selected probe/fine-tuned model is then evaluated on Test once. Do not change validation configs or use test metrics to make design choices.

| Final test item | Command | Output |
|---|---|---|
| A4 frozen evaluation | `python train.py --config configs/a4_eval_test.yaml` | `results/final_test/a4_eval/results.csv` |
| A5 fine-tuning | `python train.py --config configs/a5_finetune_test.yaml` | `results/final_test/a5_finetune/results.csv` |
| A6 robustness | `python train.py --config configs/a6_robustness_test.yaml` | `results/final_test/a6_robustness/robustness.csv` |
| A7 interpretation | `python train.py --config configs/a7_interpret_test.yaml` | `results/final_test/a7_interpret/` |

Or run all four final evaluations in sequence with `bash scripts/run_final_test.sh` after confirming the checkpoints are present.

These test configs extend their matching validation configs and change only `split` and `out_dir`. `a4_eval_test.yaml` and `a5_finetune_test.yaml` never use Test while choosing learning rates.

## 4. What you have to fill in (marked `>>>` in the configs)
1. `configs/a2_dae_final.yaml`: best `kind, level, lr, weight_decay` from `results/a1_tune_dae/results.csv`.
2. Optionally another transfer set: `transfer.name: cifar100 | stl10 | pets` (more in `src/data.py::_raw_transfer`).
3. **Bonus (A4):** add your own corruption to `src/corruptions.py` (`REGISTRY`) and list it in `configs/a6_robustness.yaml`.

## 5. Design decisions (state them in the report)
* `h` = 512-d output of the global average pool of a ResNet-18 with a 3x3/stride-1 stem and no max-pool (`src/backbone.py`).
* DAE: the encoder sees the corrupted `x~`, a light decoder reconstructs the clean `x` (MSE). Models are selected by **validation k-NN**;
  reconstruction is logged as `val_recon` (clean in, clean out: comparable across trials) and `val_recon_corrupted`.
* k-NN: cosine on l2-normalised `h`, k=20, vote weight `exp(sim/0.1)`. Linear probe: standardised frozen features, AdamW, cosine schedule, lr chosen on validation.
  Features are extracted once and cached in `results/*/features/`.
* Fine-tuning: all weights, class-balanced **nested** label subsets identical for every model, fixed number of steps, lr chosen on validation per (model, fraction), horizontal flip only.
* Colab: `precision: 16-mixed`, channels-last, data resident in memory, corruptions on the GPU, `ModelCheckpoint(save_last=True)` + automatic resume.

## 6. Compute budget (free Colab T4) - read before launching
Speed numbers are **estimates, not measurements**. Time one proxy epoch first, then scale `data.train_per_class`, `trainer.max_epochs`, `n_trials`.
Tune only in the proxy setting; final runs in the full setting, with identical settings for all compared models.
Reducing `max_epochs` / `train_per_class` in the final configs is legitimate if the report says so.

## 7. Layout
```
train.py                 Part-A entry point following the course training-script pattern
configs/                 one YAML per run / table / figure
src/                     backbone, dae, corruptions, data, evaluation, classifier, tasks, plots, callbacks, registry, common
tests/  scripts/         unit tests, end-to-end smoke test
colab_midterm_a.ipynb    thin Colab launcher
report/                  report + slides templates
```
