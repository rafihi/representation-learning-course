# Midterm – Part A: Denoising-autoencoder representations on Tiny-ImageNet

Fork of the course repository (<https://github.com/SivanSchwartz/representation-learning-course>).
All Part-A code lives in this folder. Every run is launched by `train.py` with its own YAML config:

```bash
cd midterm_a
python train.py --config configs/<name>.yaml            # override anything: --set trainer.max_epochs=40
```

* Methods are `LightningModule`s (`src/dae.py`, `src/classifier.py`); data is the `TinyImageNetDM` `LightningDataModule` (`src/data.py`).
* `train.py` parses the config, applies `--set` overrides, calls `L.seed_everything(seed)` and dispatches to the task in `src/tasks.py`.
* Logs: TensorBoard event files in `runs/<run>/tb/`. **Logs: `<PASTE THE SHARED LINK TO THE TENSORBOARD LOGS / W&B PROJECT HERE>`**
  (W&B instead of TensorBoard: `--set logging.logger=wandb logging.project=ssl-midterm`).
* Datasets, checkpoints and cached features are never committed (`.gitignore`).

## 1. Setup
```bash
cd midterm_a
pip install -r requirements.txt        # torch/torchvision: the CUDA build preinstalled in Colab
pip freeze > requirements.lock.txt     # exact environment of the runs (torch included) - commit this file
```
Tiny-ImageNet (HF `zh-plus/tiny-imagenet`) and the transfer dataset (CIFAR-100, resized to 64×64) download on first use into `data.cache_dir` (default `data_cache/`).

**Splits.** 50 images/class (10k) of the HF `train` split are held out as **validation** and used for *all* tuning and selection.
Pre-training uses the remaining 450/class (90k; proxy runs use the first 100/class of them). The official `valid` split (10k) is the **test** set,
touched only by the `*_test.yaml` configs. The split seed is fixed and independent of the run seed, so every model sees identical splits.

## 2. Check that everything works (no downloads, no GPU)
```bash
SSL_FAKE_DATA=1 SSL_TINY_BACKBONE=1 python -m pytest -q tests     # unit tests
SSL_TINY_BACKBONE=1 bash scripts/smoke_all.sh                       # every config (val + test) end-to-end on synthetic data
```
`SSL_TINY_BACKBONE=1` swaps ResNet-18 for a tiny CNN so the check runs on a CPU (tests only). `--smoke` shrinks any config.

## 3. Commands that reproduce every table and figure

Map from the assignment sections to configs (config file names are historical; the table is authoritative):

| Assignment | Item | Command | Output |
|---|---|---|---|
| **A2** | Tuning: random search over corruption kind / level / lr / weight decay (proxy setting), selected by **validation k-NN**; **figure: reconstruction vs. k-NN** | `python train.py --config configs/a1_tune_dae.yaml` | `results/a1_tune_dae/results.csv`, `recon_vs_knn.png` |
| A1 | Final DAE (full setting, best trial's hyper-parameters) | `python train.py --config configs/a2_dae_final.yaml` | `runs/dae_final/` |
| A3 | Plain-counterpart baseline: vanilla AE (same settings, no corruption) | `python train.py --config configs/a3_vanilla_ae_final.yaml` | `runs/vanilla_ae_final/` |
| **A3** | Frozen protocol table: k-NN + linear probe (top-1/top-5, 95 % CIs) on Tiny-ImageNet + CIFAR-100, random-init / vanilla AE / DAE | `python train.py --config configs/a4_eval.yaml` | `results/a4_eval/results.csv`, `paired_bootstrap.csv` |
| **A3/A4** | Fine-tuning 1 / 10 / 100 % labels incl. supervised-from-scratch; **label-efficiency figure** | `python train.py --config configs/a5_finetune.yaml` | `results/a5_finetune/results.csv`, `label_efficiency.png`, `paired_bootstrap.csv` |
| **A4** | Corruption curves, k-NN and linear probe (gaussian, mask, salt&pepper, contrast, blur) + **bonus** `patch_shuffle` | `python train.py --config configs/a6_robustness.yaml` | `results/a6_robustness/robustness.csv`, `robustness*.png` |
| **A5** (+A1, A2) | Per-class accuracy (failure classes), t-SNE, retrieval grids; **layer choice for h** (A2); **reconstructions** x / x~ / g(f(x~)) (A1) | `python train.py --config configs/a7_interpret.yaml` | `results/a7_interpret/*` (`layer_choice.png/csv`, `reconstructions.png`) |

The A2 sweep prints the best trial at the end; its values are already in `configs/a2_dae_final.yaml`.

**Final (test) numbers for the report**, after every decision has been made on validation:

| Final test item | Command | Output |
|---|---|---|
| A3 frozen protocol | `python train.py --config configs/a4_eval_test.yaml` | `results/final_test/a4_eval/` |
| A3/A4 fine-tuning | `python train.py --config configs/a5_finetune_test.yaml` | `results/final_test/a5_finetune/` |
| A4 robustness | `python train.py --config configs/a6_robustness_test.yaml` | `results/final_test/a6_robustness/` |
| A5 interpretation | `python train.py --config configs/a7_interpret_test.yaml` | `results/final_test/a7_interpret/` |

or all four: `bash scripts/run_final_test.sh`. The test configs extend the validation configs and change only `split`/`out_dir`
(+ `reuse_selected_from` for fine-tuning). The linear-probe lr and the fine-tuned models are selected on **validation only**;
`a5_finetune_test.yaml` does **not** re-train: it evaluates the models selected by `a5_finetune.yaml` (saved in `results/a5_finetune/selected/`) on test once,
so run `a5_finetune.yaml` first.

Runs are **resumable** (re-run the same command after a Colab disconnect). A finished or half-finished run is reused **only if it was made
with the same settings**; otherwise the code stops with an explicit error (use a new `out_dir`). Cached features and fine-tuning rows are
keyed by a fingerprint of the checkpoint, so re-training a model never silently reuses stale results.

## 4. Design decisions
* **h** = the 512-d global-average-pool output of ResNet-18 with a 3×3/stride-1 stem and no max-pool (`src/backbone.py`): the last
  layer shared by every downstream protocol and the representation the assignment defines; the decoder sees only `h`, so everything
  it needs to reconstruct must pass through the representation that is evaluated. Evidence: `layer_choice.png` compares the
  k-NN accuracy of the pooled `layer2` / `layer3` / `layer4` (= h) outputs for every model (decided on validation).
* **DAE**: the encoder sees `x~ ~ q(x~|x)`; a light decoder (linear → 4×4×256 → four stride-2 transposed convs → sigmoid) reconstructs the
  clean `x` with MSE. Selection by validation k-NN; reconstruction is logged as `val_recon` (clean→clean, comparable across trials)
  and `val_recon_corrupted` (corrupted→clean, the DAE objective). Both are plotted against k-NN.
* **k-NN**: cosine similarity on ℓ2-normalised `h`, k = 20, similarity-weighted vote with weight `exp(sim/0.1)` (`knn.weight: sim` gives plain cosine weights).
* **Linear probe**: frozen, cached features standardised with train statistics (an affine map, so still linear in `h`), AdamW + cosine, 100 epochs, lr ∈ {1e-3, 3e-3} chosen on validation.
* **Fine-tuning**: all weights; class-balanced **nested** label subsets identical for every model
  (Tiny-ImageNet: 1 % = 5/class = 1,000 labels, 10 % = 45/class = 9,000, 100 % = 450/class = 90,000).
  Same recipe for every model: AdamW, 5 % warm-up + cosine, random crop (pad 4) + horizontal flip, batch 128.
  Budget = 30 passes over the labelled subset clipped to [500, 5,000] steps (1 % → 500, 10 % → 2,110, 100 % → 5,000 ≈ 7 epochs);
  validation is run 10× per run and the best-validation step is kept; lr ∈ {1e-3, 3e-4} chosen on validation per (model, fraction).
  The 5,000-step cap at 100 % is a compute limit (free Colab) and is stated as such in the report.
* **Uncertainty**: every accuracy has a 95 % bootstrap CI over evaluation images; `paired_bootstrap.csv` gives the paired difference
  between every two models on the same images (difference, 95 % CI, one-sided p).
* **Robustness**: corruptions are applied to the evaluation images only, with identical noise for every model. k-NN uses a clean
  memory bank; the linear probe is trained on clean features (lr chosen on clean validation) and tested on corrupted ones.
  The corruption the DAE was trained on is marked **SEEN** (dashed) — it is not evidence of robustness to unseen shifts.
* **Bonus robustness test – `patch_shuffle`**: the image is cut into a g×g grid (g = 2, 4, 8) and the patches are randomly permuted.
  *Weakness probed*: does `h` encode object layout/shape, or only local texture and colour statistics? *Fair*: no model saw it in training,
  local statistics are preserved exactly, identical permutations for every model, clean memory bank. *Reading*: a representation that keeps
  its accuracy under shuffling is texture/colour-driven; a steep drop means it uses spatial structure (compare against random-init).
* **Colab**: `precision: 16-mixed`, channels-last, data resident in memory, corruptions/augmentation on the GPU,
  `ModelCheckpoint(save_last=True)` + automatic resume from `ckpt/last.ckpt`.

## 5. Compute (free Colab T4) – estimates, not measurements
Time one epoch first. Tuning runs in the proxy setting only; final runs use identical settings for every compared model.
Fine-tuning is the most expensive table (3 models × 2 lrs × 3 fractions × 2 datasets). If GPU time is short, reduce it **for every model
alike** and say so in the report, e.g. `--set ft.max_steps=3000` or `--set "ft.lr_grid=[3.0e-4]"`.

## 6. Layout
```
train.py                 entry point (config -> seed -> task)
configs/                 one YAML per run / table / figure (+ *_test.yaml for the final numbers)
src/                     backbone, dae, classifier, corruptions, data, evaluation, tasks, plots, callbacks, registry, common
tests/  scripts/         unit tests, end-to-end smoke test, final-test runner
colab_midterm_a.ipynb    thin Colab launcher (clone, install, run train.py)
report/                  report + slides skeletons
```
