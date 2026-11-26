# Changelog – Part A

## Review fixes
* **Install**: `requirements.txt` was unsatisfiable (`datasets==5.2.0` needs `pyarrow>=24`) → `pyarrow==24.0.0`; every package pinned
  exactly; the full environment (torch included) is recorded in `requirements.lock.txt` by the Colab launcher.
* **Fine-tuning protocol**: the fixed 2,000-step budget gave 160 epochs at 1 % and only 1.4 epochs at 100 % (an under-trained supervised
  baseline). Now: budget proportional to the labelled subset (clipped to [500, 5000] steps), warm-up + cosine schedule, crop + flip,
  best-validation step kept, lr chosen on validation. The test config re-uses the validation-selected models instead of re-training.
* **Label fractions**: 1 % was 4/class (800 labels) due to banker's rounding → 5/class (1,000 labels); `labels_per_class` is in the CSV.
* **Stale results**: feature caches and fine-tuning rows are keyed by a checkpoint fingerprint; a pre-training run is skipped/resumed only
  if its stored settings match the config (explicit error otherwise).
* **Evidence**: 95 % bootstrap CIs for every accuracy and paired bootstrap differences between models (`paired_bootstrap.csv`).
* **Robustness**: the corruption seen during pre-training is marked in the CSV/figure; relative-to-clean curves; bonus test `patch_shuffle`.
* **A2 figure**: both clean and corrupted reconstruction error vs. k-NN, with Spearman ρ.
* **Interpretation**: retrieval grids for every model (same queries), WordNet ids in the per-class table.
* `ModelCheckpoint(save_last=True)` as in the assignment; `.gitignore` for data/checkpoints/features; thin Colab notebook.
