# Report skeleton - Part A (max 6 pages + references + appendix)

> A skeleton, not content. Every claim must be backed by a number, a baseline or a figure from YOUR runs.

## 1. Methods
* **Data & splits:** Tiny-ImageNet; 450/class train, 50/class val (all tuning), official valid = test (final numbers only). Transfer: CIFAR-100 resized to 64x64 (450/class train, 50/class val, 10k test).
* **Backbone / h:** ResNet-18, 3x3 stride-1 stem, no max-pool; h = 512-d global-avg-pool output. Why this layer: ____.
* **DAE:** L = E||x - g(f(x~))||^2 ; corruption ____ (level ____) ; light decoder (linear + 4 transposed convs) ; AdamW, 60 epochs, batch 128.
* **Tuning:** random search, 20 trials, proxy setting (100 img/class, 15 epochs), ranges (table), selection by val k-NN.
* **Baselines:** random-init (frozen), vanilla AE (same settings, no corruption), supervised ResNet-18 from scratch.
* **Evaluation protocol:** k-NN (k=20, cosine, exp(sim/0.1) vote), linear probe (frozen, lr on val), fine-tuning 1/10/100 %
  (nested balanced subsets: 1,000 / 9,000 / 90,000 labels; budget 30 passes clipped to [500, 5000] steps; best val step; lr on val).
* **Table of ALL hyper-parameters and search ranges** (copy from `configs/*.yaml`): epochs, batch size, lr, wd, level, kind, seeds.

## 2. Results
| Figure/Table | File |
|---|---|
| A2 reconstruction vs. k-NN | `results/a1_tune_dae/recon_vs_knn.png` |
| A3 main table (k-NN, linear, top-1/top-5, Tiny + transfer, 95 % CI) | `results/final_test/a4_eval/results.csv` |
| A3 "better by how much" (paired differences + CI) | `results/final_test/a4_eval/paired_bootstrap.csv` |
| Fine-tuning table + A4 label efficiency | `results/final_test/a5_finetune/` |
| A4 corruption curves (seen vs. unseen marked) + bonus | `results/final_test/a6_robustness/robustness.png` |
| A5 t-SNE / retrieval / per-class | `results/final_test/a7_interpret/` |

Single pre-training seed: say so; the CIs cover evaluation-set sampling, not training randomness.

## 3. Discussion prompts (answer each with evidence)
* Does better reconstruction mean a better representation? (Spearman of both panels, which trials break the trend.)
* Better than what, and by how much? (paired differences vs. random-init, vanilla AE, supervised-from-scratch, with CIs.)
* Do k-NN, linear probe and fine-tuning rank the models identically? If not, why?
* Where does the representation fail: classes (`per_class_knn.csv`, `worst_10_classes.json`), corruptions (unseen ones!), domains (CIFAR-100)?
* What extra evidence convinces you (t-SNE caveats, retrieval grids for all models, ...)?
* Bonus `patch_shuffle`: which weakness it probes (shape vs. texture), why it is fair, what it shows.
* Limitations and what you would test next (proxy vs. full gap, single seed, fine-tuning step cap at 100 %, k and T choices, other transfer sets).

## 4. Reproducibility
Commit hash, seeds, `requirements.txt` + `requirements.lock.txt`, link to logs, commands from README.
