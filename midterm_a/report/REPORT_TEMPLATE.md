# Report skeleton - Part A (max 6 pages + references + appendix)

> A skeleton, not content. Every claim must be backed by a number, a baseline or a figure from YOUR runs.

## 1. Methods
* **Data & splits:** Tiny-ImageNet; 450/class train, 50/class val (all tuning), official valid = test (final numbers only). Transfer dataset: ____.
* **Backbone / h:** ResNet-18, 3x3 stride-1 stem, no max-pool; h = 512-d global-avg-pool output. Why this layer: ____.
* **DAE:** L = E||x - g(f(x~))||^2 ; corruption(s) ____ ; decoder ____ ; optimiser ____.
* **Tuning:** random search, N trials, ranges (table), selection by val k-NN.
* **Baselines:** random-init (frozen), vanilla AE, supervised ResNet-18 from scratch.
* **Evaluation protocol:** k-NN (k=20, cosine, weighted vote), linear probe (frozen), fine-tuning 1/10/100 % (nested balanced subsets, steps ____).
* **Table of ALL hyper-parameters and search ranges** (copy from `configs/*.yaml`): epochs, batch size, lr, wd, level, kind, seeds.

## 2. Results
| Figure/Table | File |
|---|---|
| A2 reconstruction vs. k-NN | `results/a1_tune_dae/recon_vs_knn.png` |
| A3 main table (k-NN, linear, top-1/top-5, Tiny + transfer) | `results/a4_eval/results.csv` |
| Fine-tuning table + A4 label efficiency | `results/a5_finetune/` |
| A4 corruption curves | `results/a6_robustness/robustness.png` |
| A5 t-SNE / retrieval / per-class | `results/a7_interpret/` |

Report mean +- std over seeds if you have several; say clearly if you have one.

## 3. Discussion prompts (answer each with evidence)
* Does better reconstruction mean a better representation? (Spearman, scatter, which trials break the trend.)
* Better than what, and by how much? (random-init, vanilla AE, supervised-from-scratch.)
* Do k-NN, linear probe and fine-tuning rank the models identically? If not, why?
* Where does the representation fail: classes (`per_class_knn.csv`), corruptions, domains (transfer)?
* What extra evidence convinces you (t-SNE caveats, retrieval grid, ...)?
* Bonus: your own robustness test: which weakness it probes, why it is fair, what it shows.
* Limitations and what you would test next (proxy vs. full gap, single seed, k and T choices, other transfer sets).

## 4. Reproducibility
Commit hash, seeds, `requirements.txt`, link to logs, commands from README.
