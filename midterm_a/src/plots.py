"""All figures. Every function writes a PNG and returns it (or a small dict of numbers)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from scipy.stats import spearmanr


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def a2_scatter(df, path):
    """A2: validation reconstruction error vs validation k-NN accuracy, one point per trial.
    Left: clean->clean reconstruction (comparable across trials). Right: corrupted->clean (the DAE objective)."""
    cols = [("val_recon", "val. reconstruction MSE (clean input)"),
            ("val_recon_corrupted", "val. denoising MSE (corrupted input)")]
    cols = [c for c in cols if c[0] in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(5.5 * len(cols), 4.2), squeeze=False)
    b = df.loc[df["knn_top1"].idxmax()]
    out = {}
    for ax, (col, label) in zip(axes[0], cols):
        for kind, g in df.groupby("model.kind"):
            ax.scatter(g[col], g["knn_top1"], s=55, label=str(kind))
        ax.scatter([b[col]], [b["knn_top1"]], s=260, facecolors="none", edgecolors="k", label="best by k-NN")
        rho = spearmanr(df[col], df["knn_top1"])[0] if len(df) > 2 else float("nan")
        out[f"spearman_{col}"] = rho
        ax.set_xlabel(label)
        ax.set_ylabel("validation k-NN top-1 (%)")
        ax.set_title(f"Spearman rho = {rho:.2f} (n = {len(df)})")
        ax.legend(fontsize=8)
    fig.suptitle("Does better reconstruction mean a better representation?", fontsize=10)
    _save(fig, path)
    return out


def label_efficiency(df, path):
    """A4: accuracy vs fraction of labels, one panel per dataset, 95% bootstrap CI as error bars."""
    names = list(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4), squeeze=False)
    for ax, dn in zip(axes[0], names):
        for m, g in df[df["dataset"] == dn].groupby("model", sort=False):
            g = g.sort_values("frac")
            x = g["frac"] * 100
            if {"top1_ci_lo", "top1_ci_hi"} <= set(g.columns):
                yerr = np.vstack([g["top1"] - g["top1_ci_lo"], g["top1_ci_hi"] - g["top1"]])
                ax.errorbar(x, g["top1"], yerr=yerr, marker="o", capsize=3, label=m)
            else:
                ax.plot(x, g["top1"], marker="o", label=m)
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1%", "10%", "100%"])
        ax.set_xlabel("labels used")
        ax.set_ylabel("fine-tuned top-1 (%)")
        ax.set_title(f"Label efficiency: {dn}")
        ax.legend(fontsize=8)
    _save(fig, path)


def robustness(df, path, y="knn_top1"):
    """A4: k-NN top-1 under increasing corruption, one panel per corruption kind.
    Panels whose corruption a model saw during pre-training are marked (that curve is not an 'unseen' test)."""
    kinds = list(df["kind"].unique())
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.6 * len(kinds), 3.6), squeeze=False)
    for ax, k in zip(axes[0], kinds):
        sub = df[df["kind"] == k]
        seen = []
        for m, g in sub.groupby("model", sort=False):
            is_seen = bool(g.get("seen_in_pretraining", False).any()) if "seen_in_pretraining" in g else False
            ax.plot(g["level"], g[y], marker="o", linestyle="--" if is_seen else "-",
                    label=f"{m} (seen in training)" if is_seen else m)
            if is_seen:
                seen.append(m)
        ax.set_title(k + (" - SEEN by " + ", ".join(seen) if seen else " - unseen"), fontsize=9)
        ax.set_xlabel("level")
        ax.set_ylabel("k-NN top-1 (%)" if y == "knn_top1" else "accuracy / clean accuracy")
        ax.legend(fontsize=6)
    _save(fig, path)


def tsne_grid(feats, y, path, n_classes=10):
    """t-SNE of the first n_classes classes for every encoder (intuition only, not proof)."""
    from sklearn.manifold import TSNE
    m = y < n_classes
    fig, axes = plt.subplots(1, len(feats), figsize=(4.6 * len(feats), 4.2), squeeze=False)
    for ax, (name, f) in zip(axes[0], feats.items()):
        z = F.normalize(f[m], dim=1).numpy()
        emb = TSNE(n_components=2, perplexity=min(30, max(2, len(z) // 4)), init="pca", random_state=0).fit_transform(z)
        ax.scatter(emb[:, 0], emb[:, 1], c=y[m].numpy(), cmap="tab10", s=10)
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
    _save(fig, path)


def retrieval_grid(title, f_tr, f_ev, X_tr, y_tr, X_ev, y_ev, path, n_queries=8, n_nn=6):
    """Query image + its nearest train neighbours. Green frame = same class, red = different class.
    Queries are evenly spaced over the evaluation split (same queries for every model)."""
    q = np.linspace(0, len(X_ev) - 1, n_queries).astype(int)
    nn_idx = (F.normalize(f_ev[q], dim=1) @ F.normalize(f_tr, dim=1).T).topk(n_nn, dim=1).indices
    fig, axes = plt.subplots(n_queries, n_nn + 1, figsize=(1.6 * (n_nn + 1), 1.6 * n_queries), squeeze=False)
    hits = 0
    for r, qi in enumerate(q):
        axes[r, 0].imshow(X_ev[qi].permute(1, 2, 0).numpy())
        axes[r, 0].set_title("query", fontsize=7)
        for c, j in enumerate(nn_idx[r].tolist()):
            ax = axes[r, c + 1]
            ax.imshow(X_tr[j].permute(1, 2, 0).numpy())
            same = bool(y_tr[j] == y_ev[qi])
            hits += same
            for sp in ax.spines.values():
                sp.set_edgecolor("green" if same else "red")
                sp.set_linewidth(3)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{title}: {hits}/{n_queries * n_nn} neighbours share the query's class", fontsize=9)
    _save(fig, path)
