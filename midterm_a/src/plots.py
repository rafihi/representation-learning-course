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
    """A2: validation reconstruction error vs validation k-NN accuracy, one point per trial."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for kind, g in df.groupby("model.kind"):
        ax.scatter(g["val_recon"], g["knn_top1"], s=55, label=str(kind))
    b = df.loc[df["knn_top1"].idxmax()]
    ax.scatter([b["val_recon"]], [b["knn_top1"]], s=260, facecolors="none", edgecolors="k", label="best by k-NN")
    rho = spearmanr(df["val_recon"], df["knn_top1"])[0] if len(df) > 2 else float("nan")
    ax.set_xlabel("validation reconstruction MSE (clean input)")
    ax.set_ylabel("validation k-NN top-1 (%)")
    ax.set_title(f"Reconstruction vs. representation (Spearman {rho:.2f})")
    ax.legend(fontsize=8)
    _save(fig, path)
    return {"spearman": rho}


def label_efficiency(df, path):
    """A4: accuracy vs fraction of labels, one panel per dataset."""
    names = list(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4), squeeze=False)
    for ax, dn in zip(axes[0], names):
        for m, g in df[df["dataset"] == dn].groupby("model", sort=False):
            g = g.sort_values("frac")
            ax.plot(g["frac"] * 100, g["top1"], marker="o", label=m)
        ax.set_xscale("log")
        ax.set_xlabel("labels used (%)")
        ax.set_ylabel("top-1 (%)")
        ax.set_title(f"Label efficiency: {dn}")
        ax.legend(fontsize=8)
    _save(fig, path)


def robustness(df, path):
    """A4: k-NN top-1 under increasing corruption, one panel per corruption kind."""
    kinds = list(df["kind"].unique())
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.6 * len(kinds), 3.6), squeeze=False)
    for ax, k in zip(axes[0], kinds):
        for m, g in df[df["kind"] == k].groupby("model", sort=False):
            ax.plot(g["level"], g["knn_top1"], marker="o", label=m)
        ax.set_title(k)
        ax.set_xlabel("level")
        ax.set_ylabel("k-NN top-1 (%)")
    axes[0][0].legend(fontsize=7)
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


def retrieval_grid(title, f_tr, f_ev, X_tr, y_tr, X_ev, y_ev, path, n_queries=6, n_nn=5):
    """Query image + its nearest train neighbours. Green frame = same class, red = different class."""
    q = np.linspace(0, len(X_ev) - 1, n_queries).astype(int)
    nn_idx = (F.normalize(f_ev[q], dim=1) @ F.normalize(f_tr, dim=1).T).topk(n_nn, dim=1).indices
    fig, axes = plt.subplots(n_queries, n_nn + 1, figsize=(1.6 * (n_nn + 1), 1.6 * n_queries), squeeze=False)
    for r, qi in enumerate(q):
        axes[r, 0].imshow(X_ev[qi].permute(1, 2, 0).numpy())
        axes[r, 0].set_title("query", fontsize=7)
        for c, j in enumerate(nn_idx[r].tolist()):
            ax = axes[r, c + 1]
            ax.imshow(X_tr[j].permute(1, 2, 0).numpy())
            for sp in ax.spines.values():
                sp.set_edgecolor("green" if y_tr[j] == y_ev[qi] else "red")
                sp.set_linewidth(3)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=9)
    _save(fig, path)
