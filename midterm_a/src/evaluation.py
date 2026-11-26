"""Frozen-feature evaluation: feature extraction, k-NN, linear probe, per-class accuracy."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"


@torch.no_grad()
def extract_features(encoder, X, transform=None, batch_size=512):
    """h = encoder(x) for every image (uint8 NCHW tensor). Encoder runs in eval() mode (BatchNorm uses running stats)."""
    was_training = encoder.training
    encoder = encoder.to(DEVICE).eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(DEVICE).float() / 255
        if transform is not None:
            xb = transform(xb)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            out.append(encoder(xb.contiguous(memory_format=torch.channels_last)).float().cpu())
    encoder.train(was_training)
    return torch.cat(out)


def correct_top1(scores, y):
    """Per-sample top-1 correctness (bool, CPU). Used for bootstrap confidence intervals."""
    return (scores.argmax(1).cpu() == y.cpu())


def bootstrap_ci(correct, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI (in %) of the mean of a 0/1 vector: resample the evaluation images."""
    c = torch.as_tensor(correct).float()
    g = torch.Generator().manual_seed(seed)
    means = []
    for i in range(0, n_boot, 200):
        idx = torch.randint(0, len(c), (min(200, n_boot - i), len(c)), generator=g)
        means.append(c[idx].mean(1))
    m = torch.cat(means) * 100
    return float(m.quantile(alpha / 2)), float(m.quantile(1 - alpha / 2))


def paired_bootstrap(correct_a, correct_b, n_boot=1000, seed=0, alpha=0.05):
    """Paired bootstrap of acc(a) - acc(b) on the SAME evaluation images. Returns diff, CI and
    p = fraction of bootstrap samples with diff <= 0 (one-sided evidence that a is better than b)."""
    a, b = torch.as_tensor(correct_a).float(), torch.as_tensor(correct_b).float()
    assert a.shape == b.shape, "paired bootstrap needs predictions on the same images"
    d = a - b
    g = torch.Generator().manual_seed(seed)
    diffs = []
    for i in range(0, n_boot, 200):
        idx = torch.randint(0, len(d), (min(200, n_boot - i), len(d)), generator=g)
        diffs.append(d[idx].mean(1))
    m = torch.cat(diffs) * 100
    return {"diff": float(d.mean() * 100), "ci_lo": float(m.quantile(alpha / 2)),
            "ci_hi": float(m.quantile(1 - alpha / 2)), "p_le_0": float((m <= 0).float().mean())}


def topk_acc(scores, y, ks=(1, 5)):
    """Top-k accuracy in percent. scores: (N, C)."""
    kmax = min(max(ks), scores.size(1))
    top = scores.topk(kmax, dim=1).indices.cpu()
    return {f"top{k}": (top[:, :k] == y.cpu().unsqueeze(1)).any(1).float().mean().item() * 100 for k in ks}


@torch.no_grad()
def knn_scores(train_f, train_y, query_f, num_classes, k=20, T=0.1, chunk=1000, weight="exp"):
    """Cosine k-NN on l2-normalised features with a similarity-weighted vote. Returns vote scores (Nq, C).
    weight='exp': w = exp(sim/T) (InstDisc/DINO-style, default);  weight='sim': w = sim (plain cosine weight)."""
    if weight not in ("exp", "sim"):
        raise ValueError(f"knn weight must be 'exp' or 'sim', got {weight!r}")
    bank = F.normalize(train_f, dim=1).to(DEVICE)
    bank_y = train_y.to(DEVICE)
    k = min(k, len(bank))
    out = []
    for i in range(0, len(query_f), chunk):
        q = F.normalize(query_f[i:i + chunk], dim=1).to(DEVICE)
        sim, idx = (q @ bank.T).topk(k, dim=1)
        votes = torch.zeros(q.shape[0], num_classes, device=DEVICE)
        w = (sim / T).exp() if weight == "exp" else sim.clamp_min(0)
        votes.scatter_add_(1, bank_y[idx], w)
        out.append(votes.cpu())
    return torch.cat(out)


def fit_linear_probe(train_f, train_y, num_classes, epochs=100, lr=1e-3, wd=1e-4, bs=512, seed=0):
    """Fit a linear probe using TRAIN features only and return a CPU state bundle.

    The caller evaluates the returned probe on validation data to select lr. Test
    features/labels must not be passed to this function.
    """
    torch.manual_seed(seed)
    mu = train_f.mean(0, keepdim=True)
    sd = train_f.std(0, keepdim=True) + 1e-6  # statistics from train only
    tr = ((train_f - mu) / sd).to(DEVICE)
    ytr = train_y.to(DEVICE)
    clf = nn.Linear(tr.shape[1], num_classes).to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((len(tr) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    clf.train()
    for _ in range(epochs):
        perm = torch.randperm(len(tr), device=DEVICE)
        for i in range(0, len(tr), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(clf(tr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    # Keep only a small CPU copy so each candidate can be discarded safely.
    bundle = {
        "weight": clf.weight.detach().cpu().clone(),
        "bias": clf.bias.detach().cpu().clone(),
        "mu": mu.detach().cpu().clone(),
        "sd": sd.detach().cpu().clone(),
    }
    del clf, opt, sched, tr, ytr
    return bundle


@torch.no_grad()
def evaluate_linear_probe(bundle, features, labels, bs=2048):
    """Evaluate a previously selected linear probe on one split."""
    clf = nn.Linear(bundle["weight"].shape[1], bundle["weight"].shape[0]).to(DEVICE)
    clf.weight.copy_(bundle["weight"].to(DEVICE))
    clf.bias.copy_(bundle["bias"].to(DEVICE))
    clf.eval()
    mu, sd = bundle["mu"].to(DEVICE), bundle["sd"].to(DEVICE)
    outputs = []
    for i in range(0, len(features), bs):
        xb = ((features[i:i + bs] - mu) / sd).to(DEVICE)
        outputs.append(clf(xb).float().cpu())
    scores = torch.cat(outputs)
    result = topk_acc(scores, labels)
    result["correct"] = correct_top1(scores, labels)
    del clf, outputs, scores
    return result


def linear_probe_multi(train_f, train_y, evals, num_classes, epochs=100, lr=1e-3, wd=1e-4, bs=512, seed=0):
    """Compatibility helper: fit on train_f and evaluate supplied splits.

    For final-test protocols, use fit_linear_probe/evaluate_linear_probe so lr
    selection is completed on validation before the test split is evaluated.
    """
    bundle = fit_linear_probe(train_f, train_y, num_classes, epochs, lr, wd, bs, seed)
    return {name: evaluate_linear_probe(bundle, f, y) for name, (f, y) in evals.items()}


def per_class_acc(scores, y, num_classes):
    pred = scores.argmax(1).cpu()
    y = y.cpu()
    return np.array([(pred[y == c] == c).float().mean().item() * 100 if (y == c).any() else np.nan
                     for c in range(num_classes)])
