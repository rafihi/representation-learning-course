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


def topk_acc(scores, y, ks=(1, 5)):
    """Top-k accuracy in percent. scores: (N, C)."""
    kmax = min(max(ks), scores.size(1))
    top = scores.topk(kmax, dim=1).indices.cpu()
    return {f"top{k}": (top[:, :k] == y.cpu().unsqueeze(1)).any(1).float().mean().item() * 100 for k in ks}


@torch.no_grad()
def knn_scores(train_f, train_y, query_f, num_classes, k=20, T=0.1, chunk=1000):
    """Cosine k-NN with a similarity-weighted vote (weight = exp(sim/T)). Returns vote scores (Nq, C)."""
    bank = F.normalize(train_f, dim=1).to(DEVICE)
    bank_y = train_y.to(DEVICE)
    k = min(k, len(bank))
    out = []
    for i in range(0, len(query_f), chunk):
        q = F.normalize(query_f[i:i + chunk], dim=1).to(DEVICE)
        sim, idx = (q @ bank.T).topk(k, dim=1)
        votes = torch.zeros(q.shape[0], num_classes, device=DEVICE)
        votes.scatter_add_(1, bank_y[idx], (sim / T).exp())
        out.append(votes.cpu())
    return torch.cat(out)


def linear_probe_multi(train_f, train_y, evals, num_classes, epochs=100, lr=1e-3, wd=1e-4, bs=512, seed=0):
    """Train ONE linear classifier on frozen features, return top-1/top-5 for each (name -> (features, labels))."""
    torch.manual_seed(seed)
    mu, sd = train_f.mean(0, keepdim=True), train_f.std(0, keepdim=True) + 1e-6      # statistics from train only
    tr, ytr = ((train_f - mu) / sd).to(DEVICE), train_y.to(DEVICE)
    clf = nn.Linear(tr.shape[1], num_classes).to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((len(tr) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    for _ in range(epochs):
        perm = torch.randperm(len(tr), device=DEVICE)
        for i in range(0, len(tr), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(clf(tr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        return {name: topk_acc(clf(((f - mu) / sd).to(DEVICE)), y) for name, (f, y) in evals.items()}


def per_class_acc(scores, y, num_classes):
    pred = scores.argmax(1).cpu()
    y = y.cpu()
    return np.array([(pred[y == c] == c).float().mean().item() * 100 if (y == c).any() else np.nan
                     for c in range(num_classes)])
