"""Data: Tiny-ImageNet (pre-training + evaluation), transfer datasets, and the LightningDataModule.

All images are kept in memory as uint8 tensors (N,3,64,64). Augmentation/corruption happen on the GPU.
Set the environment variable SSL_FAKE_DATA=1 to use small synthetic data (no download) for smoke tests.
"""
import os
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

SPLIT_SEED = 0          # the val/test split never depends on the run seed -> identical for every model
VAL_HOLDOUT = 50        # images per class held out of `train` (the pool for validation)
NUM_CLASSES = 200
_CACHE = {}


def fake_mode() -> bool:
    return bool(os.environ.get("SSL_FAKE_DATA"))


def _fake_arrays(n_per_class, n_classes, seed, size=64):
    """Synthetic images: each class has a color prototype + low-res pattern + noise. Learnable but tiny."""
    rng = np.random.RandomState(seed)
    protos = np.random.RandomState(123).randint(40, 215, size=(n_classes, 3)).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_per_class)
    X = np.empty((len(y), size, size, 3), np.uint8)
    for i in range(0, len(y), 500):
        yy = y[i:i + 500]
        low = rng.randn(len(yy), 8, 8, 3).astype(np.float32) * 30
        low = np.kron(low, np.ones((1, size // 8, size // 8, 1), np.float32))
        noise = rng.randn(len(yy), size, size, 3).astype(np.float32) * 15
        X[i:i + 500] = np.clip(protos[yy][:, None, None, :] + low + noise, 0, 255).astype(np.uint8)
    return X, y.astype(np.int64)


def load_tiny_arrays(cache_dir):
    """Returns dict X_train,y_train (100k) and X_valid,y_valid (10k) as uint8 NHWC numpy arrays. Cached on disk."""
    if fake_mode():
        Xtr, ytr = _fake_arrays(VAL_HOLDOUT + 10, NUM_CLASSES, 1)
        Xva, yva = _fake_arrays(10, NUM_CLASSES, 2)
        return {"X_train": Xtr, "y_train": ytr, "X_valid": Xva, "y_valid": yva}
    path = Path(cache_dir) / "tiny_imagenet_uint8.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    from datasets import load_dataset
    ds = load_dataset("zh-plus/tiny-imagenet")
    out = {}
    for split, key in [("train", "train"), ("valid", "valid")]:
        d = ds[split]
        X = np.empty((len(d), 64, 64, 3), np.uint8)
        y = np.empty(len(d), np.int64)
        for i, ex in enumerate(d):
            X[i] = np.array(ex["image"].convert("RGB"))      # a few images are grayscale -> force RGB
            y[i] = ex["label"]
        out[f"X_{key}"], out[f"y_{key}"] = X, y
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def _to_nchw(X):
    return torch.from_numpy(np.ascontiguousarray(X)).permute(0, 3, 1, 2).contiguous()


def tiny_splits(dcfg):
    """Train/val/test tensors. val = first `val_per_class` of the 50 held-out images per class;
    train = images from the remaining 450 per class (prefix of a fixed permutation -> nested subsets);
    test = random `test_per_class` per class of the official valid split."""
    key = ("tiny", str(dcfg.get("cache_dir")), dcfg["train_per_class"], dcfg["val_per_class"],
           dcfg["test_per_class"], fake_mode())
    if key in _CACHE:
        return _CACHE[key]
    a = load_tiny_arrays(dcfg.get("cache_dir", "data_cache"))
    rng = np.random.RandomState(SPLIT_SEED)
    tr, va = [], []
    for c in range(NUM_CLASSES):
        idx = rng.permutation(np.where(a["y_train"] == c)[0])
        va.extend(idx[:min(dcfg["val_per_class"], VAL_HOLDOUT)].tolist())
        tr.extend(idx[VAL_HOLDOUT:VAL_HOLDOUT + dcfg["train_per_class"]].tolist())
    rng2 = np.random.RandomState(SPLIT_SEED + 1)
    te = []
    for c in range(NUM_CLASSES):
        idx = rng2.permutation(np.where(a["y_valid"] == c)[0])
        te.extend(idx[:dcfg["test_per_class"]].tolist())
    assert not set(tr) & set(va), "train/val leakage"
    out = {
        "name": "tiny", "num_classes": NUM_CLASSES,
        "X_train": _to_nchw(a["X_train"][tr]), "y_train": torch.from_numpy(a["y_train"][tr]),
        "X_val": _to_nchw(a["X_train"][va]), "y_val": torch.from_numpy(a["y_train"][va]),
        "X_test": _to_nchw(a["X_valid"][te]), "y_test": torch.from_numpy(a["y_valid"][te]),
    }
    _CACHE[key] = out
    return out


# ----------------------------------------------------------------------------- transfer datasets
def take_per_class(y, n, seed):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    out = []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        out.extend(idx[:n].tolist() if n else idx.tolist())
    return np.array(out)


def _resize64(x):
    """uint8 NCHW tensor of any size -> uint8 NCHW 64x64 (bilinear, antialiased), done in chunks."""
    if x.shape[-1] == 64 and x.shape[-2] == 64:
        return x
    outs = []
    for i in range(0, len(x), 2000):
        outs.append(F.interpolate(x[i:i + 2000].float(), size=(64, 64), mode="bilinear",
                                  align_corners=False, antialias=True).round().clamp(0, 255).to(torch.uint8))
    return torch.cat(outs)


def _raw_transfer(name, root):
    """Returns (Xtr, ytr, Xte, yte, num_classes); X are uint8 NCHW tensors of any resolution."""
    import torchvision
    if fake_mode():
        Xtr, ytr = _fake_arrays(40, 10, 5)
        Xte, yte = _fake_arrays(15, 10, 6)
        return _to_nchw(Xtr), torch.from_numpy(ytr), _to_nchw(Xte), torch.from_numpy(yte), 10
    if name == "cifar100":
        tr = torchvision.datasets.CIFAR100(root, train=True, download=True)
        te = torchvision.datasets.CIFAR100(root, train=False, download=True)
        return (_to_nchw(tr.data), torch.tensor(tr.targets), _to_nchw(te.data), torch.tensor(te.targets), 100)
    if name == "stl10":
        tr = torchvision.datasets.STL10(root, split="train", download=True)
        te = torchvision.datasets.STL10(root, split="test", download=True)
        return (torch.from_numpy(tr.data), torch.from_numpy(tr.labels).long(),
                torch.from_numpy(te.data), torch.from_numpy(te.labels).long(), 10)
    if name == "pets":
        def load(split):
            ds = torchvision.datasets.OxfordIIITPet(root, split=split, target_types="category", download=True)
            xs, ys = [], []
            for im, lab in ds:
                xs.append(np.array(im.convert("RGB").resize((64, 64))))
                ys.append(lab)
            return _to_nchw(np.stack(xs)), torch.tensor(ys)
        Xtr, ytr = load("trainval")
        Xte, yte = load("test")
        return Xtr, ytr, Xte, yte, 37
    raise ValueError(f"unknown transfer dataset '{name}'. Add a branch in _raw_transfer().")


def load_transfer(name, cache_dir, train_per_class=None, test_per_class=None, val_frac=0.1):
    """Transfer dataset resized to 64x64 with a held-out validation split carved from its train set."""
    key = ("transfer", name, str(cache_dir), train_per_class, test_per_class, fake_mode())
    if key in _CACHE:
        return _CACHE[key]
    Xtr, ytr, Xte, yte, ncls = _raw_transfer(name, str(Path(cache_dir) / "torchvision"))
    tr_idx = take_per_class(ytr.numpy(), train_per_class, SPLIT_SEED)
    te_idx = take_per_class(yte.numpy(), test_per_class, SPLIT_SEED + 1)
    Xtr, ytr, Xte, yte = Xtr[tr_idx], ytr[tr_idx], Xte[te_idx], yte[te_idx]
    rng = np.random.RandomState(SPLIT_SEED + 2)
    va, tr = [], []
    for c in range(ncls):
        idx = rng.permutation(np.where(ytr.numpy() == c)[0])
        n_val = max(1, int(round(val_frac * len(idx))))
        va.extend(idx[:n_val].tolist())
        tr.extend(idx[n_val:].tolist())
    out = {"name": name, "num_classes": ncls,
           "X_train": _resize64(Xtr[tr]), "y_train": ytr[tr],
           "X_val": _resize64(Xtr[va]), "y_val": ytr[va],
           "X_test": _resize64(Xte), "y_test": yte}
    _CACHE[key] = out
    return out


def get_dataset(name, cfg):
    """'tiny' -> Tiny-ImageNet splits; anything else -> transfer dataset (cfg['transfer'] holds its settings)."""
    if name == "tiny":
        return tiny_splits(cfg["data"])
    t = cfg.get("transfer", {})
    return load_transfer(t.get("name", name), cfg["data"].get("cache_dir", "data_cache"),
                         t.get("train_per_class"), t.get("test_per_class"))


# ----------------------------------------------------------------------------- loaders / datamodule
class _TensorBatches(Dataset):
    """Indexed with a LIST of indices (via BatchSampler) -> returns a whole batch at once (fast, no collate)."""

    def __init__(self, *tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.tensors)


def make_loader(X, y, batch_size, shuffle, drop_last):
    ds = _TensorBatches(X, y)
    base = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    return DataLoader(ds, sampler=BatchSampler(base, min(batch_size, len(ds)), drop_last), batch_size=None, num_workers=0)


class TinyImageNetDM(L.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.cfg = data_cfg
        self.s = None

    def setup(self, stage=None):
        self.s = tiny_splits(self.cfg)

    def train_dataloader(self):
        return make_loader(self.s["X_train"], self.s["y_train"], self.cfg["batch_size"], True, True)

    def val_dataloader(self):
        return make_loader(self.s["X_val"], self.s["y_val"], 512, False, False)
