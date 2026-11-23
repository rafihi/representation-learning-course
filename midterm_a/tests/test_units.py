"""Fast unit tests (CPU, no downloads):  SSL_FAKE_DATA=1 SSL_TINY_BACKBONE=1 pytest -q"""
import os

import torch

os.environ.setdefault("SSL_FAKE_DATA", "1")

from src.common import apply_overrides, load_config
from src.corruptions import REGISTRY, corrupt
from src.data import NUM_CLASSES, tiny_splits
from src.evaluation import knn_scores, topk_acc
from src.tasks import label_subsets


def test_corruptions_shape_range_and_identity():
    x = torch.rand(4, 3, 64, 64)
    for k in REGISTRY:
        y = corrupt(x, k, 0.3 if k != "blur" else 1.0)
        assert y.shape == x.shape and y.min() >= -1e-4 and y.max() <= 1 + 1e-4
        assert torch.equal(corrupt(x, k, 0), x)                                 # level 0 = no change


def test_corruptions_actually_change_the_image():
    x = torch.rand(4, 3, 64, 64)
    for k in REGISTRY:
        assert (corrupt(x, k, 0.5 if k != "blur" else 2.0) - x).abs().mean() > 1e-3


def test_config_extends_and_scientific_notation(tmp_path):
    (tmp_path / "a.yaml").write_text("x: 1\nlr: 1e-3\nm: {a: 1, b: 2}\n")
    (tmp_path / "b.yaml").write_text(f"extends: {tmp_path / 'a.yaml'}\nm: {{b: 3}}\n")
    cfg = apply_overrides(load_config(tmp_path / "b.yaml"), ["m.c=4", "lr=2e-3"])
    assert cfg == {"x": 1, "lr": 0.002, "m": {"a": 1, "b": 3, "c": 4}}


def test_splits_disjoint_balanced_and_cached():
    d = {"cache_dir": "x", "train_per_class": 5, "val_per_class": 3, "test_per_class": 2}
    s = tiny_splits(d)
    assert len(s["X_train"]) == 5 * NUM_CLASSES and len(s["X_val"]) == 3 * NUM_CLASSES and len(s["X_test"]) == 2 * NUM_CLASSES
    assert torch.bincount(s["y_train"]).unique().tolist() == [5]
    assert tiny_splits(dict(d)) is s


def test_label_subsets_nested_balanced():
    y = torch.arange(10).repeat_interleave(20)
    sub = label_subsets(y, 10, [0.1, 0.5, 1.0])
    assert set(sub[0.1]) <= set(sub[0.5]) <= set(sub[1.0])
    assert torch.bincount(y[sub[0.5]]).unique().tolist() == [10]


def test_knn_perfect_on_separable_features():
    protos = torch.eye(5).repeat_interleave(10, 0)                              # 5 classes, one-hot "features"
    y = torch.arange(5).repeat_interleave(10)
    acc = topk_acc(knn_scores(protos, y, protos + 0.01 * torch.randn_like(protos), 5, k=5), y)
    assert acc["top1"] == 100.0
