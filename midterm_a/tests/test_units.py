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


def test_final_test_configs_extend_validation_configs():
    from pathlib import Path
    from src.common import load_config

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    pairs = [
        ("a4_eval.yaml", "a4_eval_test.yaml", "results/final_test/a4_eval"),
        ("a5_finetune.yaml", "a5_finetune_test.yaml", "results/final_test/a5_finetune"),
        ("a6_robustness.yaml", "a6_robustness_test.yaml", "results/final_test/a6_robustness"),
        ("a7_interpret.yaml", "a7_interpret_test.yaml", "results/final_test/a7_interpret"),
    ]
    for base_name, test_name, out_dir in pairs:
        base = load_config(config_dir / base_name)
        test = load_config(config_dir / test_name)
        assert test["split"] == "test"
        assert test["out_dir"] == out_dir
        for key in base:
            if key not in {"split", "out_dir"}:
                assert test[key] == base[key], (test_name, key)


def test_run_protocol_selects_probe_on_validation_before_test(monkeypatch):
    from pathlib import Path
    import src.tasks as tasks

    events = []
    bundles = {0.001: {"lr": 0.001}, 0.003: {"lr": 0.003}}

    def fake_fit(train_f, train_y, num_classes, **kwargs):
        lr = kwargs["lr"]
        events.append(("fit", lr))
        return bundles[lr]

    def fake_eval(bundle, features, labels):
        # First two calls are validation scores. The better LR is 0.003.
        if labels is ds["y_val"]:
            events.append(("val", bundle["lr"]))
            return {"top1": 70.0 if bundle["lr"] == 0.003 else 60.0, "top5": 90.0}
        events.append(("test", bundle["lr"]))
        return {"top1": 55.0, "top5": 85.0}

    def fake_feats(encoder, dataset, splits, cache_file):
        events.append(("extract", tuple(splits)))
        return {"test": torch.zeros(2, 4)}

    def fake_knn(*args, **kwargs):
        return torch.zeros(2, 2)

    def fake_topk(scores, labels):
        return {"top1": 50.0, "top5": 80.0}

    monkeypatch.setattr(tasks, "fit_linear_probe", fake_fit)
    monkeypatch.setattr(tasks, "evaluate_linear_probe", fake_eval)
    monkeypatch.setattr(tasks, "_feats", fake_feats)
    monkeypatch.setattr(tasks, "knn_scores", fake_knn)
    monkeypatch.setattr(tasks, "topk_acc", fake_topk)
    monkeypatch.setattr(tasks, "_free", lambda: None)

    ds = {
        "num_classes": 2,
        "y_train": torch.tensor([0, 1]),
        "y_val": torch.tensor([0, 1]),
        "y_test": torch.tensor([0, 1]),
    }
    feats = {"train": torch.zeros(2, 4), "val": torch.zeros(2, 4)}
    result = tasks.run_protocol(
        feats, ds, "test", {"probe": {"lr_grid": [0.001, 0.003]}},
        encoder=object(), cache_file=Path("unused.pt"),
    )

    assert result["probe_lr"] == 0.003
    test_index = next(i for i, event in enumerate(events) if event[0] == "test")
    extract_test_index = next(i for i, event in enumerate(events) if event == ("extract", ("test",)))
    assert test_index > extract_test_index
    assert sum(event[0] == "test" for event in events) == 1
    assert events.index(("val", 0.003)) < extract_test_index


def test_finetune_selects_lr_on_validation_then_evaluates_test_once(monkeypatch, tmp_path):
    import src.tasks as tasks

    events = []

    class FakeModel:
        def __init__(self, lr):
            self.lr = float(lr)

        def state_dict(self):
            return {"lr": torch.tensor(self.lr)}

        def load_state_dict(self, state):
            self.lr = float(state["lr"].item())

    class FakeClassifier(FakeModel):
        def __init__(self, encoder, num_classes, lr, wd):
            super().__init__(lr)

    x_train = torch.zeros(4, 3, 8, 8, dtype=torch.uint8)
    y_train = torch.tensor([0, 0, 1, 1])
    x_val = torch.zeros(2, 3, 8, 8, dtype=torch.uint8)
    y_val = torch.tensor([0, 1])
    x_test = torch.ones(2, 3, 8, 8, dtype=torch.uint8)
    y_test = torch.tensor([0, 1])
    ds = {
        "name": "tiny", "num_classes": 2,
        "X_train": x_train, "y_train": y_train,
        "X_val": x_val, "y_val": y_val,
        "X_test": x_test, "y_test": y_test,
    }

    monkeypatch.setattr(tasks, "get_dataset", lambda dn, cfg: ds)
    monkeypatch.setattr(tasks, "label_subsets", lambda y, n, f: {f[0]: torch.arange(len(y))})
    monkeypatch.setattr(tasks, "load_encoder", lambda spec: object())
    monkeypatch.setattr(tasks, "Classifier", FakeClassifier)
    monkeypatch.setattr(tasks, "_free", lambda: None)
    monkeypatch.setattr(tasks.plots, "label_efficiency", lambda df, path: None)

    def fake_finetune_one(enc, X, y, ncls, steps, lr, wd, bs, seed):
        events.append(("fit", float(lr)))
        return FakeModel(lr)

    def fake_eval(model, X, y):
        if X is x_val:
            events.append(("val", model.lr))
            score = 80.0 if model.lr == 0.2 else 70.0
        else:
            assert X is x_test
            events.append(("test", model.lr))
            score = 66.0
        return {"top1": score, "top5": score + 10.0}

    monkeypatch.setattr(tasks, "_fine_tune_one", fake_finetune_one)
    monkeypatch.setattr(tasks, "_eval_clf", fake_eval)

    cfg = {
        "out_dir": str(tmp_path / "final_test"),
        "split": "test",
        "datasets": ["tiny"],
        "fractions": [0.5],
        "ft": {"steps": 10, "lr_grid": [0.1, 0.2], "wd": 0.0005, "batch_size": 2},
        "models": [{"name": "dae", "arch": "dae", "path": "unused.ckpt"}],
        "seed": 0,
    }

    result = tasks.finetune(cfg)
    assert len(result) == 1
    assert float(result.iloc[0]["lr"]) == 0.2
    assert float(result.iloc[0]["top1"]) == 66.0
    test_events = [e for e in events if e[0] == "test"]
    assert len(test_events) == 1
    assert abs(test_events[0][1] - 0.2) < 1e-6
    val_idx = next(i for i, e in enumerate(events) if e[0] == "val" and abs(e[1] - 0.2) < 1e-6)
    test_idx = next(i for i, e in enumerate(events) if e[0] == "test")
    assert val_idx < test_idx


def test_local_train_entrypoint_exposes_config_smoke_and_overrides():
    from pathlib import Path
    import subprocess
    import sys

    project_dir = Path(__file__).resolve().parents[1]
    entrypoint = project_dir / "train.py"
    result = subprocess.run(
        [sys.executable, str(entrypoint), "--help"],
        cwd=project_dir,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--config" in result.stdout
    assert "--set" in result.stdout
    assert "--smoke" in result.stdout
