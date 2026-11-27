"""Fast unit tests (CPU, no downloads):  SSL_FAKE_DATA=1 SSL_TINY_BACKBONE=1 pytest -q"""
import os

import torch

os.environ.setdefault("SSL_FAKE_DATA", "1")

from src.common import apply_overrides, load_config
from src.corruptions import REGISTRY, corrupt
from src.data import NUM_CLASSES, tiny_splits
from src.evaluation import knn_scores, topk_acc
from src.tasks import label_subsets


LEVELS = {"blur": 1.0, "patch_shuffle": 4}


def test_corruptions_shape_range_and_identity():
    x = torch.rand(4, 3, 64, 64)
    for k in REGISTRY:
        y = corrupt(x, k, LEVELS.get(k, 0.3))
        assert y.shape == x.shape and y.min() >= -1e-4 and y.max() <= 1 + 1e-4
        assert torch.equal(corrupt(x, k, 0), x)                                 # level 0 = no change


def test_corruptions_actually_change_the_image():
    x = torch.rand(4, 3, 64, 64)
    for k in REGISTRY:
        assert (corrupt(x, k, {"blur": 2.0, "patch_shuffle": 8}.get(k, 0.5)) - x).abs().mean() > 1e-3


def test_patch_shuffle_only_permutes_patches():
    x = torch.rand(3, 3, 64, 64)
    y = corrupt(x, "patch_shuffle", 4)
    p = lambda t: t.reshape(3, 3, 4, 16, 4, 16).permute(0, 2, 4, 1, 3, 5).reshape(3, 16, -1)
    for b in range(3):                                                          # same multiset of patches per image
        a = p(x)[b][p(x)[b][:, 0].argsort()]
        c = p(y)[b][p(y)[b][:, 0].argsort()]
        assert torch.equal(a, c)


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
            if key not in {"split", "out_dir", "reuse_selected_from"}:
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


def test_finetune_selects_on_validation_and_test_reuses_selected_model(monkeypatch, tmp_path):
    import src.tasks as tasks

    events = []

    class FakeModel:
        def __init__(self, lr):
            self.lr = float(lr)

        def cpu(self):
            return self

        def state_dict(self):
            return {"lr": torch.tensor(self.lr)}

        def load_state_dict(self, state):
            self.lr = float(state["lr"].item())

    x_train = torch.zeros(4, 3, 8, 8, dtype=torch.uint8)
    y_train = torch.tensor([0, 0, 1, 1])
    x_val = torch.zeros(2, 3, 8, 8, dtype=torch.uint8)
    y_val = torch.tensor([0, 1])
    x_test = torch.ones(2, 3, 8, 8, dtype=torch.uint8)
    y_test = torch.tensor([0, 1])
    ds = {"name": "tiny", "num_classes": 2, "X_train": x_train, "y_train": y_train,
          "X_val": x_val, "y_val": y_val, "X_test": x_test, "y_test": y_test}

    monkeypatch.setattr(tasks, "get_dataset", lambda dn, cfg: ds)
    monkeypatch.setattr(tasks, "load_encoder", lambda spec: object())
    monkeypatch.setattr(tasks, "encoder_fingerprint", lambda spec: "fp0")
    monkeypatch.setattr(tasks, "_make_clf", lambda enc0, ncls, lr, ft, steps: FakeModel(lr))
    monkeypatch.setattr(tasks, "_free", lambda: None)
    monkeypatch.setattr(tasks.plots, "label_efficiency", lambda df, path: None)

    def fake_finetune_one(enc0, X, y, Xv, yv, ncls, steps, eval_every, lr, ft, seed):
        events.append(("fit", float(lr)))
        return FakeModel(lr), 7

    def fake_eval(model, X, y):
        if X is x_val:
            events.append(("val", model.lr))
            score = 80.0 if model.lr == 0.2 else 70.0
        else:
            assert X is x_test
            events.append(("test", model.lr))
            score = 66.0
        return {"top1": score, "top5": score + 10.0, "correct": torch.tensor([True, False])}

    monkeypatch.setattr(tasks, "_fine_tune_one", fake_finetune_one)
    monkeypatch.setattr(tasks, "_eval_clf", fake_eval)

    base = {"datasets": ["tiny"], "fractions": [1.0], "seed": 0, "n_boot": 50,
            "ft": {"epochs": 1, "min_steps": 2, "max_steps": 4, "lr_grid": [0.1, 0.2], "wd": 0.0005, "batch_size": 2},
            "models": [{"name": "dae", "arch": "dae", "path": "unused.ckpt"}]}
    val_dir = tmp_path / "val"
    res_val = tasks.finetune({**base, "out_dir": str(val_dir), "split": "val"})
    assert float(res_val.iloc[0]["lr"]) == 0.2 and float(res_val.iloc[0]["top1"]) == 80.0
    assert not any(e[0] == "test" for e in events)                             # validation run never touches test
    assert (val_dir / "selected" / "tiny__dae__1.0.pt").exists()

    events.clear()
    res_test = tasks.finetune({**base, "out_dir": str(tmp_path / "test"), "split": "test",
                               "reuse_selected_from": str(val_dir)})
    assert not any(e[0] == "fit" for e in events)                              # no re-training for the test numbers
    assert len(events) == 1 and events[0][0] == "test" and abs(events[0][1] - 0.2) < 1e-6   # val-selected, exactly once
    assert float(res_test.iloc[0]["top1"]) == 66.0

    events.clear()
    tasks.finetune({**base, "out_dir": str(val_dir), "split": "val"})          # resume: nothing recomputed
    assert events == []


def test_label_subsets_sizes_round_half_up():
    y = torch.arange(200).repeat_interleave(450)
    sub = tasks_mod().label_subsets(y, 200, [0.01, 0.1, 1.0])
    assert [len(sub[f]) for f in (0.01, 0.1, 1.0)] == [1000, 9000, 90000]


def test_ft_schedule_scales_with_labels_and_is_clipped():
    ft = {"epochs": 30, "min_steps": 500, "max_steps": 5000, "n_evals": 10, "batch_size": 128}
    ts = tasks_mod().ft_schedule
    assert ts(1000, ft) == (500, 50)
    steps, every = ts(9000, ft)
    assert 2100 <= steps <= 2200 and steps % every == 0
    assert ts(90000, ft) == (5000, 500)
    assert ts(90000, {**ft, "steps": 2000}) == (2000, 200)                       # legacy fixed budget still works


def test_random_crop_flip_keeps_shape():
    from src.classifier import random_crop_flip
    x = torch.rand(5, 3, 64, 64)
    assert random_crop_flip(x).shape == x.shape


def test_pretrain_refuses_to_reuse_a_run_with_other_settings(tmp_path):
    import pytest
    from src.common import save_json
    tasks = tasks_mod()
    cfg = {"out_dir": str(tmp_path / "run"), "seed": 0, "model": {"name": "dae", "kind": "gaussian", "level": 0.3},
           "data": {"train_per_class": 5, "val_per_class": 2, "batch_size": 8}, "trainer": {"max_epochs": 1}}
    save_json({"cfg": cfg, "metrics": {"knn_top1": 1.0}}, tmp_path / "run" / "metrics.json")
    assert tasks.pretrain(cfg) == {"knn_top1": 1.0}                            # same settings -> skipped, reused
    other = {**cfg, "model": {**cfg["model"], "level": 0.5}}
    with pytest.raises(RuntimeError):
        tasks.pretrain(other)


def test_paired_bootstrap_and_ci():
    from src.evaluation import bootstrap_ci, paired_bootstrap
    a = torch.tensor([True] * 80 + [False] * 20)
    b = torch.tensor([True] * 50 + [False] * 50)
    lo, hi = bootstrap_ci(a, 500)
    assert lo < 80 < hi
    r = paired_bootstrap(a, b, 500)
    assert abs(r["diff"] - 30) < 1e-4 and r["ci_lo"] > 0 and r["p_le_0"] == 0


def test_feature_cache_name_changes_when_checkpoint_changes(tmp_path):
    import os
    from src.registry import ckpt_fingerprint
    p = tmp_path / "last.ckpt"
    p.write_bytes(b"a")
    f1 = ckpt_fingerprint(p)
    os.utime(p, ns=(1, 1))
    assert ckpt_fingerprint(p) != f1


def test_layer_features_of_real_resnet(monkeypatch):
    monkeypatch.delenv("SSL_TINY_BACKBONE", raising=False)
    from src.backbone import make_encoder
    from src.evaluation import extract_features, extract_layer_features
    enc = make_encoder()
    X = torch.randint(0, 255, (3, 3, 64, 64), dtype=torch.uint8)
    f = extract_layer_features(enc, X, ["layer2", "layer3", "layer4", "no_such_layer"])
    assert {k: v.shape[1] for k, v in f.items()} == {"layer2": 128, "layer3": 256, "layer4": 512}
    assert torch.allclose(f["layer4"], extract_features(enc, X), rtol=1e-2, atol=1e-2)   # layer4 pooled = h (fp16 on GPU)


def test_from_sweep_takes_best_validation_knn_trial(tmp_path):
    import pandas as pd
    tasks = tasks_mod()
    csv = tmp_path / "results.csv"
    pd.DataFrame([{"trial": 0, "model.kind": "mask", "model.level": 0.2, "model.lr": 1e-3, "knn_top1": 10.0},
                  {"trial": 1, "model.kind": "gaussian", "model.level": 0.4, "model.lr": 5e-4, "knn_top1": 12.0}]).to_csv(csv, index=False)
    cfg = {"model": {"kind": "x", "level": 0.0, "lr": 1.0},
           "from_sweep": {"csv": str(csv), "keys": ["model.kind", "model.level", "model.lr"]}}
    assert tasks.apply_from_sweep(cfg)["model"] == {"kind": "gaussian", "level": 0.4, "lr": 5e-4}
    vanilla = {"model": {"kind": "none", "level": 0.0, "lr": 1.0}, "from_sweep": {"csv": str(csv), "keys": ["model.lr"]}}
    assert tasks.apply_from_sweep(vanilla)["model"] == {"kind": "none", "level": 0.0, "lr": 5e-4}


def tasks_mod():
    import src.tasks as tasks
    return tasks


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
