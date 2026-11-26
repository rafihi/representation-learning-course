"""All runnable tasks. The course `lec3/train.py --config X.yaml` entry point reads
`task:` from the Midterm config and dispatches to the function below.

pretrain    train one DAE / vanilla-AE model (checkpoint + TensorBoard/W&B logs + val k-NN metrics.json)
sweep       many pretrain runs (grid | random | trials), resumable, writes results.csv + a figure
evaluate    frozen-encoder protocol (k-NN + linear probe) on Tiny-ImageNet and a transfer dataset
finetune    fine-tune all weights with 1/10/100 % of the labels (also the supervised-from-scratch baseline)
robustness  k-NN accuracy under corruptions of the evaluation images
interpret   per-class accuracy, t-SNE, nearest-neighbour retrieval grid
"""
import copy
import gc
import itertools
import math
import random
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from . import plots
from .callbacks import KNNMonitor
from .classifier import Classifier
from .common import apply_smoke, deep_update, load_config, load_json, save_json, set_by_path
from .corruptions import corrupt
from .data import TinyImageNetDM, get_dataset, make_loader
from .evaluation import (DEVICE, evaluate_linear_probe, extract_features, fit_linear_probe,
                         knn_scores, per_class_acc, topk_acc, USE_AMP)
from .registry import build_module, load_encoder


# ----------------------------------------------------------------------------- helpers
def _knn_kw(cfg):
    k = cfg.get("knn", {})
    return {"k": k.get("k", 20), "T": k.get("T", 0.1)}


def _precision(cfg):
    return cfg.get("trainer", {}).get("precision", "16-mixed") if torch.cuda.is_available() else 32


def _logger(cfg, out):
    kind = cfg.get("logging", {}).get("logger", "tensorboard")
    if kind == "wandb":
        from lightning.pytorch.loggers import WandbLogger
        return WandbLogger(project=cfg["logging"].get("project", "ssl-midterm"), name=Path(out).name, save_dir=str(out))
    if kind == "tensorboard":
        from lightning.pytorch.loggers import TensorBoardLogger
        return TensorBoardLogger(save_dir=str(out), name="tb", version="")
    return False


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- pretrain
def pretrain(cfg):
    out = Path(cfg["out_dir"])
    mpath = out / "metrics.json"
    if cfg.get("skip_if_done", True) and mpath.exists():
        print(f"[skip] {out} already finished")
        return load_json(mpath)["metrics"]
    out.mkdir(parents=True, exist_ok=True)
    L.seed_everything(cfg.get("seed", 0), workers=True)
    dm = TinyImageNetDM(cfg["data"])
    dm.setup()
    s = dm.s
    model = build_module(cfg)
    t = cfg.get("trainer", {})

        # Create checkpoint directory explicitly
    ckpt_dir = out / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cbs = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="last",
            monitor="epoch",
            mode="max",
            save_top_k=1,
            save_last=False,
            every_n_epochs=t.get("ckpt_every", 1),
            save_on_train_epoch_end=True,
            enable_version_counter=False,
            verbose=True
        )
    ]
    if t.get("knn_every", 5) > 0:
        cbs.append(KNNMonitor(s["X_train"], s["y_train"], s["X_val"], s["y_val"], s["num_classes"], t.get("knn_every", 5), **_knn_kw(cfg)))
    trainer = L.Trainer(max_epochs=t["max_epochs"], accelerator="auto", devices=1, precision=_precision(cfg),
                        logger=_logger(cfg, out), callbacks=cbs, log_every_n_steps=20,
                        enable_progress_bar=t.get("progress", True), enable_model_summary=False)
    last = ckpt_dir / "last.ckpt"
    trainer.fit(model, datamodule=dm, ckpt_path=str(last) if cfg.get("resume", True) and last.exists() else None)

    metrics = {}
    if cfg["model"]["name"] == "dae":                                         # validation reconstruction errors
        res = trainer.validate(model, datamodule=dm, verbose=False)[0]
        metrics.update({k: float(v) for k, v in res.items()})
    f_tr, f_va = extract_features(model.encoder, s["X_train"]), extract_features(model.encoder, s["X_val"])
    knn = topk_acc(knn_scores(f_tr, s["y_train"], f_va, s["num_classes"], **_knn_kw(cfg)), s["y_val"])
    metrics.update(knn_top1=knn["top1"], knn_top5=knn["top5"])
    save_json({"cfg": cfg, "metrics": metrics}, mpath)
    print(f"[done] {out.name}: {metrics}")
    return metrics


# ----------------------------------------------------------------------------- sweep
def _sample(spec, rng):
    if "choice" in spec:
        return rng.choice(spec["choice"])
    if "uniform" in spec:
        return rng.uniform(*spec["uniform"])
    if "loguniform" in spec:
        a, b = spec["loguniform"]
        return 10 ** rng.uniform(math.log10(a), math.log10(b))
    raise ValueError(f"bad search-space entry {spec}")


def build_trials(cfg):
    mode = cfg["mode"]
    if mode == "trials":
        trials = [dict(t) for t in cfg["trials"]]
    elif mode == "grid":
        keys = list(cfg["grid"])
        trials = [dict(zip(keys, v)) for v in itertools.product(*[cfg["grid"][k] for k in keys])]
    elif mode == "random":
        rng = random.Random(cfg.get("seed", 0))
        trials = [{k: _sample(v, rng) for k, v in cfg["space"].items()} for _ in range(cfg["n_trials"])]
    else:
        raise ValueError(f"unknown sweep mode '{mode}'")
    return trials[:1] if cfg.get("smoke") else trials


def sweep(cfg):
    base = deep_update(load_config(cfg["base_config"]), cfg.get("base_overrides", {}))
    if cfg.get("smoke"):
        apply_smoke(base)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ov in enumerate(build_trials(cfg)):
        t = copy.deepcopy(base)
        for k, v in ov.items():
            set_by_path(t, k, v)
        t["out_dir"] = str(out / f"trial_{i:02d}")
        t["seed"] = cfg.get("seed", base.get("seed", 0))
        m = pretrain(t)                                                       # resumable: finished trials are skipped
        row = {"trial": i, **{k: (str(v) if isinstance(v, list) else v) for k, v in ov.items()}, **m}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
        _free()
    df = pd.DataFrame(rows)
    kind = cfg.get("plot")
    if kind == "a2":
        print(plots.a2_scatter(df, out / "recon_vs_knn.png"))
    print(df.round(3).to_string())
    return df


# ----------------------------------------------------------------------------- evaluate (frozen protocol)
def _feats(enc, ds, splits, cache_file):
    """Extract/cache only the explicitly requested split(s).

    Separate files are intentional: during final-test evaluation, train/val feature
    caches are loaded first; the Test feature file is not even opened or created
    until validation-only hyperparameter selection has finished.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    result = {}
    for sp in splits:
        split_file = cache_file.with_name(f"{cache_file.stem}__{sp}.pt")
        if split_file.exists():
            result[sp] = torch.load(split_file, map_location="cpu", weights_only=False)
        else:
            result[sp] = extract_features(enc, ds[f"X_{sp}"])
            torch.save(result[sp], split_file)
    return result


def run_protocol(feats, ds, split, cfg, encoder=None, cache_file=None):
    """Frozen protocol with strict validation-only probe selection.

    All lr candidates are fitted on train features and scored on validation only.
    When split='test', test features are extracted and evaluated only after the
    best validation lr has been selected.
    """
    ncls, p = ds["num_classes"], cfg.get("probe", {})
    best = None
    for lr in p.get("lr_grid", [1e-3]):
        bundle = fit_linear_probe(
            feats["train"], ds["y_train"], ncls,
            epochs=2 if cfg.get("smoke") else p.get("epochs", 100),
            lr=lr, wd=p.get("wd", 1e-4), seed=cfg.get("seed", 0),
        )
        val_result = evaluate_linear_probe(bundle, feats["val"], ds["y_val"])
        if best is None or val_result["top1"] > best["val"]["top1"]:
            best = {"bundle": bundle, "val": val_result, "lr": lr}
        del bundle
        _free()

    if best is None:
        raise RuntimeError("No linear-probe learning-rate candidates were evaluated.")

    if split == "test":
        if encoder is None or cache_file is None:
            raise ValueError("Test evaluation requires encoder and feature cache path.")
        # Critical ordering: the test features are not extracted until lr selection is complete.
        test_feats = _feats(encoder, ds, ["test"], cache_file)["test"]
        target_feats, target_labels = test_feats, ds["y_test"]
    elif split == "val":
        target_feats, target_labels = feats["val"], ds["y_val"]
    else:
        raise ValueError(f"Unsupported evaluation split: {split!r}")

    # k-NN has no tuned hyperparameter here; it is evaluated only on the selected target split.
    knn = topk_acc(
        knn_scores(feats["train"], ds["y_train"], target_feats, ncls, **_knn_kw(cfg)),
        target_labels,
    )
    # The selected probe is evaluated on Test exactly once (or reuses its val score).
    lin = best["val"] if split == "val" else evaluate_linear_probe(best["bundle"], target_feats, target_labels)

    return {
        "knn_top1": knn["top1"], "knn_top5": knn["top5"],
        "lin_top1": lin["top1"], "lin_top5": lin["top5"],
        "probe_lr": best["lr"],
    }


def evaluate(cfg):
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    split = cfg.get("split", "val")
    if split == "test":
        print("!!! evaluating on the TEST split: use this only for final numbers !!!")
    rows = []
    for spec in cfg["models"]:
        spec = {"type": "ckpt", **spec}
        enc = load_encoder(spec)
        row = {"model": spec["name"]}
        for dn in cfg.get("datasets", ["tiny", "transfer"]):
            ds = get_dataset(dn, cfg)
            cache = out / "features" / f"{spec['name']}__{ds['name']}__{len(ds['X_train'])}.pt"
            # Never extract or score the Test split until validation-only LR selection is done.
            feats = _feats(enc, ds, ["train", "val"], cache)
            row.update({f"{ds['name']}_{k}": v for k, v in run_protocol(
                feats, ds, split, cfg, encoder=enc, cache_file=cache
            ).items()})
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
        _free()
    df = pd.DataFrame(rows)
    print(df.round(2).to_string())
    return df


# ----------------------------------------------------------------------------- finetune
def label_subsets(y, num_classes, fractions, seed=0):
    """Class-balanced, NESTED subsets (1% subset is inside the 10% subset...), identical for every model."""
    rng = np.random.RandomState(seed)
    y = y.numpy()
    perm = {c: rng.permutation(np.where(y == c)[0]) for c in range(num_classes)}
    return {f: np.concatenate([perm[c][:max(1, round(f * len(perm[c])))] for c in range(num_classes)]) for f in fractions}


@torch.no_grad()
def _eval_clf(model, X, y, bs=512):
    model = model.to(DEVICE).eval()
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            out.append(model(X[i:i + bs].to(DEVICE).float() / 255).float().cpu())
    return topk_acc(torch.cat(out), y)


def _fine_tune_one(enc0, X, y, ncls, steps, lr, wd, bs, seed):
    L.seed_everything(seed, workers=True)
    model = Classifier(copy.deepcopy(enc0), ncls, lr=lr, wd=wd)               # deepcopy: never modify the pretrained encoder
    trainer = L.Trainer(max_steps=steps, max_epochs=-1, accelerator="auto", devices=1, precision=32 if not USE_AMP else "16-mixed",
                        logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                        log_every_n_steps=1)
    trainer.fit(model, make_loader(X, y, bs, shuffle=True, drop_last=len(X) > bs))
    return model


def finetune(cfg):
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "results.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["dataset"], r["model"], float(r["frac"])) for r in rows}
    f = cfg["ft"]
    split, fractions = cfg.get("split", "val"), cfg["fractions"]
    steps = 1 if cfg.get("smoke") else f["steps"]

    for dn in cfg.get("datasets", ["tiny", "transfer"]):
        ds = get_dataset(dn, cfg)
        subsets = label_subsets(ds["y_train"], ds["num_classes"], fractions)
        for spec in cfg["models"]:
            enc0 = None
            for frac in fractions:
                key = (ds["name"], spec["name"], float(frac))
                if key in done:
                    continue
                enc0 = enc0 if enc0 is not None else load_encoder({"type": "ckpt", **spec})
                idx = subsets[frac]
                best = None

                # Train every LR candidate, but inspect VALIDATION only while selecting.
                for lr in f["lr_grid"]:
                    model = _fine_tune_one(
                        enc0, ds["X_train"][idx], ds["y_train"][idx],
                        ds["num_classes"], steps, lr,
                        f.get("wd", 5e-4), f.get("batch_size", 64), cfg.get("seed", 0),
                    )
                    val = _eval_clf(model, ds["X_val"], ds["y_val"])
                    if best is None or val["top1"] > best["val"]["top1"]:
                        best = {
                            "val": val,
                            "lr": lr,
                            "state_dict": {
                                k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()
                            },
                        }
                    del model
                    _free()

                if best is None:
                    raise RuntimeError(f"No fine-tuning LR candidates for {key}.")

                if split == "test":
                    # Restore only the validation-selected candidate, then touch Test once.
                    final_model = Classifier(
                        copy.deepcopy(enc0), ds["num_classes"],
                        lr=best["lr"], wd=f.get("wd", 5e-4),
                    )
                    final_model.load_state_dict(best["state_dict"])
                    target = _eval_clf(final_model, ds["X_test"], ds["y_test"])
                    del final_model
                    _free()
                elif split == "val":
                    target = best["val"]
                else:
                    raise ValueError(f"Unsupported fine-tuning split: {split!r}")

                row = {
                    "dataset": ds["name"], "model": spec["name"],
                    "frac": float(frac), "n_labels": len(idx),
                    "top1": target["top1"], "top5": target["top5"],
                    "val_top1": best["val"]["top1"],
                    "lr": best["lr"], "split": split,
                }
                rows.append(row)
                done.add(key)
                pd.DataFrame(rows).to_csv(csv, index=False)
                print(row, flush=True)

    df = pd.DataFrame(rows)
    plots.label_efficiency(df, out / "label_efficiency.png")
    return df


# ----------------------------------------------------------------------------- robustness
def robustness(cfg):
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    ds = get_dataset("tiny", cfg)
    split = cfg.get("split", "val")
    X_ev, y_ev = ds[f"X_{split}"], ds[f"y_{split}"]
    rows = []
    for spec in cfg["models"]:
        enc = load_encoder({"type": "ckpt", **spec})
        f_tr = extract_features(enc, ds["X_train"])                           # clean memory bank
        for kind, levels in cfg["corruptions"].items():
            for lvl in (levels[:2] if cfg.get("smoke") else levels):
                torch.manual_seed(0)                                          # identical corruption noise for every model
                f_c = extract_features(enc, X_ev, transform=lambda x, k=kind, l=lvl: corrupt(x, k, l))
                acc = topk_acc(knn_scores(f_tr, ds["y_train"], f_c, ds["num_classes"], **_knn_kw(cfg)), y_ev)["top1"]
                rows.append({"model": spec["name"], "kind": kind, "level": lvl, "knn_top1": acc})
        _free()
    df = pd.DataFrame(rows)
    df.to_csv(out / "robustness.csv", index=False)
    plots.robustness(df, out / "robustness.png")
    print(df.round(2).to_string())
    return df


# ----------------------------------------------------------------------------- interpret
def interpret(cfg):
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    ds = get_dataset("tiny", cfg)
    split = cfg.get("split", "val")
    y_ev, X_ev = ds[f"y_{split}"], ds[f"X_{split}"]
    feats, pc = {}, {}
    for spec in cfg["models"]:
        enc = load_encoder({"type": "ckpt", **spec})
        feats[spec["name"]] = {"train": extract_features(enc, ds["X_train"]), "eval": extract_features(enc, X_ev)}
        votes = knn_scores(feats[spec["name"]]["train"], ds["y_train"], feats[spec["name"]]["eval"], ds["num_classes"], **_knn_kw(cfg))
        pc[spec["name"]] = per_class_acc(votes, y_ev, ds["num_classes"])
    pd.DataFrame(pc).rename_axis("class_idx").to_csv(out / "per_class_knn.csv")          # map class_idx -> wnid with words.txt
    plots.tsne_grid({n: f["eval"] for n, f in feats.items()}, y_ev, out / "tsne.png", cfg.get("tsne_classes", 10))
    rm = cfg.get("retrieval_model", cfg["models"][0]["name"])
    plots.retrieval_grid(rm, feats[rm]["train"], feats[rm]["eval"], ds["X_train"], ds["y_train"], X_ev, y_ev, out / f"retrieval_{rm}.png")
    worst = {n: np.argsort(np.nan_to_num(a, nan=1e9))[:10].tolist() for n, a in pc.items()}
    save_json(worst, out / "worst_10_classes.json")
    print("worst classes per model:", worst)


TASKS = {"pretrain": pretrain, "sweep": sweep, "evaluate": evaluate, "finetune": finetune,
         "robustness": robustness, "interpret": interpret}
