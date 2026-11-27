"""All runnable tasks. The local `midterm_a/train.py --config X.yaml` entry point reads
`task:` from the Midterm config and dispatches to the function below.

pretrain    train one DAE / vanilla-AE model (checkpoint + TensorBoard/W&B logs + val k-NN metrics.json)
sweep       many pretrain runs (grid | random | trials), resumable, writes results.csv + a figure
evaluate    frozen-encoder protocol (k-NN + linear probe) on Tiny-ImageNet and a transfer dataset
finetune    fine-tune all weights with 1/10/100 % of the labels (also the supervised-from-scratch baseline)
robustness  k-NN accuracy under corruptions of the evaluation images
interpret   per-class accuracy, t-SNE, nearest-neighbour retrieval grids
"""
import copy
import gc
import hashlib
import itertools
import json
import math
import random
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint

from . import plots
from .callbacks import KeepBestState, KNNMonitor
from .classifier import Classifier
from .common import apply_smoke, deep_update, load_config, load_json, save_json, set_by_path
from .corruptions import corrupt
from .data import TinyImageNetDM, get_dataset, make_loader, tiny_class_names
from .evaluation import (DEVICE, USE_AMP, bootstrap_ci, correct_top1, evaluate_linear_probe, extract_features,
                         extract_layer_features, fit_linear_probe, knn_scores, paired_bootstrap, per_class_acc, topk_acc)
from .registry import build_module, encoder_fingerprint, encoder_hparams, load_autoencoder, load_encoder


# ----------------------------------------------------------------------------- helpers
def _knn_kw(cfg):
    k = cfg.get("knn", {})
    return {"k": k.get("k", 20), "T": k.get("T", 0.1), "weight": k.get("weight", "exp")}


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


def _hash(obj, n=10) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def _print_df(df):
    with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 250):
        print(df.to_string())


def _data_key(cfg, dn):
    """Everything that defines the images of one dataset (used in cache names)."""
    d = cfg.get("data", {})
    key = {k: d.get(k) for k in ("train_per_class", "val_per_class", "test_per_class")}
    if dn != "tiny":
        key["transfer"] = cfg.get("transfer", {})
    return _hash(key, 8)


# ----------------------------------------------------------------------------- pretrain
def _run_signature(cfg):
    """The settings that define a pre-training run. A finished/started run is only reused if these match."""
    d, t = cfg.get("data", {}), cfg.get("trainer", {})
    return {"model": cfg.get("model"),
            "data": {k: d.get(k) for k in ("train_per_class", "val_per_class", "batch_size")},
            "max_epochs": t.get("max_epochs"), "seed": cfg.get("seed", 0)}


def _check_same_run(cfg, old_sig, where):
    new_sig = _run_signature(cfg)
    if old_sig != new_sig and not cfg.get("allow_config_change", False):
        raise RuntimeError(
            f"{where} was produced with different settings than the current config.\n"
            f"  stored : {old_sig}\n  current: {new_sig}\n"
            f"Use a new out_dir (e.g. --set out_dir=runs/dae_final_v2), delete '{cfg['out_dir']}', "
            f"or pass --set allow_config_change=true if this is intended (e.g. extending max_epochs).")


def pretrain(cfg):
    out = Path(cfg["out_dir"])
    mpath = out / "metrics.json"
    ckpt_dir = out / "ckpt"
    last = ckpt_dir / "last.ckpt"
    if cfg.get("skip_if_done", True) and mpath.exists():
        old = load_json(mpath)
        _check_same_run(cfg, _run_signature(old["cfg"]), str(mpath))
        print(f"[skip] {out} already finished")
        return old["metrics"]
    resume = cfg.get("resume", True) and last.exists()
    if resume:                                                                # never resume a run with other hyper-parameters
        hp = torch.load(last, map_location="cpu", weights_only=False).get("hyper_parameters", {})
        want = {k: v for k, v in cfg["model"].items() if k != "name"}
        if {k: hp.get(k) for k in want} != want and not cfg.get("allow_config_change", False):
            raise RuntimeError(f"{last} has hyper-parameters {dict(hp)} but the config asks for {want}. "
                               f"Use a new out_dir or delete the old run.")
    out.mkdir(parents=True, exist_ok=True)
    L.seed_everything(cfg.get("seed", 0), workers=True)
    dm = TinyImageNetDM(cfg["data"])
    dm.setup()
    s = dm.s
    model = build_module(cfg)
    t = cfg.get("trainer", {})
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # As required by the assignment: ModelCheckpoint(save_last=True) -> ckpt/last.ckpt every epoch, used for resuming.
    cbs = [ModelCheckpoint(dirpath=str(ckpt_dir), save_last=True, save_top_k=0,
                           every_n_epochs=t.get("ckpt_every", 1), save_on_train_epoch_end=True)]
    if t.get("knn_every", 5) > 0:
        cbs.append(KNNMonitor(s["X_train"], s["y_train"], s["X_val"], s["y_val"], s["num_classes"],
                              t.get("knn_every", 5), **_knn_kw(cfg)))
    trainer = L.Trainer(max_epochs=t["max_epochs"], accelerator="auto", devices=1, precision=_precision(cfg),
                        logger=_logger(cfg, out), callbacks=cbs, log_every_n_steps=20,
                        enable_progress_bar=t.get("progress", True), enable_model_summary=False)
    trainer.fit(model, datamodule=dm, ckpt_path=str(last) if resume else None)

    metrics = {}
    if cfg["model"]["name"] == "dae":                                         # validation reconstruction errors
        torch.manual_seed(0)                                                  # same corruption noise for every trial
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
    if cfg.get("plot") == "a2":
        print(plots.a2_scatter(df, out / "recon_vs_knn.png"))
    best = df.loc[df["knn_top1"].idxmax()]
    print("\nBEST TRIAL BY VALIDATION k-NN (copy into configs/a2_dae_final.yaml):")
    print({k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in df.columns if k.startswith("model.")},
          "| knn_top1 =", round(float(best["knn_top1"]), 2))
    _print_df(df)
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
    knn_s = knn_scores(feats["train"], ds["y_train"], target_feats, ncls, **_knn_kw(cfg))
    knn = topk_acc(knn_s, target_labels)
    # The selected probe is evaluated on Test exactly once (or reuses its val score).
    lin = best["val"] if split == "val" else evaluate_linear_probe(best["bundle"], target_feats, target_labels)

    return {
        "knn_top1": knn["top1"], "knn_top5": knn["top5"],
        "lin_top1": lin["top1"], "lin_top5": lin["top5"],
        "probe_lr": best["lr"],
        "knn_correct": correct_top1(knn_s, target_labels),
        "lin_correct": lin.get("correct"),
    }


def _pairwise(correct, extra_cols, n_boot=1000):
    """correct: {model_name: bool vector} on the same images -> rows of paired bootstrap differences (a - b)."""
    rows = []
    names = list(correct)
    for a, b in itertools.permutations(names, 2):
        if correct[a] is None or correct[b] is None:
            continue
        rows.append({**extra_cols, "model_a": a, "model_b": b, **paired_bootstrap(correct[a], correct[b], n_boot)})
    return rows


def evaluate(cfg):
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    feat_dir = Path(cfg.get("features_dir", out / "features"))
    split = cfg.get("split", "val")
    n_boot = 100 if cfg.get("smoke") else cfg.get("n_boot", 1000)
    if split == "test":
        print("!!! evaluating on the TEST split: use this only for final numbers !!!")
    rows, correct = [], {}
    for spec in cfg["models"]:
        fp = encoder_fingerprint(spec)
        enc = load_encoder(spec)
        row = {"model": spec["name"], "encoder_id": fp}
        for dn in cfg.get("datasets", ["tiny", "transfer"]):
            ds = get_dataset(dn, cfg)
            # The cache name contains the checkpoint fingerprint: a re-trained model never reuses old features.
            cache = feat_dir / f"{spec['name']}__{fp}__{ds['name']}__{_data_key(cfg, dn)}.pt"
            # Never extract or score the Test split until validation-only LR selection is done.
            feats = _feats(enc, ds, ["train", "val"], cache)
            res = run_protocol(feats, ds, split, cfg, encoder=enc, cache_file=cache)
            for m in ("knn", "lin"):
                c = res.pop(f"{m}_correct")
                correct.setdefault((ds["name"], m), {})[spec["name"]] = c
                if c is not None:
                    lo, hi = bootstrap_ci(c, n_boot)
                    res[f"{m}_top1_ci_lo"], res[f"{m}_top1_ci_hi"] = lo, hi
            row.update({f"{ds['name']}_{k}": v for k, v in res.items()})
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
        del enc
        _free()
    df = pd.DataFrame(rows)
    pair_rows = []
    for (dsn, method), c in correct.items():
        pair_rows += _pairwise(c, {"dataset": dsn, "method": method, "split": split}, n_boot)
    pd.DataFrame(pair_rows).to_csv(out / "paired_bootstrap.csv", index=False)
    _print_df(df)
    print(f"paired differences with 95% CIs: {out / 'paired_bootstrap.csv'}")
    return df


# ----------------------------------------------------------------------------- finetune
def label_subsets(y, num_classes, fractions, seed=0):
    """Class-balanced, NESTED subsets (1% subset is inside the 10% subset...), identical for every model.
    Per class we keep floor(f * n_c + 0.5) images (round half UP, at least 1). With 450 images/class:
    1% -> 5/class (1,000 labels), 10% -> 45/class (9,000), 100% -> 450/class (90,000)."""
    rng = np.random.RandomState(seed)
    y = y.numpy() if torch.is_tensor(y) else np.asarray(y)
    perm = {c: rng.permutation(np.where(y == c)[0]) for c in range(num_classes)}
    return {f: np.concatenate([perm[c][:max(1, int(math.floor(f * len(perm[c]) + 0.5)))] for c in range(num_classes)])
            for f in fractions}


def ft_schedule(n_train, ft, smoke=False):
    """Number of optimisation steps and validation interval for one fine-tuning run.
    Budget = `epochs` passes over the labelled subset, clipped to [min_steps, max_steps]; a fixed `steps`
    (legacy) overrides it. Validation is run `n_evals` times; the best validation step is kept."""
    if smoke:
        return 2, 1
    bs = ft.get("batch_size", 128)
    if ft.get("steps"):
        steps = int(ft["steps"])
    else:
        steps = math.ceil(ft.get("epochs", 30) * n_train / bs)
        steps = int(min(max(steps, ft.get("min_steps", 500)), ft.get("max_steps", 5000)))
    eval_every = max(1, math.ceil(steps / ft.get("n_evals", 10)))
    return eval_every * math.ceil(steps / eval_every), eval_every


@torch.no_grad()
def _eval_clf(model, X, y, bs=512):
    model = model.to(DEVICE).eval()
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            out.append(model(X[i:i + bs].to(DEVICE).float() / 255).float().cpu())
    scores = torch.cat(out)
    return {**topk_acc(scores, y), "correct": correct_top1(scores, y)}


def _make_clf(enc0, ncls, lr, ft, steps):
    return Classifier(copy.deepcopy(enc0), ncls, lr=lr, wd=ft.get("wd", 5e-4), total_steps=steps,   # deepcopy: never
                      warmup_frac=ft.get("warmup_frac", 0.05), crop_pad=ft.get("crop_pad", 4))      # modify the encoder


def _fine_tune_one(enc0, X, y, Xv, yv, ncls, steps, eval_every, lr, ft, seed):
    """Fine-tune every weight with one lr. Returns (model with the BEST-VALIDATION weights loaded, best step)."""
    L.seed_everything(seed, workers=True)
    model = _make_clf(enc0, ncls, lr, ft, steps)
    keep = KeepBestState("val_top1")
    bs = ft.get("batch_size", 128)
    trainer = L.Trainer(max_steps=steps, max_epochs=-1, accelerator="auto", devices=1,
                        precision=32 if not USE_AMP else "16-mixed",
                        val_check_interval=eval_every, check_val_every_n_epoch=None, num_sanity_val_steps=0,
                        callbacks=[keep], logger=False, enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False, log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=make_loader(X, y, bs, shuffle=True, drop_last=len(X) > bs),
                val_dataloaders=make_loader(Xv, yv, 512, shuffle=False, drop_last=False))
    if keep.state is not None:
        model.load_state_dict(keep.state)
    return model, keep.best_step


def _ft_fingerprint(cfg, dn, spec, frac):
    return _hash({"ft": cfg["ft"], "seed": cfg.get("seed", 0), "data": _data_key(cfg, dn), "frac": float(frac),
                  "encoder": encoder_fingerprint(spec), "smoke": bool(cfg.get("smoke"))})


def finetune(cfg):
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "results.csv"
    f = cfg["ft"]
    split, fractions = cfg.get("split", "val"), cfg["fractions"]
    reuse = cfg.get("reuse_selected_from") if split == "test" else None
    n_boot = 100 if cfg.get("smoke") else cfg.get("n_boot", 1000)

    # Resumable: a row is reused only if it was produced by exactly the same protocol and checkpoint.
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    if rows and "fp" not in rows[0]:
        print(f"[finetune] {csv} comes from an older protocol: recomputing every row")
        rows = []
    fps = {}
    for dn in cfg.get("datasets", ["tiny", "transfer"]):
        for spec in cfg["models"]:
            for frac in fractions:
                fps[(dn, spec["name"], float(frac))] = _ft_fingerprint(cfg, dn, spec, frac)
    valid_fps = set(fps.values())
    stale = [r for r in rows if r.get("fp") not in valid_fps]
    if stale:
        print(f"[finetune] dropping {len(stale)} stale row(s) produced with other settings/checkpoints")
    rows = [r for r in rows if r.get("fp") in valid_fps]
    done = {r["fp"] for r in rows}

    for dn in cfg.get("datasets", ["tiny", "transfer"]):
        ds = get_dataset(dn, cfg)
        subsets = label_subsets(ds["y_train"], ds["num_classes"], fractions)
        for spec in cfg["models"]:
            enc0 = None
            for frac in fractions:
                fp = fps[(dn, spec["name"], float(frac))]
                tag = f"{ds['name']}__{spec['name']}__{frac}"
                if fp in done:
                    continue
                idx = subsets[frac]
                steps, eval_every = ft_schedule(len(idx), f, cfg.get("smoke"))

                if reuse:
                    # Final test: the model was already selected on VALIDATION by the val config; evaluate it once.
                    sel_path = Path(reuse) / "selected" / f"{tag}.pt"
                    if not sel_path.exists():
                        raise FileNotFoundError(f"{sel_path} missing: run the validation config "
                                                f"(configs/a5_finetune.yaml) first.")
                    sel = torch.load(sel_path, map_location="cpu", weights_only=False)
                    if sel["fp"] != fp:
                        raise RuntimeError(f"{sel_path} was selected with other settings/checkpoint; "
                                           f"re-run configs/a5_finetune.yaml first.")
                    best = {k: sel[k] for k in ("lr", "val", "best_step")}
                    enc0 = enc0 if enc0 is not None else load_encoder(spec)
                    model = _make_clf(enc0, ds["num_classes"], sel["lr"], f, steps)
                    model.load_state_dict(sel["state_dict"])
                else:
                    enc0 = enc0 if enc0 is not None else load_encoder(spec)
                    best, model = None, None
                    # Train every LR candidate, keep its best-validation step; compare candidates on VALIDATION only.
                    for lr in f["lr_grid"]:
                        cand, step = _fine_tune_one(enc0, ds["X_train"][idx], ds["y_train"][idx],
                                                    ds["X_val"], ds["y_val"], ds["num_classes"],
                                                    steps, eval_every, lr, f, cfg.get("seed", 0))
                        val = _eval_clf(cand, ds["X_val"], ds["y_val"])
                        print(f"  {tag} lr={lr:g}: best val top-1 {val['top1']:.2f} at step {step}/{steps}", flush=True)
                        if best is None or val["top1"] > best["val"]["top1"]:
                            best = {"val": {k: v for k, v in val.items() if k != "correct"}, "lr": lr,
                                    "best_step": step, "val_correct": val["correct"]}
                            model = cand.cpu()
                        else:
                            del cand
                        _free()
                    if split == "val":
                        (out / "selected").mkdir(exist_ok=True)
                        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                    "lr": best["lr"], "val": best["val"], "best_step": best["best_step"], "fp": fp},
                                   out / "selected" / f"{tag}.pt")

                if split == "test":
                    target = _eval_clf(model, ds["X_test"], ds["y_test"])            # Test is touched once
                elif split == "val":
                    target = {**best["val"], "correct": best["val_correct"]}
                else:
                    raise ValueError(f"Unsupported fine-tuning split: {split!r}")
                del model
                _free()

                lo, hi = bootstrap_ci(target["correct"], n_boot)
                (out / "predictions").mkdir(exist_ok=True)
                torch.save({"correct": target["correct"], "fp": fp}, out / "predictions" / f"{tag}.pt")
                row = {"dataset": ds["name"], "model": spec["name"], "frac": float(frac), "n_labels": len(idx),
                       "labels_per_class": len(idx) // ds["num_classes"],
                       "top1": target["top1"], "top5": target["top5"], "top1_ci_lo": lo, "top1_ci_hi": hi,
                       "val_top1": best["val"]["top1"], "lr": best["lr"], "steps": steps,
                       "best_step": best["best_step"], "split": split, "fp": fp, "pred_file": f"{tag}.pt"}
                rows.append(row)
                done.add(fp)
                pd.DataFrame(rows).to_csv(csv, index=False)
                print(row, flush=True)

    df = pd.DataFrame(rows)
    pair_rows = []
    for (dsn, frac), g in df.groupby(["dataset", "frac"]):
        correct = {}
        for _, r in g.iterrows():
            p = out / "predictions" / str(r["pred_file"])
            if p.exists():
                pr = torch.load(p, map_location="cpu", weights_only=False)
                if pr["fp"] == r["fp"]:
                    correct[r["model"]] = pr["correct"]
        pair_rows += _pairwise(correct, {"dataset": dsn, "frac": frac, "split": split}, n_boot)
    pd.DataFrame(pair_rows).to_csv(out / "paired_bootstrap.csv", index=False)
    plots.label_efficiency(df, out / "label_efficiency.png")
    _print_df(df.drop(columns=["fp", "pred_file"]))
    return df


# ----------------------------------------------------------------------------- robustness
def _select_probe(f_tr, y_tr, f_va, y_va, ncls, cfg):
    """Linear probe trained on CLEAN train features, lr chosen on CLEAN validation features (never on test)."""
    p = cfg.get("probe", {})
    best = None
    for lr in p.get("lr_grid", [1e-3]):
        b = fit_linear_probe(f_tr, y_tr, ncls, epochs=2 if cfg.get("smoke") else p.get("epochs", 100),
                             lr=lr, wd=p.get("wd", 1e-4), seed=cfg.get("seed", 0))
        v = evaluate_linear_probe(b, f_va, y_va)["top1"]
        if best is None or v > best[1]:
            best = (b, v, lr)
    return best


def robustness(cfg):
    """k-NN (clean memory bank) and linear probe (trained on clean features) when the EVALUATION images are corrupted."""
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    ds = get_dataset("tiny", cfg)
    split = cfg.get("split", "val")
    X_ev, y_ev = ds[f"X_{split}"], ds[f"y_{split}"]
    use_probe = cfg.get("linear_probe", True)
    rows = []
    for spec in cfg["models"]:
        enc = load_encoder(spec)
        hp = encoder_hparams(spec)                                            # which corruption the model was trained on
        f_tr = extract_features(enc, ds["X_train"])                           # clean memory bank / probe training set
        probe = None
        if use_probe:
            probe = _select_probe(f_tr, ds["y_train"], extract_features(enc, ds["X_val"]), ds["y_val"],
                                  ds["num_classes"], cfg)
        for kind, levels in cfg["corruptions"].items():
            for lvl in (levels[:2] if cfg.get("smoke") else levels):
                torch.manual_seed(0)                                          # identical corruption noise for every model
                f_c = extract_features(enc, X_ev, transform=lambda x, k=kind, l=lvl: corrupt(x, k, l))
                acc = topk_acc(knn_scores(f_tr, ds["y_train"], f_c, ds["num_classes"], **_knn_kw(cfg)), y_ev)["top1"]
                seen = hp.get("kind") == kind and float(hp.get("level", 0) or 0) > 0
                row = {"model": spec["name"], "kind": kind, "level": lvl, "knn_top1": acc,
                       "seen_in_pretraining": seen, "pretrain_level": hp.get("level") if seen else None}
                if probe is not None:
                    row["lin_top1"] = evaluate_linear_probe(probe[0], f_c, y_ev)["top1"]
                    row["probe_lr"] = probe[2]
                rows.append(row)
        del enc, probe
        _free()
    df = pd.DataFrame(rows)
    # Relative robustness: accuracy at each level divided by the model's clean accuracy (separates robustness from skill).
    for m in ["knn"] + (["lin"] if use_probe else []):
        clean = df[df["level"] == 0].groupby("model")[f"{m}_top1"].first()
        df[f"{m}_relative_to_clean"] = [a / clean[mm] if clean.get(mm, 0) > 0 else np.nan
                                        for mm, a in zip(df["model"], df[f"{m}_top1"])]
    df["relative_to_clean"] = df["knn_relative_to_clean"]                      # backwards-compatible column name
    df.to_csv(out / "robustness.csv", index=False)
    plots.robustness(df, out / "robustness.png")
    plots.robustness(df, out / "robustness_relative.png", y="knn_relative_to_clean")
    if use_probe:
        plots.robustness(df, out / "robustness_linear.png", y="lin_top1")
        plots.robustness(df, out / "robustness_linear_relative.png", y="lin_relative_to_clean")
    _print_df(df)
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
        enc = load_encoder(spec)
        feats[spec["name"]] = {"train": extract_features(enc, ds["X_train"]), "eval": extract_features(enc, X_ev)}
        votes = knn_scores(feats[spec["name"]]["train"], ds["y_train"], feats[spec["name"]]["eval"], ds["num_classes"],
                           **_knn_kw(cfg))
        pc[spec["name"]] = per_class_acc(votes, y_ev, ds["num_classes"])
        del enc
        _free()
    table = pd.DataFrame(pc).rename_axis("class_idx")
    names = tiny_class_names(cfg["data"].get("cache_dir", "data_cache"))
    if names:
        table.insert(0, "wnid", [names[i] for i in table.index])
    table.to_csv(out / "per_class_knn.csv")
    plots.tsne_grid({n: f["eval"] for n, f in feats.items()}, y_ev, out / "tsne.png", cfg.get("tsne_classes", 10))
    rms = cfg.get("retrieval_models") or ([cfg["retrieval_model"]] if cfg.get("retrieval_model") else list(feats))
    for rm in rms:                                                            # same queries for every model
        plots.retrieval_grid(rm, feats[rm]["train"], feats[rm]["eval"], ds["X_train"], ds["y_train"], X_ev, y_ev,
                             out / f"retrieval_{rm}.png")
    _layer_choice(cfg, ds, split, out)
    _reconstructions(cfg, X_ev, out)
    worst = {}
    for n, a in pc.items():
        order = np.argsort(np.nan_to_num(a, nan=1e9))[:10].tolist()
        worst[n] = [{"class_idx": int(c), "wnid": names[c] if names else None, "knn_top1": float(a[c])} for c in order]
    save_json(worst, out / "worst_10_classes.json")
    print("worst classes per model:", {n: [w["class_idx"] for w in v] for n, v in worst.items()})


def _layer_choice(cfg, ds, split, out):
    """A2 'which layer is h, and why': k-NN accuracy of the pooled output of several ResNet stages for every model.
    layer4 pooled = h. Decide on the VALIDATION run; the test run only reports."""
    layers = cfg.get("layers", ["layer2", "layer3", "layer4"])
    rows = []
    for spec in cfg["models"]:
        enc = load_encoder(spec)
        f_tr = extract_layer_features(enc, ds["X_train"], layers)
        if not f_tr:
            print(f"[layer choice] skipped: encoder has none of the modules {layers} (tiny test backbone?)")
            return None
        f_ev = extract_layer_features(enc, ds[f"X_{split}"], layers)
        for name in f_tr:
            scores = knn_scores(f_tr[name], ds["y_train"], f_ev[name], ds["num_classes"], **_knn_kw(cfg))
            acc = topk_acc(scores, ds[f"y_{split}"])
            rows.append({"model": spec["name"], "layer": name + (" (= h)" if name == "layer4" else ""),
                         "dim": f_tr[name].shape[1], "knn_top1": acc["top1"], "knn_top5": acc["top5"], "split": split})
        del enc, f_tr, f_ev
        _free()
    df = pd.DataFrame(rows)
    df.to_csv(out / "layer_choice.csv", index=False)
    plots.layer_choice(df, out / "layer_choice.png")
    _print_df(df)
    return df


@torch.no_grad()
def _reconstructions(cfg, X_ev, out, n=8):
    """A1 qualitative evidence: clean x, the model's input x~ (its own training corruption) and g(f(x~)), per model."""
    panels = []
    idx = torch.linspace(0, len(X_ev) - 1, n).long()
    x = X_ev[idx].float() / 255
    for spec in cfg["models"]:
        ae = load_autoencoder(spec)
        if ae is None:
            continue
        ae = ae.to(DEVICE).eval()
        torch.manual_seed(0)
        xd = x.to(DEVICE)
        x_in = corrupt(xd, ae.hparams.kind, ae.hparams.level)
        rec = ae.decoder(ae.encoder(x_in.contiguous(memory_format=torch.channels_last))).float()
        rec_clean = ae.decoder(ae.encoder(xd.contiguous(memory_format=torch.channels_last))).float()
        mse = float(F.mse_loss(rec, xd))
        panels.append((spec["name"], ae.hparams.kind, ae.hparams.level, x_in.cpu(), rec.cpu(), rec_clean.cpu(), mse))
        del ae
        _free()
    if panels:
        plots.reconstructions(x, panels, out / "reconstructions.png")


TASKS = {"pretrain": pretrain, "sweep": sweep, "evaluate": evaluate, "finetune": finetune,
         "robustness": robustness, "interpret": interpret}
