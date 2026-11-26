"""Config loading, CLI overrides and small helpers shared by all tasks."""
import copy
import json
import re
from pathlib import Path

import yaml


def deep_update(base: dict, new: dict) -> dict:
    """Recursively merge `new` into a copy of `base` (dicts are merged, everything else is replaced)."""
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


_SCI = re.compile(r"^[+-]?\d+(\.\d*)?[eE][+-]?\d+$")


def _fix_floats(obj):
    """PyYAML reads 1e-3 (no dot) as a string. Convert such scientific-notation strings to floats."""
    if isinstance(obj, dict):
        return {k: _fix_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_floats(v) for v in obj]
    if isinstance(obj, str) and _SCI.match(obj.strip()):
        return float(obj)
    return obj


def load_config(path) -> dict:
    """Load a YAML config. If it has `extends: other.yaml` the parent is loaded first and merged.
    Paths in `extends` are relative to the repo root (run everything from the repo root)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _fix_floats(cfg)
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_update(load_config(parent), cfg)
    return cfg


def set_by_path(cfg: dict, dotted: str, value) -> None:
    """set_by_path(cfg, 'model.lr', 1e-3) -> cfg['model']['lr'] = 1e-3 (creates missing dicts)."""
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def apply_overrides(cfg: dict, overrides) -> dict:
    """Apply CLI overrides of the form key.sub=value (value parsed as YAML: numbers, lists, true/false...)."""
    for item in overrides or []:
        key, val = item.split("=", 1)
        set_by_path(cfg, key, _fix_floats(yaml.safe_load(val)))
    return cfg


def apply_smoke(cfg: dict) -> dict:
    """Shrink a config for a CPU-leaning end-to-end smoke test (not for real results)."""
    cfg["smoke"] = True
    d = cfg.setdefault("data", {})
    # Keep smoke runs genuinely small on CPU: 1 image/class in train/val/test.
    # This changes only the throwaway smoke config, never the real experiment config.
    d["train_per_class"] = min(d.get("train_per_class", 1), 1)
    d["val_per_class"] = min(d.get("val_per_class", 1), 1)
    d["test_per_class"] = min(d.get("test_per_class", 1), 1)
    d["batch_size"] = min(d.get("batch_size", 32), 32)
    cfg.setdefault("trainer", {})["max_epochs"] = 1
    cfg["trainer"]["knn_every"] = 0
    return cfg


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
