"""Part-A training entry point, following the course ``train.py`` pattern.

Every experiment is launched with its own YAML configuration:

    python train.py --config configs/a2_dae_final.yaml
    python train.py --config configs/a1_tune_dae.yaml --smoke
    python train.py --config configs/a4_eval.yaml --set probe.epochs=20

The course's lecture folders are intentionally left untouched. Part A uses the
same high-level training conventions (seeded run, config-driven model/task
selection, LightningModules, LightningDataModules and Lightning Trainer), while
its Tiny-ImageNet-specific implementations live in ``src/``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Course-style entry point for Representation Learning Midterm Part A"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML configuration for this run (for example configs/a2_dae_final.yaml)",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override YAML values, e.g. --set trainer.max_epochs=3 model.lr=1e-3",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Shrink the selected config for a quick pipeline check; not for reporting results",
    )
    return parser.parse_args(argv)


def _resolve_config_path(value: str, project_dir: Path) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_absolute():
        candidate = supplied.resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Configuration file not found: {candidate}")

    # Support both `python train.py --config configs/x.yaml` from midterm_a/
    # and `python midterm_a/train.py --config midterm_a/configs/x.yaml` from repo root.
    candidates = [
        (Path.cwd() / supplied).resolve(),
        (project_dir / supplied).resolve(),
        (project_dir.parent / supplied).resolve(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find config {value!r}. Tried:\n  {tried}\n"
        "Run from midterm_a/ or pass a valid config path."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    project_dir = Path(__file__).resolve().parent
    config_path = _resolve_config_path(args.config, project_dir)

    # Config inheritance (extends:) and relative data/output paths are defined
    # relative to the Part-A project directory, regardless of launch location.
    os.chdir(project_dir)
    project_str = str(project_dir)
    if project_str not in sys.path:
        sys.path.insert(0, project_str)

    import torch
    import lightning as L

    from src.common import apply_overrides, apply_smoke, load_config
    from src.tasks import TASKS

    cfg = load_config(config_path)
    cfg = apply_overrides(cfg, args.set)
    if args.smoke:
        cfg = apply_smoke(cfg)

    task_name = cfg.get("task")
    if not task_name:
        raise ValueError(f"Config {config_path} does not define a 'task' field.")
    if task_name not in TASKS:
        available = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task {task_name!r}. Available tasks: {available}")

    seed = int(cfg.get("seed", 0))
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    print("=" * 72, flush=True)
    print("Part-A course-style training entry point", flush=True)
    print(f"Project: {project_dir}", flush=True)
    print(f"Config:  {config_path}", flush=True)
    print(f"Task:    {task_name}", flush=True)
    print(f"Seed:    {seed}", flush=True)
    print(f"Smoke:   {bool(args.smoke)}", flush=True)
    if cfg.get("split") == "test":
        print("WARNING: TEST split selected. Use only for final reporting.", flush=True)
    print("=" * 72, flush=True)

    TASKS[task_name](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
