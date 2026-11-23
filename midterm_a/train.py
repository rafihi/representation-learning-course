"""Single entry point. Every run of this project is launched as

    python train.py --config configs/<name>.yaml [--set key.sub=value ...] [--smoke]

--smoke shrinks the config (tiny data, 1 epoch...) to test the pipeline in about a minute.
With SSL_FAKE_DATA=1 it also needs no downloads.
"""
import argparse

import torch

from src.common import apply_overrides, apply_smoke, load_config
from src.tasks import TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. --set model.lr=1e-3 trainer.max_epochs=5")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.smoke:
        apply_smoke(cfg)
    torch.backends.cudnn.benchmark = True
    TASKS[cfg["task"]](cfg)


if __name__ == "__main__":
    main()
