"""Build modules from config and load trained encoders."""
import hashlib
from pathlib import Path

import torch

from .backbone import make_encoder
from .dae import DAE


def build_module(cfg):
    m = dict(cfg["model"])
    name = m.pop("name")
    if name == "dae":
        return DAE(**m)
    raise ValueError(f"unknown model '{name}' (only 'dae' in this submission)")


def ckpt_fingerprint(path) -> str:
    """Short id of a checkpoint file (path + size + mtime). Changes whenever the checkpoint is re-trained,
    so cached features / fine-tuning results of an older checkpoint are never silently reused."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}. Train it first (see README).")
    st = p.stat()
    return hashlib.sha1(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:10]


def encoder_fingerprint(spec) -> str:
    spec = {"type": "ckpt", **spec}
    if spec["type"] == "random":
        return f"random{spec.get('seed', 0)}"
    return ckpt_fingerprint(spec["path"])


def encoder_hparams(spec) -> dict:
    """Pre-training hyper-parameters stored in the checkpoint ({} for the random-init encoder)."""
    spec = {"type": "ckpt", **spec}
    if spec["type"] == "random":
        return {}
    ck = torch.load(spec["path"], map_location="cpu", weights_only=False)
    return dict(ck.get("hyper_parameters", {}))


def load_encoder(spec):
    """spec: {type: random, seed: 0} -> untrained encoder;  {type: ckpt, arch: dae, path: ...} -> trained encoder."""
    spec = {"type": "ckpt", **spec}
    if spec["type"] == "random":
        torch.manual_seed(spec.get("seed", 0))
        return make_encoder()
    if spec["type"] == "ckpt":
        return DAE.load_from_checkpoint(spec["path"], map_location="cpu").encoder
    raise ValueError(f"unknown model type '{spec['type']}'")
