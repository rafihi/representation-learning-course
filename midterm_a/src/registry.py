"""Build modules from config and load trained encoders."""
import torch

from .backbone import make_encoder
from .dae import DAE


def build_module(cfg):
    m = dict(cfg["model"])
    name = m.pop("name")
    if name == "dae":
        return DAE(**m)
    raise ValueError(f"unknown model '{name}' (only 'dae' in this submission)")


def load_encoder(spec):
    """spec: {type: random, seed: 0} -> untrained encoder;  {type: ckpt, arch: dae, path: ...} -> trained encoder."""
    if spec["type"] == "random":
        torch.manual_seed(spec.get("seed", 0))
        return make_encoder()
    if spec["type"] == "ckpt":
        return DAE.load_from_checkpoint(spec["path"], map_location="cpu").encoder
    raise ValueError(f"unknown model type '{spec['type']}'")
