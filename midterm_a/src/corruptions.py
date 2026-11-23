"""Corruptions q(x_tilde | x) used (a) as the DAE noise model and (b) for robustness curves.
All take a float batch in [0,1] and return the same shape. `level` meaning is per kind (see comments)."""
import torch
import torch.nn.functional as F


def gaussian_blur_batch(x, sigma, k=7):
    """Separable Gaussian blur with one sigma per sample (sigma: (B,)). Tiny sigma = identity."""
    B, C, H, W = x.shape
    t = torch.arange(k, device=x.device).float() - k // 2
    ker = torch.exp(-t[None, :] ** 2 / (2 * sigma[:, None] ** 2))
    ker = (ker / ker.sum(1, keepdim=True)).repeat_interleave(C, 0)          # (B*C, k)
    y = x.reshape(1, B * C, H, W)
    p = k // 2
    y = F.conv2d(F.pad(y, (p, p, 0, 0), mode="reflect"), ker.view(B * C, 1, 1, k), groups=B * C)
    y = F.conv2d(F.pad(y, (0, 0, p, p), mode="reflect"), ker.view(B * C, 1, k, 1), groups=B * C)
    return y.view(B, C, H, W)


def gaussian(x, level):                                                          # level = sigma
    return (x + level * torch.randn_like(x)).clamp(0, 1)


def mask(x, level):                                                              # level = P(pixel dropped)
    keep = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) > level).to(x.dtype)
    return x * keep


def saltpepper(x, level):                                                        # level = fraction of corrupted pixels
    r = torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
    x = torch.where(r < level / 2, torch.zeros_like(x), x)
    return torch.where(r > 1 - level / 2, torch.ones_like(x), x)


def contrast(x, level):                                                          # level in [0,1): 0 = none, 1 = flat gray
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * (1 - level) + m).clamp(0, 1)


def blur(x, level):                                                              # level = Gaussian sigma (pixels)
    return gaussian_blur_batch(x, torch.full((x.shape[0],), float(level), device=x.device))


# BONUS HOOK: add your own corruption here (a function (x, level) -> x) and list it in the robustness config.
REGISTRY = {"gaussian": gaussian, "mask": mask, "saltpepper": saltpepper, "contrast": contrast, "blur": blur}


def corrupt(x, kind, level):
    if kind == "none" or level == 0:
        return x
    if kind not in REGISTRY:
        raise ValueError(f"unknown corruption '{kind}'. Available: {list(REGISTRY)}")
    return REGISTRY[kind](x, level)
