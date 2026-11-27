"""Fine-tuning / supervised baseline: encoder + linear head, all weights trained with labels."""
import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


def random_crop_flip(x, pad=4):
    """GPU augmentation used for EVERY fine-tuned model (identical protocol): reflect-pad + random crop + h-flip."""
    B, C, H, W = x.shape
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    if pad <= 0:
        return x
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = i[:, None] + torch.arange(H, device=x.device)[None]                     # (B, H)
    cols = j[:, None] + torch.arange(W, device=x.device)[None]                     # (B, W)
    xp = xp.gather(2, rows[:, None, :, None].expand(B, C, H, W + 2 * pad))
    return xp.gather(3, cols[:, None, None, :].expand(B, C, H, W))


class Classifier(L.LightningModule):
    """encoder f_theta + linear head; AdamW with linear warm-up + cosine decay over `total_steps`.
    Validation top-1 is logged as `val_top1` (used to keep the best step, see callbacks.KeepBestState)."""

    def __init__(self, encoder, num_classes, lr=5e-4, wd=5e-4, total_steps=None, warmup_frac=0.05, crop_pad=4):
        super().__init__()
        self.encoder = encoder.to(memory_format=torch.channels_last)
        self.head = nn.Linear(512, num_classes)
        self.lr, self.wd = lr, wd
        self.total_steps, self.warmup_frac, self.crop_pad = total_steps, warmup_frac, crop_pad
        self._correct, self._n = 0, 0

    def forward(self, x):
        return self.head(self.encoder(x.contiguous(memory_format=torch.channels_last)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = random_crop_flip(x.float() / 255, self.crop_pad)
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True, batch_size=x.size(0))
        return loss

    def on_validation_epoch_start(self):
        self._correct, self._n = 0, 0

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x.float() / 255).argmax(1)
        self._correct += int((pred == y).sum())
        self._n += int(y.numel())

    def on_validation_epoch_end(self):
        self.log("val_top1", 100.0 * self._correct / max(1, self._n))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        if not self.total_steps:
            return opt
        total = max(1, int(self.total_steps))
        warm = max(1, int(self.warmup_frac * total))

        def schedule(step):
            if step < warm:
                return (step + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
