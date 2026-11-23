"""Fine-tuning / supervised baseline: encoder + linear head, all weights trained with labels."""
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(L.LightningModule):
    def __init__(self, encoder, num_classes, lr=5e-4, wd=5e-4):
        super().__init__()
        self.encoder = encoder.to(memory_format=torch.channels_last)
        self.head = nn.Linear(512, num_classes)
        self.lr, self.wd = lr, wd

    def forward(self, x):
        return self.head(self.encoder(x.contiguous(memory_format=torch.channels_last)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float() / 255
        flip = torch.rand(x.shape[0], device=x.device) < 0.5                 # only augmentation: horizontal flip (same for all models)
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True, batch_size=x.size(0))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
