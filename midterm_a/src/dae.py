"""Denoising autoencoder (A1). kind='none' gives the plain/vanilla AE baseline."""
import lightning as L
import torch
import torch.nn.functional as F

from .backbone import Decoder, make_encoder
from .corruptions import corrupt


class DAE(L.LightningModule):
    def __init__(self, kind="gaussian", level=0.3, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = make_encoder().to(memory_format=torch.channels_last)
        self.decoder = Decoder()

    def forward(self, x):
        return self.encoder(x)                                               # h (512-d)

    def _recon(self, x, x_in):
        return self.decoder(self.encoder(x_in.contiguous(memory_format=torch.channels_last)))

    def training_step(self, batch, batch_idx):
        x = batch[0].float() / 255                                           # labels (batch[1]) are never used here
        x_tilde = corrupt(x, self.hparams.kind, self.hparams.level)          # encoder sees the corrupted image...
        loss = F.mse_loss(self._recon(x, x_tilde), x)                        # ...decoder must rebuild the CLEAN one
        self.log("train_loss", loss, prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0].float() / 255
        self.log("val_recon", F.mse_loss(self._recon(x, x), x), batch_size=x.size(0))       # clean->clean: comparable across trials
        x_tilde = corrupt(x, self.hparams.kind, self.hparams.level)
        self.log("val_recon_corrupted", F.mse_loss(self._recon(x, x_tilde), x), batch_size=x.size(0))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
