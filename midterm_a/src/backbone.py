"""ResNet-18 encoder with a small-image stem and the light DAE decoder."""
import os

import torch.nn as nn
import torch.nn.functional as F
import torchvision


def make_encoder():
    """f_theta. Output h = global-average-pooled 512-d feature (before any head/decoder)."""
    if os.environ.get("SSL_TINY_BACKBONE"):                                    # unit/smoke tests on CPU only; never set for real runs
        return nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                             nn.Conv2d(32, 512, 3, 4, 1), nn.BatchNorm2d(512), nn.ReLU(),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten())
    m = torchvision.models.resnet18(weights=None)                              # random init, NO pretrained weights
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 3x3 stride 1 (instead of 7x7 stride 2)
    m.maxpool = nn.Identity()                                                  # no max-pool
    m.fc = nn.Identity()                                                       # remove ImageNet classifier -> h in R^512
    return m


class Decoder(nn.Module):
    """g_phi: light decoder, 512 -> 4x4x256 -> four stride-2 transposed convs -> 64x64x3 in [0,1]."""

    def __init__(self, h_dim=512):
        super().__init__()
        self.fc = nn.Linear(h_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, h):
        return self.net(F.relu(self.fc(h)).view(-1, 256, 4, 4))

