from __future__ import annotations

import torch
from torch import nn


class DCGANGenerator(nn.Module):
    def __init__(self, z_dim: int = 128, image_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.project = nn.Sequential(
            nn.Linear(z_dim, base_channels * 8 * 4 * 4),
            nn.BatchNorm1d(base_channels * 8 * 4 * 4),
            nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).view(z.size(0), -1, 4, 4)
        return self.net(x)
