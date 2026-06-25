from __future__ import annotations

import torch
from torch import nn


class MinibatchStdDev(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        if batch <= 1:
            std_feature = x.new_zeros(batch, 1, height, width)
        else:
            std = torch.sqrt(x.var(dim=0, unbiased=False) + self.eps)
            mean_std = std.mean().view(1, 1, 1, 1)
            std_feature = mean_std.repeat(batch, 1, height, width)
        return torch.cat([x, std_feature], dim=1)


class Discriminator(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        use_minibatch_std: bool = False,
    ) -> None:
        super().__init__()
        final_in_channels = base_channels * 8 + (1 if use_minibatch_std else 0)
        final_layers: list[nn.Module] = []
        if use_minibatch_std:
            final_layers.append(MinibatchStdDev())
        final_layers.append(nn.Conv2d(final_in_channels, 1, 4, 1, 0, bias=False))
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            *final_layers,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images).flatten(1).squeeze(1)
