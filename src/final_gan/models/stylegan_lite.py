from __future__ import annotations

import random

import torch
from torch import nn
from torch.nn import functional as F


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int = 128, w_dim: int = 256, num_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [PixelNorm()]
        in_dim = z_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2, inplace=True)])
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AdaIN(nn.Module):
    def __init__(self, channels: int, w_dim: int) -> None:
        super().__init__()
        self.affine = nn.Linear(w_dim, channels * 2)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        style = self.affine(w).view(w.size(0), 2, x.size(1), 1, 1)
        gamma = style[:, 0] + 1.0
        beta = style[:, 1]
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True, unbiased=False) + 1e-8
        return gamma * (x - mean) / std + beta


class NoiseInjection(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class StyledConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, w_dim: int, upsample: bool) -> None:
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.noise = NoiseInjection(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.adain = AdaIN(out_channels, w_dim)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        x = self.noise(x)
        x = self.activation(x)
        return self.adain(x, w)


class StyleGANLiteGenerator(nn.Module):
    def __init__(self, z_dim: int = 128, w_dim: int = 256, image_channels: int = 3) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.mapping = MappingNetwork(z_dim=z_dim, w_dim=w_dim)
        channels = [512, 512, 256, 128, 64, 32]
        self.constant = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        blocks: list[nn.Module] = []
        for i in range(len(channels) - 1):
            blocks.append(
                StyledConv(
                    channels[i],
                    channels[i + 1],
                    w_dim=w_dim,
                    upsample=i > 0,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.Sequential(nn.Conv2d(channels[-1], image_channels, 1), nn.Tanh())

    def forward_w(self, w: torch.Tensor) -> torch.Tensor:
        x = self.constant.repeat(w.size(0), 1, 1, 1)
        for block in self.blocks:
            x = block(x, w)
        return self.to_rgb(x)

    def forward(
        self,
        z: torch.Tensor,
        mixing_z: torch.Tensor | None = None,
        style_mixing_prob: float = 0.0,
    ) -> torch.Tensor:
        w = self.mapping(z)
        if mixing_z is None or style_mixing_prob <= 0 or random.random() >= style_mixing_prob:
            return self.forward_w(w)

        w2 = self.mapping(mixing_z)
        cutoff = random.randint(1, len(self.blocks) - 1)
        x = self.constant.repeat(z.size(0), 1, 1, 1)
        for idx, block in enumerate(self.blocks):
            x = block(x, w if idx < cutoff else w2)
        return self.to_rgb(x)
