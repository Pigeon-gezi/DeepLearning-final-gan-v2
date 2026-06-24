from __future__ import annotations

import torch

from final_gan.models import DCGANGenerator, Discriminator, StyleGANLiteGenerator, weights_init_normal


def build_generator(config: dict) -> torch.nn.Module:
    model_cfg = config["model"]
    name = model_cfg.get("name", "dcgan")
    z_dim = int(model_cfg.get("z_dim", 128))
    if name == "dcgan":
        return DCGANGenerator(z_dim=z_dim)
    if name == "stylegan_lite":
        return StyleGANLiteGenerator(z_dim=z_dim, w_dim=int(model_cfg.get("w_dim", 256)))
    raise ValueError(f"Unknown model '{name}'. Use dcgan or stylegan_lite.")


def build_discriminator(config: dict) -> torch.nn.Module:
    return Discriminator()


def initialize_models(generator: torch.nn.Module, discriminator: torch.nn.Module) -> None:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
