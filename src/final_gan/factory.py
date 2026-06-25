from __future__ import annotations

import torch

from final_gan.models import (
    DCGANGenerator,
    Discriminator,
    StyleGANLiteGenerator,
    StyleGANLiteV2Generator,
    weights_init_normal,
)


def build_generator(config: dict) -> torch.nn.Module:
    model_cfg = config["model"]
    name = model_cfg.get("name", "dcgan")
    z_dim = int(model_cfg.get("z_dim", 128))
    if name == "dcgan":
        return DCGANGenerator(z_dim=z_dim)
    if name == "stylegan_lite":
        return StyleGANLiteGenerator(z_dim=z_dim, w_dim=int(model_cfg.get("w_dim", 256)))
    if name == "stylegan_lite_v2":
        return StyleGANLiteV2Generator(z_dim=z_dim, w_dim=int(model_cfg.get("w_dim", 256)))
    raise ValueError(f"Unknown model '{name}'. Use dcgan, stylegan_lite, or stylegan_lite_v2.")


def build_discriminator(config: dict) -> torch.nn.Module:
    model_cfg = config.get("model", {})
    use_minibatch_std = bool(model_cfg.get("use_minibatch_std", False))
    if model_cfg.get("name") == "stylegan_lite_v2":
        use_minibatch_std = True
    return Discriminator(use_minibatch_std=use_minibatch_std)


def initialize_models(generator: torch.nn.Module, discriminator: torch.nn.Module) -> None:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
