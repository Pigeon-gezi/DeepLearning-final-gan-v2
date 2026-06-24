from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import save_image

from final_gan.factory import build_generator
from final_gan.utils import denormalize, ensure_dir, get_device


def load_generator(checkpoint_path: str | Path, device: torch.device | str = "auto") -> tuple[torch.nn.Module, dict]:
    device = get_device(device) if isinstance(device, str) else device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    generator = build_generator(config).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator, config


@torch.no_grad()
def save_samples(
    checkpoint_path: str | Path,
    output_path: str | Path,
    num_images: int = 64,
    nrow: int = 8,
    device: str = "auto",
) -> None:
    device_obj = get_device(device)
    generator, config = load_generator(checkpoint_path, device_obj)
    z_dim = int(config["model"].get("z_dim", 128))
    model_name = config["model"].get("name", "dcgan")
    z = torch.randn(num_images, z_dim, device=device_obj)
    images = generator(z, style_mixing_prob=0.0) if model_name == "stylegan_lite" else generator(z)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    save_image(denormalize(images), output_path, nrow=nrow)


@torch.no_grad()
def save_interpolations(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    pairs: int = 2,
    steps: int = 11,
    space: str = "z",
    device: str = "auto",
) -> None:
    device_obj = get_device(device)
    generator, config = load_generator(checkpoint_path, device_obj)
    z_dim = int(config["model"].get("z_dim", 128))
    model_name = config["model"].get("name", "dcgan")
    output_dir = ensure_dir(output_dir)
    alphas = torch.linspace(0, 1, steps, device=device_obj).view(steps, 1)

    for pair_idx in range(pairs):
        z1 = torch.randn(1, z_dim, device=device_obj)
        z2 = torch.randn(1, z_dim, device=device_obj)
        if space == "w":
            if model_name != "stylegan_lite":
                raise ValueError("w-space interpolation is only available for stylegan_lite.")
            w1 = generator.mapping(z1)
            w2 = generator.mapping(z2)
            w = (1 - alphas) * w1 + alphas * w2
            images = generator.forward_w(w)
        else:
            z = (1 - alphas) * z1 + alphas * z2
            images = generator(z, style_mixing_prob=0.0) if model_name == "stylegan_lite" else generator(z)
        save_image(denormalize(images), output_dir / f"interpolation_{space}_{pair_idx:02d}.png", nrow=steps)
