from __future__ import annotations

from collections.abc import Iterable

import torch

from final_gan.utils import images_to_uint8


def _require_torchmetrics():
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.inception import InceptionScore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Image metrics require torchmetrics and torch-fidelity. "
            "Install them with: pip install -r requirements.txt"
        ) from exc
    return FrechetInceptionDistance, InceptionScore


@torch.no_grad()
def evaluate_generator(
    generator: torch.nn.Module,
    dataloader: Iterable,
    model_name: str,
    z_dim: int,
    device: torch.device,
    num_images: int = 2048,
    batch_size: int = 64,
) -> dict[str, float]:
    FrechetInceptionDistance, InceptionScore = _require_torchmetrics()
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.set_dtype(torch.float64)
    inception = InceptionScore(normalize=False).to(device)

    generator.eval()
    seen_real = 0
    for real, _ in dataloader:
        real = real.to(device)
        fid.update(images_to_uint8(real), real=True)
        seen_real += real.size(0)
        if seen_real >= num_images:
            break

    generated = 0
    while generated < num_images:
        current = min(batch_size, num_images - generated)
        z = torch.randn(current, z_dim, device=device)
        if model_name == "stylegan_lite":
            fake = generator(z, style_mixing_prob=0.0)
        else:
            fake = generator(z)
        fake_uint8 = images_to_uint8(fake)
        fid.update(fake_uint8, real=False)
        inception.update(fake_uint8)
        generated += current

    is_mean, is_std = inception.compute()
    return {
        "fid": float(fid.compute().item()),
        "is_mean": float(is_mean.item()),
        "is_std": float(is_std.item()),
    }
