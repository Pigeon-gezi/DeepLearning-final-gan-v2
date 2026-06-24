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


class ImageMetricsEvaluator:
    def __init__(
        self,
        dataloader: Iterable,
        model_name: str,
        z_dim: int,
        device: torch.device,
        num_images: int = 2048,
        batch_size: int = 64,
    ) -> None:
        FrechetInceptionDistance, InceptionScore = _require_torchmetrics()
        try:
            self.fid = FrechetInceptionDistance(
                feature=2048,
                normalize=False,
                reset_real_features=False,
            ).to(device)
            self._manual_real_cache = False
        except TypeError:
            self.fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
            self._manual_real_cache = True
        self.fid.set_dtype(torch.float64)
        self.inception = InceptionScore(normalize=False).to(device)
        self.dataloader = dataloader
        self.model_name = model_name
        self.z_dim = z_dim
        self.device = device
        self.num_images = num_images
        self.batch_size = batch_size
        self._real_ready = False
        self._real_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _prepare_real_stats(self) -> None:
        if self._real_ready:
            return
        seen_real = 0
        for real, _ in self.dataloader:
            real = real.to(self.device)
            self.fid.update(images_to_uint8(real), real=True)
            seen_real += real.size(0)
            if seen_real >= self.num_images:
                break
        if self._manual_real_cache:
            self._real_cache = {
                "real_features_sum": self.fid.real_features_sum.detach().clone(),
                "real_features_cov_sum": self.fid.real_features_cov_sum.detach().clone(),
                "real_features_num_samples": self.fid.real_features_num_samples.detach().clone(),
            }
        self._real_ready = True

    def _reset_metrics(self) -> None:
        self.fid.reset()
        if self._manual_real_cache:
            for name, value in self._real_cache.items():
                getattr(self.fid, name).copy_(value)
        self.inception.reset()

    @torch.no_grad()
    def evaluate(self, generator: torch.nn.Module) -> dict[str, float]:
        self._prepare_real_stats()
        self._reset_metrics()
        generator.eval()

        generated = 0
        while generated < self.num_images:
            current = min(self.batch_size, self.num_images - generated)
            z = torch.randn(current, self.z_dim, device=self.device)
            if self.model_name == "stylegan_lite":
                fake = generator(z, style_mixing_prob=0.0)
            else:
                fake = generator(z)
            fake_uint8 = images_to_uint8(fake)
            self.fid.update(fake_uint8, real=False)
            self.inception.update(fake_uint8)
            generated += current

        is_mean, is_std = self.inception.compute()
        return {
            "fid": float(self.fid.compute().item()),
            "is_mean": float(is_mean.item()),
            "is_std": float(is_std.item()),
        }


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
    evaluator = ImageMetricsEvaluator(
        dataloader=dataloader,
        model_name=model_name,
        z_dim=z_dim,
        device=device,
        num_images=num_images,
        batch_size=batch_size,
    )
    return evaluator.evaluate(generator)
