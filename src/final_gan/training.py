from __future__ import annotations

from copy import deepcopy
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torchvision.utils import save_image
from tqdm import tqdm

from final_gan.config import save_config
from final_gan.data import build_dataloader
from final_gan.factory import build_discriminator, build_generator, initialize_models
from final_gan.metrics import ImageMetricsEvaluator
from final_gan.utils import append_jsonl, count_parameters, denormalize, ensure_dir, get_device, set_seed


def _sample_generator(generator: nn.Module, model_name: str, z: torch.Tensor) -> torch.Tensor:
    if model_name.startswith("stylegan_lite"):
        return generator(z, style_mixing_prob=0.0)
    return generator(z)


def save_checkpoint(
    path: str | Path,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    config: dict,
    metrics: dict | None = None,
    ema_generator: nn.Module | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    checkpoint = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "config": config,
            "metrics": metrics or {},
    }
    if ema_generator is not None:
        checkpoint["generator_ema"] = ema_generator.state_dict()
    torch.save(checkpoint, path)


def create_optimizers(
    generator: nn.Module,
    discriminator: nn.Module,
    train_cfg: dict,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    lr = float(train_cfg["lr"])
    lr_g_value = train_cfg.get("lr_g")
    lr_d_value = train_cfg.get("lr_d")
    lr_g = float(lr if lr_g_value is None else lr_g_value)
    lr_d = float(lr if lr_d_value is None else lr_d_value)
    betas = (float(train_cfg["beta1"]), float(train_cfg["beta2"]))
    return Adam(generator.parameters(), lr=lr_g, betas=betas), Adam(discriminator.parameters(), lr=lr_d, betas=betas)


def create_ema_generator(generator: nn.Module, ema_beta: float | None) -> nn.Module | None:
    if ema_beta is None:
        return None
    ema_generator = deepcopy(generator)
    ema_generator.eval()
    for param in ema_generator.parameters():
        param.requires_grad_(False)
    return ema_generator


@torch.no_grad()
def update_ema(ema_generator: nn.Module, generator: nn.Module, beta: float) -> None:
    ema_state = ema_generator.state_dict()
    model_state = generator.state_dict()
    for key, ema_value in ema_state.items():
        model_value = model_state[key].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(beta).add_(model_value, alpha=1.0 - beta)
        else:
            ema_value.copy_(model_value)


def r1_regularization(real_logits: torch.Tensor, real_images: torch.Tensor) -> torch.Tensor:
    grad_real = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad_real.pow(2).view(grad_real.size(0), -1).sum(dim=1).mean()


def load_best_fid(log_path: str | Path, checkpoint_metrics: dict | None = None) -> float:
    best_fid = float("inf")
    log_path = Path(log_path)
    if log_path.exists():
        try:
            with log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if "fid" in row:
                        best_fid = min(best_fid, float(row["fid"]))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            best_fid = float("inf")
    if best_fid == float("inf") and checkpoint_metrics and "fid" in checkpoint_metrics:
        try:
            best_fid = float(checkpoint_metrics["fid"])
        except (TypeError, ValueError):
            pass
    return best_fid


def load_training_checkpoint(
    checkpoint_path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
    ema_generator: nn.Module | None = None,
) -> tuple[int, dict]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_config = checkpoint.get("config", {})
    ckpt_model = ckpt_config.get("model", {}).get("name")
    current_model = config.get("model", {}).get("name")
    if ckpt_model is not None and ckpt_model != current_model:
        raise RuntimeError(
            f"Checkpoint model '{ckpt_model}' does not match current config model '{current_model}'."
        )
    required = ["epoch", "generator", "discriminator", "opt_g", "opt_d"]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise RuntimeError(f"Checkpoint is missing required fields: {', '.join(missing)}")

    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    opt_g.load_state_dict(checkpoint["opt_g"])
    opt_d.load_state_dict(checkpoint["opt_d"])
    if ema_generator is not None:
        if "generator_ema" in checkpoint:
            ema_generator.load_state_dict(checkpoint["generator_ema"])
        else:
            ema_generator.load_state_dict(generator.state_dict())
    start_epoch = int(checkpoint["epoch"]) + 1
    return start_epoch, checkpoint.get("metrics", {})


def train(config: dict, resume_from: str | Path | None = None) -> None:
    set_seed(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    output_dir = ensure_dir(config["paths"]["output_dir"])
    sample_dir = ensure_dir(output_dir / "samples")
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_path = output_dir / "train_log.jsonl"
    save_config(config, output_dir / "resolved_config.yaml")

    dataloader = build_dataloader(config, shuffle=True)
    eval_config = {**config, "train": {**config["train"], "batch_size": config["eval"].get("batch_size", 64)}}
    eval_loader = build_dataloader(eval_config, shuffle=False)

    generator = build_generator(config).to(device)
    discriminator = build_discriminator(config).to(device)
    initialize_models(generator, discriminator)

    z_dim = int(config["model"].get("z_dim", 128))
    model_name = config["model"].get("name", "dcgan")
    style_mixing_prob = float(config["model"].get("style_mixing_prob", 0.0))
    train_cfg = config["train"]
    criterion = nn.BCEWithLogitsLoss()
    opt_g, opt_d = create_optimizers(generator, discriminator, train_cfg)
    loss_name = train_cfg.get("loss", "bce")
    r1_gamma = float(train_cfg.get("r1_gamma", 0.0))
    r1_interval = int(train_cfg.get("r1_interval", 16))
    ema_beta_value = train_cfg.get("ema_beta", None)
    ema_beta = float(ema_beta_value) if ema_beta_value is not None else None
    ema_generator = create_ema_generator(generator, ema_beta)

    fixed_noise = torch.randn(64, z_dim, device=device)
    start_epoch = 1
    checkpoint_metrics: dict = {}
    if resume_from is not None:
        start_epoch, checkpoint_metrics = load_training_checkpoint(
            resume_from,
            generator,
            discriminator,
            opt_g,
            opt_d,
            config,
            device,
            ema_generator=ema_generator,
        )
        print(f"Resumed from {resume_from} at epoch {start_epoch - 1}.")

    best_fid = load_best_fid(log_path, checkpoint_metrics)
    evaluator = None
    start_time = time.time()
    print(f"Using device: {device}")
    print(f"Generator parameters: {count_parameters(generator):,}")
    print(f"Discriminator parameters: {count_parameters(discriminator):,}")
    if best_fid < float("inf"):
        print(f"Best historical FID: {best_fid:.4f}")
    if ema_generator is not None:
        print(f"Using generator EMA with beta={ema_beta:.4f}")

    total_epochs = int(train_cfg["epochs"])
    if start_epoch > total_epochs:
        print(
            f"Checkpoint already reached epoch {start_epoch - 1}; "
            f"target epochs is {total_epochs}. Nothing to train."
        )
        return

    global_step = (start_epoch - 1) * len(dataloader)

    for epoch in range(start_epoch, total_epochs + 1):
        generator.train()
        discriminator.train()
        epoch_g = 0.0
        epoch_d = 0.0
        epoch_r1 = 0.0
        r1_batches = 0
        batches = 0
        progress = tqdm(dataloader, desc=f"epoch {epoch}", leave=False)
        for real, _ in progress:
            real = real.to(device)
            batch_size = real.size(0)
            real_targets = torch.full((batch_size,), float(train_cfg["real_label"]), device=device)
            fake_targets = torch.full((batch_size,), float(train_cfg["fake_label"]), device=device)
            use_r1 = loss_name == "non_saturating" and r1_gamma > 0 and global_step % r1_interval == 0
            real_for_d = real.detach().requires_grad_(use_r1)

            opt_d.zero_grad(set_to_none=True)
            real_logits = discriminator(real_for_d)
            z = torch.randn(batch_size, z_dim, device=device)
            if model_name.startswith("stylegan_lite"):
                mixing_z = torch.randn(batch_size, z_dim, device=device)
                fake = generator(z, mixing_z=mixing_z, style_mixing_prob=style_mixing_prob)
            else:
                fake = generator(z)
            fake_logits = discriminator(fake.detach())
            if loss_name == "non_saturating":
                loss_d = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
                if use_r1:
                    r1_penalty = r1_regularization(real_logits, real_for_d)
                    loss_d = loss_d + r1_penalty * (r1_gamma * 0.5 * r1_interval)
                    epoch_r1 += r1_penalty.item()
                    r1_batches += 1
            else:
                loss_d_real = criterion(real_logits, real_targets)
                loss_d_fake = criterion(fake_logits, fake_targets)
                loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            fake_logits_for_g = discriminator(fake)
            if loss_name == "non_saturating":
                loss_g = F.softplus(-fake_logits_for_g).mean()
            else:
                loss_g = criterion(fake_logits_for_g, real_targets)
            loss_g.backward()
            opt_g.step()
            if ema_generator is not None and ema_beta is not None:
                update_ema(ema_generator, generator, ema_beta)

            epoch_g += loss_g.item()
            epoch_d += loss_d.item()
            batches += 1
            global_step += 1
            progress.set_postfix({"g": f"{loss_g.item():.3f}", "d": f"{loss_d.item():.3f}"})

        row = {
            "epoch": epoch,
            "loss_g": epoch_g / max(1, batches),
            "loss_d": epoch_d / max(1, batches),
            "elapsed_sec": round(time.time() - start_time, 2),
        }
        if r1_batches:
            row["r1"] = epoch_r1 / r1_batches

        if epoch % int(train_cfg.get("sample_every_epochs", 1)) == 0:
            sample_generator = ema_generator if ema_generator is not None else generator
            sample_generator.eval()
            with torch.no_grad():
                samples = _sample_generator(sample_generator, model_name, fixed_noise)
            save_image(denormalize(samples), sample_dir / f"epoch_{epoch:04d}.png", nrow=8)

        if epoch % int(config["eval"].get("compute_every_epochs", 10)) == 0:
            try:
                if evaluator is None:
                    evaluator = ImageMetricsEvaluator(
                        dataloader=eval_loader,
                        model_name=model_name,
                        z_dim=z_dim,
                        device=device,
                        num_images=int(config["eval"].get("num_images", 2048)),
                        batch_size=int(config["eval"].get("batch_size", 64)),
                        diversity_config=config["eval"].get("diversity", {}),
                    )
                eval_generator = ema_generator if ema_generator is not None else generator
                metrics = evaluator.evaluate(eval_generator)
                row.update(metrics)
                if metrics["fid"] < best_fid:
                    best_fid = metrics["fid"]
                    save_checkpoint(
                        ckpt_dir / "best_fid.pt",
                        epoch,
                        generator,
                        discriminator,
                        opt_g,
                        opt_d,
                        config,
                        metrics,
                        ema_generator=ema_generator,
                    )
            except RuntimeError as exc:
                row["eval_error"] = str(exc)

        if epoch % int(train_cfg.get("checkpoint_every_epochs", 1)) == 0:
            save_checkpoint(
                ckpt_dir / "last.pt",
                epoch,
                generator,
                discriminator,
                opt_g,
                opt_d,
                config,
                row,
                ema_generator=ema_generator,
            )

        append_jsonl(log_path, row)
        print(
            f"epoch={epoch} loss_g={row['loss_g']:.4f} loss_d={row['loss_d']:.4f}"
            + (f" fid={row['fid']:.4f}" if "fid" in row else "")
        )
