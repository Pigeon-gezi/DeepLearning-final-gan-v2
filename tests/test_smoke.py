from PIL import Image
import pytest
import torch
from pathlib import Path
from uuid import uuid4

from final_gan.data import build_transforms
from final_gan.factory import build_discriminator, build_generator, initialize_models
from final_gan.metrics import _require_torchmetrics, compute_diversity_metrics
from final_gan.training import (
    create_ema_generator,
    create_optimizers,
    load_training_checkpoint,
    r1_regularization,
    save_checkpoint,
    train,
    update_ema,
)


def test_transform_shape_and_range():
    image = Image.new("RGB", (128, 96), color=(128, 64, 32))
    tensor = build_transforms(64)(image)
    assert tensor.shape == (3, 64, 64)
    assert tensor.min().item() >= -1.0
    assert tensor.max().item() <= 1.0


@pytest.mark.parametrize("model_name", ["dcgan", "stylegan_lite", "stylegan_lite_v2"])
def test_generator_and_discriminator_shapes(model_name):
    config = {"model": {"name": model_name, "z_dim": 128, "w_dim": 256}}
    generator = build_generator(config)
    discriminator = build_discriminator(config)
    z = torch.randn(2, 128)
    with torch.no_grad():
        fake = generator(z, style_mixing_prob=0.0) if model_name == "stylegan_lite" else generator(z)
        logits = discriminator(fake)
    assert fake.shape == (2, 3, 64, 64)
    assert fake.min().item() >= -1.0
    assert fake.max().item() <= 1.0
    assert logits.shape == (2,)


def test_stylegan_v2_uses_separate_learning_rates_and_ema():
    config = {
        "model": {"name": "stylegan_lite_v2", "z_dim": 128, "w_dim": 256},
        "train": {"lr": 0.0002, "lr_g": 0.0002, "lr_d": 0.00005, "beta1": 0.5, "beta2": 0.999},
    }
    generator = build_generator(config)
    discriminator = build_discriminator(config)
    opt_g, opt_d = create_optimizers(generator, discriminator, config["train"])
    assert opt_g.param_groups[0]["lr"] == pytest.approx(0.0002)
    assert opt_d.param_groups[0]["lr"] == pytest.approx(0.00005)

    ema = create_ema_generator(generator, 0.999)
    assert ema is not None
    before = next(ema.parameters()).detach().clone()
    with torch.no_grad():
        for param in generator.parameters():
            param.add_(0.1)
            break
    update_ema(ema, generator, 0.5)
    after = next(ema.parameters()).detach()
    assert not torch.equal(before, after)


def test_r1_regularization_backward_on_stylegan_v2_discriminator():
    config = {"model": {"name": "stylegan_lite_v2", "z_dim": 128, "w_dim": 256}}
    discriminator = build_discriminator(config)
    real = torch.randn(2, 3, 64, 64, requires_grad=True)
    logits = discriminator(real)
    penalty = r1_regularization(logits, real)
    penalty.backward()
    assert penalty.item() >= 0
    assert real.grad is not None


def test_one_training_step_updates_parameters():
    config = {"model": {"name": "dcgan", "z_dim": 128}}
    generator = build_generator(config)
    discriminator = build_discriminator(config)
    initialize_models(generator, discriminator)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = torch.nn.BCEWithLogitsLoss()
    real = torch.randn(2, 3, 64, 64).clamp(-1, 1)
    z = torch.randn(2, 128)

    before = next(generator.parameters()).detach().clone()
    opt_d.zero_grad(set_to_none=True)
    fake = generator(z)
    loss_d = criterion(discriminator(real), torch.ones(2)) + criterion(discriminator(fake.detach()), torch.zeros(2))
    loss_d.backward()
    opt_d.step()

    opt_g.zero_grad(set_to_none=True)
    loss_g = criterion(discriminator(fake), torch.ones(2))
    loss_g.backward()
    opt_g.step()

    after = next(generator.parameters()).detach()
    assert not torch.equal(before, after)


def test_metrics_dependency_message_or_import():
    try:
        fid_cls, is_cls = _require_torchmetrics()
    except RuntimeError as exc:
        assert "pip install -r requirements.txt" in str(exc)
    else:
        assert fid_cls is not None
        assert is_cls is not None


def test_ms_ssim_diversity_metric_runs_on_64px_images():
    images = torch.rand(4, 3, 64, 64)
    result = compute_diversity_metrics(
        images,
        {
            "metric": "ms_ssim",
            "max_pairs": 4,
            "pair_batch_size": 2,
            "ms_ssim_betas": [0.3, 0.3, 0.4],
        },
        torch.device("cpu"),
        show_progress=False,
    )
    assert "diversity_ms_ssim" in result
    assert "diversity_score" in result
    assert 0.0 <= result["diversity_ms_ssim"] <= 1.0


def _make_test_dir():
    root = Path(".test_tmp") / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def _tiny_train_config(root, epochs):
    image_dir = root / "images"
    image_dir.mkdir()
    for idx in range(2):
        Image.new("RGB", (80, 80), color=(idx * 80, 64, 128)).save(image_dir / f"{idx:03d}.jpg")
    return {
        "seed": 123,
        "device": "cpu",
        "data": {
            "root": str(image_dir),
            "dataset": "recursive_images",
            "image_size": 64,
            "num_workers": 0,
            "prefetch_factor": 2,
            "persistent_workers": False,
        },
        "model": {"name": "dcgan", "z_dim": 128},
        "train": {
            "epochs": epochs,
            "batch_size": 2,
            "lr": 0.0002,
            "beta1": 0.5,
            "beta2": 0.999,
            "real_label": 0.9,
            "fake_label": 0.0,
            "sample_every_epochs": 100,
            "checkpoint_every_epochs": 1,
        },
        "eval": {"compute_every_epochs": 100, "num_images": 2, "batch_size": 2},
        "paths": {"output_dir": str(root / "run")},
    }


def test_resume_training_starts_after_checkpoint_epoch():
    root = _make_test_dir()
    config = _tiny_train_config(root, epochs=1)
    train(config)
    checkpoint = root / "run" / "checkpoints" / "last.pt"
    assert checkpoint.exists()
    first = torch.load(checkpoint, map_location="cpu")
    assert first["epoch"] == 1
    assert "opt_g" in first and "opt_d" in first

    config["train"]["epochs"] = 2
    train(config, resume_from=checkpoint)
    second = torch.load(checkpoint, map_location="cpu")
    assert second["epoch"] == 2


def test_resume_rejects_model_mismatch():
    dcgan_config = {"model": {"name": "dcgan", "z_dim": 128}}
    generator = build_generator(dcgan_config)
    discriminator = build_discriminator(dcgan_config)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    checkpoint = _make_test_dir() / "mismatch.pt"
    save_checkpoint(checkpoint, 1, generator, discriminator, opt_g, opt_d, dcgan_config)

    style_config = {"model": {"name": "stylegan_lite", "z_dim": 128, "w_dim": 256}}
    style_generator = build_generator(style_config)
    style_discriminator = build_discriminator(style_config)
    style_opt_g = torch.optim.Adam(style_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    style_opt_d = torch.optim.Adam(style_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    with pytest.raises(RuntimeError, match="does not match"):
        load_training_checkpoint(
            checkpoint,
            style_generator,
            style_discriminator,
            style_opt_g,
            style_opt_d,
            style_config,
            torch.device("cpu"),
        )
