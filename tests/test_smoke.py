from PIL import Image
import pytest
import torch

from final_gan.data import build_transforms
from final_gan.factory import build_discriminator, build_generator, initialize_models
from final_gan.metrics import _require_torchmetrics


def test_transform_shape_and_range():
    image = Image.new("RGB", (128, 96), color=(128, 64, 32))
    tensor = build_transforms(64)(image)
    assert tensor.shape == (3, 64, 64)
    assert tensor.min().item() >= -1.0
    assert tensor.max().item() <= 1.0


@pytest.mark.parametrize("model_name", ["dcgan", "stylegan_lite"])
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
