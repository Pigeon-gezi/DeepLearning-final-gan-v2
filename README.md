# LFW Face Generation with GANs

This project implements a course-ready GAN pipeline for human head image generation on LFW:

- DCGAN baseline.
- StyleGAN-lite bonus model with mapping network, AdaIN, noise injection, and style mixing.
- Latent interpolation in `z` space, plus `w` space for StyleGAN-lite.
- FID as the main metric and Inception Score as a secondary metric.

## Environment

Use the virtual environment Python provided for this project:

```powershell
E:\Python_env\AI\Scripts\python.exe -m pip install -r requirements.txt
```

`torchmetrics` and `torch-fidelity` are required only for FID/IS evaluation. Training, generation, and interpolation use PyTorch and torchvision.

## Dataset

Place LFW images under `data/lfw-py`, or pass `--data-root`. The default loader recursively scans image files, so both of these layouts work:

```text
data/lfw-py/Person_Name/image.jpg
data/lfw-py/image.jpg
```

Torchvision no longer supports automatic LFW download, so the dataset must be prepared manually.

## Train

DCGAN:

```powershell
E:\Python_env\AI\Scripts\python.exe run.py train --config configs/lfw_dcgan.yaml
```

StyleGAN-lite:

```powershell
E:\Python_env\AI\Scripts\python.exe run.py train --config configs/lfw_stylegan_lite.yaml
```

Outputs are written to `runs/<experiment>`:

- `samples/epoch_XXXX.png`
- `checkpoints/last.pt`
- `checkpoints/best_fid.pt` when FID is computed successfully
- `train_log.jsonl`
- `resolved_config.yaml`

## Generate and Interpolate

Generate a sample grid:

```powershell
E:\Python_env\AI\Scripts\python.exe run.py generate --checkpoint runs/dcgan_lfw/checkpoints/last.pt --output runs/dcgan_lfw/generated.png
```

Interpolate in latent space:

```powershell
E:\Python_env\AI\Scripts\python.exe run.py interpolate --checkpoint runs/dcgan_lfw/checkpoints/last.pt --output-dir runs/dcgan_lfw/interpolations --space z
```

For StyleGAN-lite, `--space w` is also supported.

## Evaluate

```powershell
E:\Python_env\AI\Scripts\python.exe run.py evaluate --checkpoint runs/dcgan_lfw/checkpoints/last.pt --data-root data/lfw-py --num-images 2048
```

FID compares generated samples with real LFW images. Inception Score is reported for completeness, but it is less meaningful for LFW faces because ImageNet classes are not aligned with face identity or image quality.

## Implementation Notes

The implementation follows the course plan and the referenced papers:

- GAN: adversarial training between generator and discriminator.
- DCGAN: convolutional generator/discriminator, BatchNorm, ReLU/Tanh generator, LeakyReLU discriminator, Adam with `lr=0.0002` and `beta1=0.5`.
- StyleGAN-lite: compact version of mapping network, AdaIN, stochastic noise, and style mixing for a feasible course-scale comparison.
