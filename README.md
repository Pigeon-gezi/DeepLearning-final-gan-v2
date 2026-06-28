# Face Generation with GANs

This project implements a course-ready GAN pipeline for human head image generation:

- DCGAN baseline.
- StyleGAN-lite model with mapping network, AdaIN, noise injection, and style mixing.
- StyleGAN-lite-v2 stabilized model with minibatch stddev discriminator, non-saturating loss, R1 regularization, TTUR-style learning rates, and generator EMA.
- Latent interpolation in `z` space, plus `w` space for StyleGAN-style models.
- FID, Inception Score, and optional MS-SSIM diversity evaluation.

## Environment

Use the virtual environment Python provided for this project:

```powershell
E:\Python_env\AI\Scripts\python.exe -m pip install -r requirements.txt
```

`torchmetrics` and `torch-fidelity` are required for FID/IS/MS-SSIM evaluation. Training, generation, and interpolation use PyTorch and torchvision.

## Dataset

The default loader recursively scans image files, so both flat and nested layouts work:

```text
data/celeba/000001.jpg
data/lfw/Person_Name/image.jpg
```

Torchvision no longer supports automatic LFW download, so datasets should be prepared manually. You can either place data under `data/<name>` or override the path with `--data-root`.

## Configs

Available experiment configs:

```text
configs/lfw_dcgan.yaml
configs/lfw_stylegan_lite.yaml
configs/celeba_dcgan.yaml
configs/celeba_stylegan_lite.yaml
configs/celeba_stylegan_lite_v2.yaml
```

`celeba_stylegan_lite_v2.yaml` enables the stabilized StyleGAN-lite-v2 path and MS-SSIM diversity logging.

## Train

Train DCGAN on CelebA:

```powershell
E:\Python_env\AI\Scripts\python.exe run.py train --config configs/celeba_dcgan.yaml
```

Train StyleGAN-lite-v2 on CelebA:

```powershell
E:\Python_env\AI\Scripts\python.exe run.py train --config configs/celeba_stylegan_lite_v2.yaml
```

Override dataset or output directory:

```powershell
E:\Python_env\AI\Scripts\python.exe run.py train --config configs/celeba_dcgan.yaml --data-root data/celeba --output-dir runs/dcgan_celeba
```

Resume from a checkpoint. Increase `train.epochs` in the config first if the checkpoint has already reached the old target epoch:

```powershell
E:\Python_env\AI\Scripts\python.exe run.py train --config configs/celeba_stylegan_lite_v2.yaml --resume runs/stylegan_lite_v2_celeba/checkpoints/last.pt
```

Outputs are written to `runs/<experiment>`:

- `samples/epoch_XXXX.png`
- `checkpoints/last.pt`
- `checkpoints/best_fid.pt` when FID improves
- `train_log.jsonl`
- `resolved_config.yaml`

Checkpoints include model weights, discriminator weights, optimizer states, config, metrics, and `generator_ema` when EMA is enabled.

## Generate

Generate a sample grid:

```powershell
E:\Python_env\AI\Scripts\python.exe run.py generate --checkpoint runs/stylegan_lite_v2_celeba/checkpoints/best_fid.pt --output runs/stylegan_lite_v2_celeba/best_samples.png
```

Optional arguments:

```text
--num-images 64
--nrow 8
--device auto
```

If a checkpoint contains `generator_ema`, generation uses EMA weights by default.

## Interpolate

Interpolate in `z` space:

```powershell
E:\Python_env\AI\Scripts\python.exe run.py interpolate --checkpoint runs/dcgan_celeba/checkpoints/best_fid.pt --output-dir runs/dcgan_celeba/interpolations --space z
```

For StyleGAN-style models, `w` space interpolation is also supported:

```powershell
E:\Python_env\AI\Scripts\python.exe run.py interpolate --checkpoint runs/stylegan_lite_v2_celeba/checkpoints/best_fid.pt --output-dir runs/stylegan_lite_v2_celeba/interpolations_w --space w
```

Optional arguments:

```text
--pairs 2
--steps 11
--device auto
```

## Evaluate

Compute FID and Inception Score from a checkpoint:

```powershell
E:\Python_env\AI\Scripts\python.exe run.py evaluate --checkpoint runs/stylegan_lite_v2_celeba/checkpoints/best_fid.pt --data-root data/celeba --num-images 2048
```

Use a shared evaluation config to override checkpoint evaluation settings, such as enabling MS-SSIM diversity for all models:

```powershell
E:\Python_env\AI\Scripts\python.exe run.py evaluate --checkpoint runs/dcgan_celeba/checkpoints/best_fid.pt --config configs/evaluation.yaml --data-root data/celeba
```

Optional arguments:

```text
--config <yaml>
--data-root <path>
--num-images 2048
--device auto
--output <json>
```

Evaluation loads model/data settings from the checkpoint. If `--config` is provided, only its `eval` section is merged as an override; model architecture and training settings remain those stored in the checkpoint. Evaluation shows progress bars for real FID statistics, fake samples, and optional diversity pairs. Real FID statistics are cached inside the evaluator during training, so later evaluations in the same run only recompute generated samples.

Single-run evaluation prints metrics to the terminal and saves them as JSON. If `--output` is omitted, results are written under the experiment directory:

```text
runs/<experiment>/evaluations/<checkpoint>_n<num_images>_<timestamp>.json
```

FID compares generated samples with real images. Inception Score is reported for completeness, but it is less meaningful for face-only datasets because ImageNet classes are not aligned with face identity or image quality.

## Diversity

Optional diversity evaluation can be enabled in a config:

```yaml
eval:
  diversity:
    enabled: true
    metric: ms_ssim
    num_images: 512
    max_pairs: 512
    pair_batch_size: 64
    ms_ssim_betas: [0.3, 0.3, 0.4]
```

`diversity_ms_ssim` is pairwise MS-SSIM among generated samples. Lower values mean generated images are less structurally similar and usually more diverse. `diversity_score` is reported as `1 - diversity_ms_ssim` for easier reading.

The diversity interface is config-driven so another metric, such as LPIPS, can be added later without changing training commands.

## CLI Summary

```powershell
E:\Python_env\AI\Scripts\python.exe run.py train --config <yaml> [--data-root <path>] [--output-dir <path>] [--resume <checkpoint>]
E:\Python_env\AI\Scripts\python.exe run.py generate --checkpoint <pt> --output <png> [--num-images 64] [--nrow 8] [--device auto]
E:\Python_env\AI\Scripts\python.exe run.py interpolate --checkpoint <pt> --output-dir <dir> [--pairs 2] [--steps 11] [--space z|w] [--device auto]
E:\Python_env\AI\Scripts\python.exe run.py evaluate --checkpoint <pt> [--config <yaml>] [--data-root <path>] [--num-images 2048] [--device auto] [--output <json>]
```

## Implementation Notes

The implementation follows the course plan and the referenced papers:

- GAN: adversarial training between generator and discriminator.
- DCGAN: convolutional generator/discriminator, BatchNorm, ReLU/Tanh generator, LeakyReLU discriminator, Adam with `lr=0.0002` and `beta1=0.5`.
- StyleGAN-lite: compact version of mapping network, AdaIN, stochastic noise, and style mixing.
- StyleGAN-lite-v2: stabilized course-scale variant using non-saturating loss, R1 regularization, generator EMA, minibatch stddev, reduced style mixing probability, and per-resolution paired styled convolutions.
