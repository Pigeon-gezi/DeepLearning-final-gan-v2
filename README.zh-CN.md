# 基于 GAN 的人脸图像生成

本项目实现了一套面向课程大作业的人头图像生成 GAN 流程：

- DCGAN 基线模型。
- StyleGAN-lite：包含 mapping network、AdaIN、噪声注入和 style mixing。
- StyleGAN-lite-v2：稳定化改进版本，包含 minibatch stddev 判别器、non-saturating loss、R1 regularization、TTUR 风格学习率和 generator EMA。
- 支持 `z` 空间潜变量插值；StyleGAN 系列模型额外支持 `w` 空间插值。
- 支持 FID、Inception Score，以及可选的 MS-SSIM 多样性评估。

## 环境配置

本项目使用指定虚拟环境中的 Python：

```powershell
E:\Python_env\AI\Scripts\python.exe -m pip install -r requirements.txt
```

FID / IS / MS-SSIM 评估依赖 `torchmetrics` 和 `torch-fidelity`。训练、生成和插值依赖 PyTorch 与 torchvision。

## 数据集

默认数据加载器会递归扫描图像文件，因此平铺目录和嵌套目录都可以使用：

```text
data/celeba/000001.jpg
data/lfw/Person_Name/image.jpg
```

torchvision 已不再支持自动下载 LFW，因此数据集需要手动准备。可以将数据放在 `data/<name>` 下，也可以通过 `--data-root` 覆盖数据路径。

## 配置文件

当前可用实验配置：

```text
configs/lfw_dcgan.yaml
configs/lfw_stylegan_lite.yaml
configs/celeba_dcgan.yaml
configs/celeba_stylegan_lite.yaml
configs/celeba_stylegan_lite_v2.yaml
configs/evaluation.yaml
```

`celeba_stylegan_lite_v2.yaml` 启用稳定化 StyleGAN-lite-v2 路径，并支持 MS-SSIM 多样性日志。`evaluation.yaml` 用于统一最终评估设置，例如对所有模型启用 MS-SSIM。

## 训练

训练 CelebA 上的 DCGAN：

```powershell
E:\Python_env\AI\Scripts\python.exe run.py train --config configs/celeba_dcgan.yaml
```

训练 CelebA 上的 StyleGAN-lite-v2：

```powershell
E:\Python_env\AI\Scripts\python.exe run.py train --config configs/celeba_stylegan_lite_v2.yaml
```

覆盖数据集路径或输出目录：

```powershell
E:\Python_env\AI\Scripts\python.exe run.py train --config configs/celeba_dcgan.yaml --data-root data/celeba --output-dir runs/dcgan_celeba
```

从 checkpoint 断点续训。如果 checkpoint 已经达到旧配置中的目标 epoch，需要先增大配置文件里的 `train.epochs`：

```powershell
E:\Python_env\AI\Scripts\python.exe run.py train --config configs/celeba_stylegan_lite_v2.yaml --resume runs/stylegan_lite_v2_celeba/checkpoints/last.pt
```

训练输出会写入 `runs/<experiment>`：

- `samples/epoch_XXXX.png`
- `checkpoints/last.pt`
- `checkpoints/best_fid.pt`，当 FID 变好时更新
- `train_log.jsonl`
- `resolved_config.yaml`

checkpoint 中包含生成器权重、判别器权重、优化器状态、配置、指标；如果启用 EMA，还会包含 `generator_ema`。

## 生成图像

生成采样网格：

```powershell
E:\Python_env\AI\Scripts\python.exe run.py generate --checkpoint runs/stylegan_lite_v2_celeba/checkpoints/best_fid.pt --output runs/stylegan_lite_v2_celeba/best_samples.png
```

可选参数：

```text
--num-images 64
--nrow 8
--device auto
```

如果 checkpoint 中包含 `generator_ema`，生成默认使用 EMA 权重。

## 潜空间插值

在 `z` 空间插值：

```powershell
E:\Python_env\AI\Scripts\python.exe run.py interpolate --checkpoint runs/dcgan_celeba/checkpoints/best_fid.pt --output-dir runs/dcgan_celeba/interpolations --space z
```

StyleGAN 系列模型还支持 `w` 空间插值：

```powershell
E:\Python_env\AI\Scripts\python.exe run.py interpolate --checkpoint runs/stylegan_lite_v2_celeba/checkpoints/best_fid.pt --output-dir runs/stylegan_lite_v2_celeba/interpolations_w --space w
```

可选参数：

```text
--pairs 2
--steps 11
--device auto
```

## 评估

从 checkpoint 计算 FID 和 Inception Score：

```powershell
E:\Python_env\AI\Scripts\python.exe run.py evaluate --checkpoint runs/stylegan_lite_v2_celeba/checkpoints/best_fid.pt --data-root data/celeba --num-images 2048
```

使用统一评估配置覆盖 checkpoint 中的评估设置，例如对所有模型启用 MS-SSIM 多样性：

```powershell
E:\Python_env\AI\Scripts\python.exe run.py evaluate --checkpoint runs/dcgan_celeba/checkpoints/best_fid.pt --config configs/evaluation.yaml --data-root data/celeba
```

可选参数：

```text
--config <yaml>
--data-root <path>
--num-images 2048
--device auto
--output <json>
```

评估会从 checkpoint 读取模型和数据设置。如果提供 `--config`，只会合并其中的 `eval` 部分作为覆盖；模型结构和训练设置仍使用 checkpoint 中保存的配置。评估阶段会显示真实图像 FID 统计、生成样本和可选多样性评估的进度条。训练中的评估器会缓存真实图像 FID statistics，因此同一次训练后续评估只需要重新生成 fake samples。

单次评估会在终端打印指标，并保存为 JSON。如果没有提供 `--output`，结果会写到实验目录下：

```text
runs/<experiment>/evaluations/<checkpoint>_n<num_images>_<timestamp>.json
```

FID 比较生成图像和真实图像在 Inception 特征空间中的分布差异。Inception Score 也会报告，但对于纯人脸数据集解释力有限，因为 ImageNet 类别并不直接对应人脸身份、多样性或图像质量。

## 多样性评估

可以在配置文件中启用多样性评估：

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

`diversity_ms_ssim` 是生成样本之间的成对 MS-SSIM。数值越低，表示生成图之间结构越不相似，通常意味着多样性更高。为了更直观阅读，项目同时报告 `diversity_score = 1 - diversity_ms_ssim`。

多样性评估接口由配置驱动，因此后续可以继续接入 LPIPS 等指标，而不需要改变训练命令。

## CLI 汇总

```powershell
E:\Python_env\AI\Scripts\python.exe run.py train --config <yaml> [--data-root <path>] [--output-dir <path>] [--resume <checkpoint>]
E:\Python_env\AI\Scripts\python.exe run.py generate --checkpoint <pt> --output <png> [--num-images 64] [--nrow 8] [--device auto]
E:\Python_env\AI\Scripts\python.exe run.py interpolate --checkpoint <pt> --output-dir <dir> [--pairs 2] [--steps 11] [--space z|w] [--device auto]
E:\Python_env\AI\Scripts\python.exe run.py evaluate --checkpoint <pt> [--config <yaml>] [--data-root <path>] [--num-images 2048] [--device auto] [--output <json>]
```

## 实现说明

实现遵循课程计划和相关论文设计：

- GAN：生成器和判别器进行对抗训练。
- DCGAN：卷积生成器和判别器；生成器使用 BatchNorm、ReLU/Tanh；判别器使用 LeakyReLU；Adam 默认 `lr=0.0002`、`beta1=0.5`。
- StyleGAN-lite：轻量实现 mapping network、AdaIN、随机噪声注入和 style mixing。
- StyleGAN-lite-v2：课程规模的稳定化版本，使用 non-saturating loss、R1 regularization、generator EMA、minibatch stddev、较低的 style mixing probability，以及每个分辨率成对 StyledConv。
