#import "@preview/charged-ieee:0.1.4": ieee



#show: ieee.with(
    title: [基于 GAN 的人头图像生成与 StyleGAN-lite 稳定化改进],
    authors: (
        (
            name: "赵晨星",
            email: [524030910233],
        ),
    ),
    abstract: [
        本文围绕课程大作业中的人头图像生成任务，使用 PyTorch 实现并比较了 DCGAN、StyleGAN-lite 与稳定化改进后的 StyleGAN-lite-v2。DCGAN 作为基础生成模型，在 CelebA 上形成了稳定 baseline；初版 StyleGAN-lite 虽引入 mapping network、AdaIN、噪声注入和 style mixing，但训练中出现明显模式崩溃，最终 10000 张图评估 FID 为 100.91。针对该问题，本文进一步加入每分辨率双 StyledConv、minibatch standard deviation、non-saturating logistic loss、R1 regularization、TTUR 和 generator EMA。最终 StyleGAN-lite-v2 在 10000 张生成图像上达到 FID 12.78，优于 DCGAN 的 FID 24.16，并取得 MS-SSIM 0.1990、diversity score 0.8010，说明其在图像质量和多样性上均有明显改善。
    ],
    bibliography: bibliography("refs.bib"),
)

#set table(stroke: (x, y) => {
    if y == 0 {
        (top: (1pt + black), bottom: (0.5pt + black))
    } else {
        none
    }
})

= 引言

生成对抗网络（Generative Adversarial Network, GAN）通过生成器与判别器的对抗优化学习真实数据分布 @goodfellow2014gan。本文目标是完成基于 GAN 的人头图像生成系统，并按照课程要求实现基础 GAN 模型、生成样本展示、潜空间线性插值以及 FID/IS 评价。

本文采用 CelebA 作为主实验数据集。相比 LFW，CelebA 的样本规模更大，更适合训练 StyleGAN-lite 这类结构更复杂的生成器。实验按照“基础复现、失败诊断、稳定化改进、定量与定性验证”的顺序展开：首先实现 DCGAN baseline；随后尝试 StyleGAN-lite；最后针对其模式崩溃现象设计 StyleGAN-lite-v2，并验证改进效果。

#figure(
    image("assets/project_pipeline.png", width: 100%),
    caption: [项目整体实验流程。],
)

本文主要贡献如下：

- 实现了完整的 GAN 训练、采样、插值和评估流程。
- 复现 DCGAN baseline，并实现基于 StyleGAN 思想的轻量生成器。
- 基于训练现象分析 old StyleGAN-lite 的模式崩溃。
- 通过 R1、TTUR、EMA、minibatch stddev 和更完整的 synthesis blocks 显著提升 StyleGAN-lite 稳定性。

= 相关工作

GAN 原论文将生成建模表述为生成器 $G$ 与判别器 $D$ 的二人博弈。判别器学习区分真实样本与生成样本，生成器则学习欺骗判别器 @goodfellow2014gan。实践中，非饱和生成器目标常用于缓解早期梯度不足。

DCGAN 将 GAN 推广到卷积图像生成任务，提出用卷积和反卷积结构替代全连接网络，并使用 BatchNorm、ReLU/Tanh、LeakyReLU 以及 Adam 优化器 @radford2015dcgan。本文的 baseline 直接遵循这些设计原则。

StyleGAN 进一步将输入潜变量映射到中间潜空间，并通过 AdaIN 控制每层样式，同时使用 learned constant、per-layer noise 与 style mixing 提升生成质量和可控性 @karras2019stylegan。本文实现的是课程规模的 StyleGAN-lite，而非完整 StyleGAN 复现。CycleGAN 面向非配对图像翻译任务，核心是 cycle consistency @zhu2017cyclegan，因此本文仅将其作为不采用方案的参考。

#figure(
    image("assets/gan_training_diagram.png", width: 100%),
    caption: [GAN 对抗训练机制。],
)

= 数据集与预处理

主实验使用 CelebA aligned and cropped images。所有图像以 RGB 形式读取，经 resize/crop 得到 $64 times 64$ 输入，并归一化到 $[-1, 1]$。这一范围与生成器输出层的 Tanh 激活一致。

CelebA 官方 aligned 图像已经根据眼睛位置做过相似变换和裁剪，因此部分样本上边缘可能出现拉伸或扭曲痕迹。这是数据集对齐流程带来的边界现象，不是本文训练代码额外加入的噪声。

#figure(
    image("assets/evaluation_pipeline.png", width: 100%),
    caption: [真实图像统计、生成样本和评价指标的计算流程。],
)

= 方法

== DCGAN Baseline

DCGAN 生成器输入 $z in R^128$，首先经线性层投影到 $4 times 4 times 512$ 特征图，然后通过 4 个 ConvTranspose2d 上采样块生成 $64 times 64 times 3$ RGB 图像。中间层使用 BatchNorm 和 ReLU，输出层使用 Tanh。

判别器采用 stride-2 Conv2d 逐步下采样，通道数为 64、128、256、512。除输入层外使用 BatchNorm，激活函数为 LeakyReLU(0.2)，最后输出单个 logit。损失函数使用 `BCEWithLogitsLoss`，真实标签采用 0.9 的 one-sided label smoothing。

#figure(
    image("assets/dcgan_architecture.png", width: 100%),
    caption: [DCGAN 生成器结构。],
)

== StyleGAN-lite

StyleGAN-lite 引入 StyleGAN 的核心思想：mapping network 将 $z$ 映射为 $w$，生成器从 learned constant 开始，通过 StyledConv、AdaIN 和 noise injection 逐步生成图像。训练时开启 style mixing，用两个 latent 在随机层切换样式。

初版模型保持轻量化，每个分辨率使用较少卷积层，并继续使用与 DCGAN 同族的判别器。实验表明，这一设计虽然引入了 StyleGAN 的机制，但训练稳定性不足，生成结果存在重复脸和结构退化。

#figure(
    image("assets/stylegan_lite_architecture.png", width: 100%),
    caption: [StyleGAN-lite 初版生成器结构。],
)

== StyleGAN-lite-v2

StyleGAN-lite-v2 的目标不是复现完整 StyleGAN，而是在课程规模下修复初版模型暴露出的模式崩溃问题。其改动包括：

- 每个分辨率使用两个 StyledConv，使 synthesis network 更接近 StyleGAN 的分层生成设计。
- 将 style mixing probability 从 0.5 降为 0.1，降低训练早期扰动。
- 判别器末端加入 minibatch standard deviation layer，使其能感知批内多样性不足。
- 使用 non-saturating logistic loss、R1 regularization 和 TTUR。
- 使用 generator EMA，采样和评估默认采用 EMA 权重。

StyleGAN-lite-v2 的学习率设为 $lr_G=2e-4$、$lr_D=5e-5$。R1 regularization 的 $gamma=10$，每 16 step 施加一次。EMA 参数为 0.999。

#figure(
    image("assets/stylegan_lite_v2_architecture.png", width: 100%),
    caption: [StyleGAN-lite-v2 生成器结构。],
)

#figure(
    image("assets/improvement_path.png", width: 100%),
    caption: [从 old StyleGAN-lite 到 StyleGAN-lite-v2 的稳定化路径。],
)

= 训练与工程实现

训练流程由 YAML 配置驱动，提供 `train`、`generate`、`interpolate` 和 `evaluate` 四类 CLI 入口。训练过程中每个 epoch 保存固定噪声采样图、checkpoint 和 JSONL 日志。checkpoint 包含生成器、判别器、优化器、配置和指标；StyleGAN-lite-v2 额外保存 EMA generator。

#figure(
    image("assets/project_pipeline.png", width: 100%),
    caption: [训练、采样、插值与评估的工程流程。],
)

#figure(
    table(
        columns: (auto, auto, auto, auto),
        align: horizon,
        inset: 5pt,
        table.header([模型], [Epochs], [学习率], [关键设置]),
        [DCGAN], [100], [`2e-4`], [BCE loss, label smoothing],
        [StyleGAN-lite], [100], [`2e-4`], [AdaIN, noise, style mixing 0.5],
        [StyleGAN-lite-v2], [150], [`G:2e-4, D:5e-5`], [R1, TTUR, EMA, minibatch stddev],
        table.hline()
    ),
    caption: [模型训练超参数对比。],
)

为提高实验效率，评估器在训练过程中缓存真实图像的 FID statistics，后续周期评估只需重新生成 fake samples。数据加载使用 `num_workers=4`、`prefetch_factor=2` 和 persistent workers。单独 evaluate 命令会将结果保存为 JSON，便于报告复现。

= 评估协议

FID 是本文主指标。它在 Inception feature 空间比较真实图像和生成图像的均值与协方差，数值越低代表生成分布越接近真实分布 @heusel2017ttur。IS 作为辅助指标报告，但由于 ImageNet 类别与人脸质量或身份多样性并不直接对应，本文不将 IS 作为主要判断依据。

对于多样性，StyleGAN-lite-v2 额外计算 MS-SSIM @wang2003msssim。MS-SSIM 越低表示图像之间结构越不相似，因此本文同时报告 $1 - "MS-SSIM"$ 作为 diversity score。

#figure(
    table(
        columns: (auto, auto, auto, auto),
        align: horizon,
        inset: 5pt,
        table.header([模型], [Checkpoint], [生成图数量], [指标]),
        [DCGAN], [`best_fid.pt`], [10000], [FID, IS],
        [StyleGAN-lite], [`best_fid.pt`], [10000], [FID, IS],
        [StyleGAN-lite-v2], [`best_fid.pt`], [10000], [FID, IS, MS-SSIM],
        table.hline()
    ),
    supplement: [表],
    caption: [最终评估协议。],
)

= 实验结果

== 定量结果

最终 10000 张生成图像评估结果如表所示。StyleGAN-lite-v2 在 FID 上显著优于 DCGAN 和 old StyleGAN-lite。

#figure(
    table(
        columns: (auto, auto, auto, auto, auto, auto),
        align: horizon,
        inset: 5pt,
        table.header([模型], [生成图], [FID ↓], [IS ↑], [MS-SSIM ↓], [Diversity ↑]),
        [DCGAN], [10000], [24.1616], [2.4636 ± 0.0572], [N/A], [N/A],
        [StyleGAN-lite], [10000], [100.9125], [1.8161 ± 0.0131], [N/A], [N/A],
        [StyleGAN-lite-v2], [10000], [*12.7835*], [2.4901 ± 0.0805], [0.1990], [0.8010],
        table.hline()
    ),
    caption: [三种模型在 10000 张生成图像上的最终评估结果。],
)

#figure(
    image("assets/fid_curve.png", width: 100%),
    caption: [训练过程中 FID 变化曲线。],
)

#figure(
    image("assets/final_fid_bar.png", width: 100%),
    caption: [三种模型最终 FID 对比，均使用 10000 张生成图像。],
)

#figure(
    image("assets/final_is_bar.png", width: 100%),
    caption: [三种模型最终 Inception Score 对比。],
)

训练期结果进一步说明了模型差异。DCGAN 在第 40 epoch 达到训练期最佳 FID 33.28，之后有所震荡；old StyleGAN-lite 在第 20 epoch 达到 FID 106.13，但随后未能持续改善；StyleGAN-lite-v2 从 early stage 开始稳定下降，到第 150 epoch 训练期 FID 为 21.76。

#figure(
    image("assets/loss_curves.png", width: 100%),
    caption: [三种模型的生成器与判别器 loss 曲线。],
)

#figure(
    image("assets/v2_diversity_curve.png", width: 100%),
    caption: [StyleGAN-lite-v2 的 MS-SSIM 与 diversity score。],
)

old StyleGAN-lite 后期 `loss_g` 升高到约 5.22，`loss_d` 降至约 0.36，说明判别器相对生成器过强。StyleGAN-lite-v2 采用 non-saturating loss 后，loss 数值尺度与 BCE 路径不同，不能直接和 DCGAN 数值比较；更重要的是其 FID 曲线持续下降，且 diversity score 保持在 0.78 到 0.80 附近，没有出现严重重复脸。

== 定性结果

DCGAN 能生成较稳定的人脸结构，但局部纹理和细节仍有限。old StyleGAN-lite 的样本网格中可观察到重复脸、局部结构混乱和质量不稳定。StyleGAN-lite-v2 的面部结构、颜色一致性和样本多样性更好，但仍存在少量边缘、发际线或局部五官错乱。

#figure(
    image("assets/sample_grid_comparison.png", width: auto),
    caption: [DCGAN、old StyleGAN-lite 与 StyleGAN-lite-v2 的采样结果对比。],
)

#figure(
    image("assets/stylegan_lite_failure_epoch_0020.png", width: 100%),
    caption: [old StyleGAN-lite 在最佳 FID 附近的样本。],
)

#figure(
    image("assets/stylegan_lite_failure_epoch_0100.png", width: 100%),
    caption: [old StyleGAN-lite 后期样本，重复与退化现象更加明显。],
)

== 潜空间插值

线性插值用于检查潜空间的连续性。DCGAN 在 $z$ 空间插值时能够产生逐渐变化的人脸。StyleGAN-lite-v2 支持 $z$ 和 $w$ 两种空间插值，其中 $w$ 是 mapping network 后的中间潜空间。在 StyleGAN 思想中，$w$ 空间通常更利于样式解耦，因此本文同时展示两类插值结果。

#figure(
    image("assets/interpolation_comparison.png", width: 100%),
    caption: [DCGAN $z$ 空间插值、StyleGAN-lite-v2 $z$ 空间插值与 $w$ 空间插值。],
)

= 讨论

old StyleGAN-lite 的失败表明，在小规模轻量实现中，仅加入 mapping network、AdaIN、noise injection 和 style mixing 并不足以保证训练稳定。StyleGAN 原论文中的优势来自一整套结构和训练策略，而不是单个模块。本文观察到 old StyleGAN-lite 中判别器 loss 持续降低、生成器 loss 持续升高，并伴随重复脸，说明训练进入判别器占优和模式崩溃状态。

StyleGAN-lite-v2 的改进针对这些现象展开。R1 regularization 限制判别器在真实图像附近的梯度，TTUR 降低判别器学习率，EMA 平滑生成器权重，minibatch stddev 帮助判别器感知批内多样性不足，双 StyledConv 则增强每个分辨率的 synthesis 表达能力。最终 FID 从 old StyleGAN-lite 的 100.91 降到 12.78，并且优于 DCGAN 的 24.16。

本文仍有局限。首先，所有实验均在 $64 times 64$ 分辨率进行，不能代表高分辨率人脸生成质量。其次，StyleGAN-lite-v2 不是完整 StyleGAN 复现，没有使用完整官方训练配置。第三，IS 对人脸生成任务解释力有限。最后，MS-SSIM 只衡量结构相似性，未来可加入 LPIPS 作为感知多样性指标。

#figure(
    image("assets/metrics_dashboard.png", width: 100%),
    caption: [训练和最终评估指标总览。],
)

= 结论

本文完成了基于 GAN 的人头图像生成课程项目，实现了 DCGAN baseline、StyleGAN-lite 和 StyleGAN-lite-v2，并构建了训练、采样、插值与评估的完整 Python/PyTorch 流程。实验表明，DCGAN 是稳定但表达能力有限的 baseline；old StyleGAN-lite 出现明显模式崩溃；StyleGAN-lite-v2 通过结构和训练稳定化改进，在 10000 张生成图像评估下取得 FID 12.78，显著优于 DCGAN 和 old StyleGAN-lite。该结果说明，针对训练现象进行有依据的稳定化设计，是课程规模 GAN 项目中提升生成质量和多样性的有效路径。

= 附录：复现命令

训练 DCGAN：

```powershell
E:\Python_env\AI\Scripts\python.exe run.py train --config configs/celeba_dcgan.yaml
```

训练 StyleGAN-lite-v2：

```powershell
E:\Python_env\AI\Scripts\python.exe run.py train --config configs/celeba_stylegan_lite_v2.yaml
```

断点续训：

```powershell
E:\Python_env\AI\Scripts\python.exe run.py train --config configs/celeba_stylegan_lite_v2.yaml --resume runs/stylegan_lite_v2_celeba/checkpoints/last.pt
```

最终评估：

```powershell
E:\Python_env\AI\Scripts\python.exe run.py evaluate --checkpoint runs/stylegan_lite_v2_celeba/checkpoints/best_fid.pt --data-root data/celeba --num-images 10000
```

生成报告图：

```powershell
E:\Python_env\AI\Scripts\python.exe scripts\make_report_figures.py
```
