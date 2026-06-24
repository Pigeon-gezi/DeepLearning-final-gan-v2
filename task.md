# 任务D：基于GAN的人头图像生成

## 背景

生成对抗网络（GAN）是一种强大的生成模型，通过生成器与判别器的对抗训练，能够生成高质量的数据样本。本项目旨在利用GAN模型实现人头图像的生成，探索其在图像生成领域的应用。

## 项目目标

理解GAN的基本原理，包括生成器与判别器的对抗训练过程。
复现一个基于GAN的人头图像生成系统。
在选定的数据集上训练模型，并生成高质量的人头图像。

## 说明

模型选择：可以参考以下GAN模型或其改进版本：
DCGAN（Deep Convolutional GAN）：适用于图像生成的经典GAN模型。
StyleGAN：能够生成高质量且多样化的图像。
CycleGAN：适用于图像风格迁移任务，可尝试生成特定风格的人头图像。

## 数据集：可以使用以下公开的人脸数据集

CelebA：包含超过20万张名人头像的图像数据集。大小1.3G
LFW（Labeled Faces in the Wild）：包含超过1.3万张人脸图像，适用于小规模实验。大小118M
评估指标：使用以下指标评估生成图像的质量：
FID（Fréchet Inception Distance）：衡量生成图像与真实图像之间的分布距离。
IS（Inception Score）：衡量生成图像的多样性与清晰度。

## 基本要求

实现基础的GAN模型（如DCGAN）并进行人头图像生成。
在选定数据集上训练模型，生成高质量的人头图像。
测试两张人头图像之间线性插值的一系列结果。
使用FID或IS评估生成图像的质量。

## Bonus任务

对比改进的GAN模型（如StyleGAN或CycleGAN）与基础模型的性能差异。
探索解决GAN训练中的模式崩溃问题，并提出改进方法。

## 参考资料与论文

论文“Generative Adversarial Nets” [arXiv:1406.2661](https://ar5iv.labs.arxiv.org/html/1406.2661)。
DCGAN：论文“Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks” [arXiv:1511.06434](https://ar5iv.labs.arxiv.org/html/1511.06434)。
StyleGAN：论文“A Style-Based Generator Architecture for Generative Adversarial Networks” [arXiv:1812.04948](https://ar5iv.labs.arxiv.org/html/1812.04948)。
CycleGAN：论文“Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks” [arXiv:1703.10593](https://ar5iv.labs.arxiv.org/html/1703.10593)。
