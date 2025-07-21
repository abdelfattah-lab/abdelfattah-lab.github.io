---
title: 'FLIQS: One-Shot Mixed-Precision Floating-Point and Integer Quantization Search'
authors:
  - key: jordan
  - key: gang
  - key: andrew
  - key: muhammad
  - key: yun
  - key: mohamed
  - key: zhiru
  - key: liqun
  - key: martin
  - key: norman
  - key: quoc
  - key: sheng
venue: automl
year: 2024
date: 2024-09-01
teaser: ''
tags:
  - dnn compression
  - automl
  - quantization
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2308.03290
    type: file-pdf
---
Quantization has become a mainstream compression technique for reducing model size, computational requirements, and energy consumption for modern deep neural networks (DNNs). With improved numerical support in recent hardware, including multiple variants of integer and floating point, mixed-precision quantization has become necessary to achieve high-quality results with low model cost. Prior mixed-precision methods have performed either a post-training quantization search, which compromises on accuracy, or a differentiable quantization search, which leads to high memory usage from branching. Therefore, we propose the first one-shot mixed-precision quantization search that eliminates the need for retraining in both integer and low-precision floating point models. We evaluate our search (FLIQS) on multiple convolutional and vision transformer networks to discover Pareto-optimal models. Our approach improves upon uniform precision, manual mixed-precision, and recent integer quantization search methods. With integer models, we increase the accuracy of ResNet-18 on ImageNet by 1.31% and ResNet-50 by 0.90% with equivalent model cost over previous methods. Additionally, for the first time, we explore a novel mixed-precision floating-point search and improve MobileNetV2 by up to 0.98% compared to prior state-of-the-art FP8 models. Finally, we extend FLIQS to simultaneously search a joint quantization and neural architecture space and improve the ImageNet accuracy by 2.69% with similar model cost on a MobileNetV2 search space.
