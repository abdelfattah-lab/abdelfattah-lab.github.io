---
title: 'FlashDepth: Real-time Streaming Depth Estimation at 2K Resolution'
authors:
  - key: gene
  - key: wenqi
  - key: guandao
  - key: mohamed
  - key: bharath
  - key: noah
  - key: ning
  - key: paul
venue: iccv
year: 2025
date: 2025-04-01
teaser: ''
tags:
  - dnn compression
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2504.07093
    type: file-pdf
  - name: Code
    url: https://github.com/Eyeline-Research/FlashDepth
    type: code
preprint:
    server: arxiv
    id: 2504.07093
---
A versatile video depth estimation model should (1) be accurate and consistent across frames, (2) produce high-resolution depth maps, and (3) support real-time streaming. We propose FlashDepth, a method that satisfies all three requirements, performing depth estimation on a 2044x1148 streaming video at 24 FPS. We show that, with careful modifications to pretrained single-image depth models, these capabilities are enabled with relatively little data and training. We evaluate our approach across multiple unseen datasets against state-of-the-art depth models, and find that ours outperforms them in terms of boundary sharpness and speed by a significant margin, while maintaining competitive accuracy. We hope our model will enable various applications that require high-resolution depth, such as video editing, and online decision-making, such as robotics. We release all code and model weights at https://github.com/Eyeline-Research/FlashDepth
