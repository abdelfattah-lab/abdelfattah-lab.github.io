---
title: 'PQA: Exploring the Potential of Product Quantization in DNN Hardware Acceleration'
authors:
  - name: Ahmed F. AbouElhamayed
  - name: Angela Cui
  - name: Javier Fernandez-Marques
  - name: Nicholas D. Lane
  - key: mohamed
venue: trets
year: 2024
date: 2024-05-01
teaser: ''
tags:
  - machine learning
  - generative ai
  - inference-time techniques
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2305.18334
    type: file-pdf
---
Conventional multiply-accumulate (MAC) operations have long dominated computation time for deep neural networks (DNNs), espcially convolutional neural networks (CNNs). Recently, product quantization (PQ) has been applied to these workloads, replacing MACs with memory lookups to pre-computed dot products. To better understand the efficiency tradeoffs of product-quantized DNNs (PQ-DNNs), we create a custom hardware accelerator to parallelize and accelerate nearest-neighbor search and dot-product lookups. Additionally, we perform an empirical study to investigate the efficiency--accuracy tradeoffs of different PQ parameterizations and training methods. We identify PQ configurations that improve performance-per-area for ResNet20 by up to 3.1$\times$, even when compared to a highly optimized conventional DNN accelerator, with similar improvements on two additional compact DNNs. When comparing to recent PQ solutions, we outperform prior work by $4\times$ in terms of performance-per-area with a 0.6% accuracy degradation. Finally, we reduce the bitwidth of PQ operations to investigate the impact on both hardware efficiency and accuracy. With only 2-6-bit precision on three compact DNNs, we were able to maintain DNN accuracy eliminating the need for DSPs.
