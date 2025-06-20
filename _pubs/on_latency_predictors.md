---
title: 'On Latency Predictors for Neural Architecture Search'
authors:
  - key: yash
  - key: mohamed
venue: mlsys
year: 2024
date: 2024-05-01
teaser: ''
tags:
  - machine learning
  - generative ai
  - inference-time techniques
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2403.02446.pdf
    type: file-pdf
---
Efficient deployment of neural networks (NN) requires the co-optimization of accuracy and latency. For example, hardware-aware neural architecture search has been used to automatically find NN architectures that satisfy a latency constraint on a specific hardware device. Central to these search algorithms is a prediction model that is designed to provide a hardware latency estimate for a candidate NN architecture. Recent research has shown that the sample efficiency of these predictive models can be greatly improved through pre-training on some \textit{training} devices with many samples, and then transferring the predictor on the \textit{test} (target) device. Transfer learning and meta-learning methods have been used for this, but often exhibit significant performance variability. Additionally, the evaluation of existing latency predictors has been largely done on hand-crafted training/test device sets, making it difficult to ascertain design features that compose a robust and general latency predictor. To address these issues, we introduce a comprehensive suite of latency prediction tasks obtained in a principled way through automated partitioning of hardware device sets. We then design a general latency predictor to comprehensively study (1) the predictor architecture, (2) NN sample selection methods, (3) hardware device representations, and (4) NN operation encoding schemes. Building on conclusions from our study, we present an end-to-end latency predictor training strategy that outperforms existing methods on 11 out of 12 difficult latency prediction tasks, improving latency prediction by 22.5\% on average, and up to to 87.6\% on the hardest tasks. Focusing on latency prediction, our HW-Aware NAS reports a $5.8\times$ speedup in wall-clock time. Our code is available on \href{https://github.com/abdelfattah-lab/nasflat_latency}{https://github.com/abdelfattah-lab/nasflat\_latency}.
