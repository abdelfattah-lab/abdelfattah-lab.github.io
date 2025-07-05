---
title: 'Towards Neural Architecture Search through Hierarchical Generative Modeling'
authors:
  - key: lichuan
  - key: lukasz
  - key: mohamed
  - key: abhinav
  - key: nicholas
  - key: hongkai
venue: icml
year: 2024
date: 2024-08-01
teaser: ''
tags:
  - machine learning
  - generative ai
  - inference-time techniques
materials:
  - name: PDF
    url: https://openreview.net/pdf?id=VdZfEMuoj2
    type: file-pdf
---

Neural Architecture Search (NAS) is gaining popularity in automating designing deep neural networks for various tasks. A typical NAS pipeline begins with a manually designed search space which is methodically explored during the process, aiding the discovery of high-performance models. Although NAS has shown impressive results in many cases, the strong performance remains largely dependent on, among other things, the prior knowledge about good designs which is implicitly incorporated into the process by carefully designing search spaces. In general, this dependency is undesired, as it limits the applicability of NAS to less-studied tasks and/or results in an explosion of the cost needed to obtain strong results. In this work, our aim is to address this limitation by leaning on the recent advances in generative modelling -- we propose a method that can navigate an extremely large, general-purpose search space efficiently, by training a two-level hierarchy of generative models. The first level focuses on micro-cell design and leverages Conditional Continuous Normalizing Flow (CCNF) and the subsequent level uses a transformer-based sequence generator to produce macro architectures for a given task and architectural constraints. To make the process computationally feasible, we perform task-agnostic pretraining of the generative models using a metric space of graphs and their zero-cost (ZC) similarity. We evaluate our method on typical tasks, including CIFAR-10, CIFAR-100 and ImageNet models, where we show state-of-the-art performance compared to other low-cost NAS approaches.