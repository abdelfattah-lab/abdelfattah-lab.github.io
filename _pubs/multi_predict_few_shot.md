---
title: 'Multi-Predict: Few Shot Predictors for Efficient Neural Architecture Search'
authors:
  - key: yash
  - key: mohamed
venue: automl
year: 2023
date: 2023-09-01
teaser: ''
tags:
  - automl
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2306.02459.pdf
    type: file-pdf
---
Many hardware-aware neural architecture search (NAS) methods have been developed to optimize the topology of neural networks (NN) with the joint objectives of higher accuracy and lower latency. Recently, both accuracy and latency predictors have been used in NAS with great success, achieving high sample efficiency and accurate modeling of hardware (HW) device latency respectively. However, a new accuracy predictor needs to be trained for every new NAS search space or NN task, and a new latency predictor needs to be additionally trained for every new HW device. In this paper, we explore methods to enable multi-task, multi-search-space, and multi-HW adaptation of accuracy and latency predictors to reduce the cost of NAS. We introduce a novel search-space independent NN encoding based on zero-cost proxies that achieves sample-efficient prediction on multiple tasks and NAS search spaces, improving the end-to-end sample efficiency of latency and accuracy predictors by over an order of magnitude in multiple scenarios. For example, our NN encoding enables multi-search-space transfer of latency predictors from NASBench-201 to FBNet (and vice-versa) in under 85 HW measurements, a 400$\times$ improvement in sample efficiency compared to a recent meta-learning approach. Our method also improves the total sample efficiency of accuracy predictors by over an order of magnitude. Finally, we demonstrate the effectiveness of our method for multi-search-space and multi-task accuracy prediction on 28 NAS search spaces and tasks.
