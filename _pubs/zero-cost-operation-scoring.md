---
title: 'Zero-Cost Operation Scoring in Differentiable Architecture Search'
authors:
  - name: Lichuan Xiang
  - name: \L ukasz Dudziak
  - key: mohamed
  - name: Thomas Chau
  - name: Nicholas D. Lane
  - name: Hongkai Wen
venue: preprint
year: 2022
date: 2022-06-01
tags:
  - machine learning
  - generative ai
  - inference-time techniques
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2106.06799.pdf
    type: file-pdf
preprint:
    server: arxiv
    id: 2106.06799.pdf
---
We formalize and analyze a fundamental component of differentiable neural architecture search (NAS): local "operation scoring" at each operation choice. We view existing operation scoring functions as inexact proxies for accuracy, and we find that they perform poorly when analyzed empirically on NAS benchmarks. From this perspective, we introduce a novel \textit{perturbation-based zero-cost operation scoring} (Zero-Cost-PT) approach, which utilizes zero-cost proxies that were recently studied in multi-trial NAS but degrade significantly on larger search spaces, typical for differentiable NAS. We conduct a thorough empirical evaluation on a number of NAS benchmarks and large search spaces, from NAS-Bench-201, NAS-Bench-1Shot1, NAS-Bench-Macro, to DARTS-like and MobileNet-like spaces, showing significant improvements in both search time and accuracy. On the ImageNet classification task on the DARTS search space, our approach improved accuracy compared to the best current training-free methods (TE-NAS) while being over 10$\times$ faster (total searching time 25 minutes on a single GPU), and observed significantly better transferability on architectures searched on the CIFAR-10 dataset with an accuracy increase of 1.8 pp. Our code is available at: https://github.com/zerocostptnas/zerocost_operation_score.
