---
title: 'Learned Connectivity Sparsification for LUT-based Neural Networks'
authors:
  - name: Erwei Wang
  - name: Georgios Stavrou
  - name: Peter Cheung
  - name: George Constantinides
  - key: mohamed
  - name: James Davis
venue: trets
year: 2023
date: 2023-03-01
teaser: ''
tags:
  - machine learning
  - generative ai
  - inference-time techniques
materials:
  - name: PDF
    url: https://dl.acm.org/doi/pdf/10.1145/3583075
    type: file-pdf
---

Field-programmable gate array (FPGA)–specific deep neural network (DNN) architectures using native lookup tables (LUTs) as independently trainable inference operators have been shown to achieve favorable area-accuracy and energy-accuracy trade-offs. The first work in this area, LUTNet, exhibited state-of-the-art performance for standard DNN benchmarks. In this article, we propose the learned optimization of such LUT-based topologies, resulting in higher-efficiency designs than via the direct use of off-the-shelf, hand-designed networks. Existing implementations of this class of architecture require the manual specification of the number of inputs per LUT, K. Choosing appropriate K a priori is challenging. Doing so at even high granularity, for example, per layer, is a time-consuming and error-prone process that leaves FPGAs’ spatial flexibility underexploited. Furthermore, prior works see LUT inputs connected randomly, which does not guarantee a good choice of network topology. To address these issues, we propose logic shrinkage, a fine-grained netlist pruning methodology enabling K to be automatically learned for every LUT in a neural network targeted for FPGA inference. By removing LUT inputs determined to be of low importance, our method increases the efficiency of the resultant accelerators. Our GPU-friendly solution to LUT input removal is capable of processing large topologies during their training with negligible slowdown. With logic shrinkage, we improve the area and energy efficiency of the best-performing LUTNet implementation of the CNV network classifying CIFAR-10 by 1.54× and 1.31×, respectively, while matching its accuracy. This implementation also reaches 2.71× the area efficiency of an equally accurate, heavily pruned binary neural network (BNN). On ImageNet, with the Bi-Real Net architecture, employment of logic shrinkage results in a post-synthesis area reduction of 2.67× vs. LUTNet, allowing for implementation that was previously impossible on today’s largest FPGAs. We validate the benefits of logic shrinkage in the context of real application deployment by implementing a face mask detection DNN using a BNN, LUTNet, and logic-shrunk layers. Our results show that logic shrinkage results in area gains versus LUTNet (up to 1.20×) and equally pruned BNNs (up to 1.08×), along with accuracy improvements.