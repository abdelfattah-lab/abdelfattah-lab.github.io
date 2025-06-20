---
title: 'M4BRAM: Mixed-Precision Matrix-Matrix Multiplication in FPGA Block RAMs'
authors:
  - key: yuzong
  - key: jordan
  - key: mohamed
venue: fpt
year: 2023
date: 2023-12-01
teaser: ''
tags:
  - machine learning
  - generative ai
  - inference-time techniques
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2311.02758.pdf
    type: file-pdf
---
Mixed-precision quantization is a popular approach for compressing deep neural networks (DNNs). However, it is challenging to scale the performance efficiently with mixed-precision DNNs given the current FPGA architecture and conventional accelerator dataflows. In this work, we enhance the FPGA's capability for accelerating mixed-precision DNNs by proposing M4BRAM, a novel compute-in-block RAM (BRAM) architecture that can compute mixed-precision matrix-matrix multiplication. On the precision side, M4BRAM supports a wide range of mixed-precision DNN configurations -- the weight precision can be 2/4/8 bits while the activation precision can vary from 2 to 8 bits. On the dataflow side, M4BRAM leverages a novel in-BRAM data duplication scheme to achieve high hardware utilization. Moreover, during M4BRAM computation, other FPGA resources can seamlessly access its data without the need for a separate buffer. Hence, unlike prior compute-in-BRAM proposals, M4BRAM can simultaneously perform mixed-precision computation and maintain full functionality as a memory unit to \textit{truly} complement the existing compute resources on FPGAs. Experiments show that adding M4BRAM to a tiled DNN accelerator can achieve an average speedup of 2.16$\times$ across various DNNs on the ImageNet classification task while incurring a negligible accuracy loss of $<$ 0.5%. Compared to the same tiled accelerator that employs a prior compute-in-BRAM architecture, M4BRAM delivers 1.43$\times$ higher performance on average across various DNNs.
