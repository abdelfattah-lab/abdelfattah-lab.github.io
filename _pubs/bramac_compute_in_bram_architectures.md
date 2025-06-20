---
title: 'BRAMAC: Compute-in-BRAM Architectures for Multiply-Accumulate on FPGAs'
authors:
  - key: yuzong
  - key: mohamed
venue: fccm
year: 2023
date: 2023-05-01
teaser: ''
tags:
  - machine learning
  - generative ai
  - inference-time techniques
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2304.03974.pdf
    type: file-pdf
---
Deep neural network (DNN) inference using reduced integer precision has been shown to achieve significant improvements in memory utilization and compute throughput with little or no accuracy loss compared to full-precision floating-point. Modern FPGA-based DNN inference relies heavily on the on-chip block RAM (BRAM) for model storage and the digital signal processing (DSP) unit for implementing the multiply-accumulate (MAC) operation, a fundamental DNN primitive. In this paper, we enhance the existing BRAM to also compute MAC by proposing BRAMAC (Compute-in-$\underline{\text{BR}}$AM $\underline{\text{A}}$rchitectures for $\underline{\text{M}}$ultiply-$\underline{\text{Ac}}$cumulate). BRAMAC supports 2's complement 2- to 8-bit MAC in a small dummy BRAM array using a hybrid bit-serial & bit-parallel data flow. Unlike previous compute-in-BRAM architectures, BRAMAC allows read/write access to the main BRAM array while computing in the dummy BRAM array, enabling both persistent and tiling-based DNN inference. We explore two BRAMAC variants: BRAMAC-2SA (with 2 synchronous dummy arrays) and BRAMAC-1DA (with 1 double-pumped dummy array). BRAMAC-2SA/BRAMAC-1DA can boost the peak MAC throughput of a large Arria-10 FPGA by 2.6$\times$/2.1$\times$, 2.3$\times$/2.0$\times$, and 1.9$\times$/1.7$\times$ for 2-bit, 4-bit, and 8-bit precisions, respectively at the cost of 6.8%/3.4% increase in the FPGA core area. By adding BRAMAC-2SA/BRAMAC-1DA to a state-of-the-art tiling-based DNN accelerator, an average speedup of 2.05$\times$/1.7$\times$ and 1.33$\times$/1.52$\times$ can be achieved for AlexNet and ResNet-34, respectively across different model precisions.
