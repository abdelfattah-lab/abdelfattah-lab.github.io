---
title: 'Kratos: An FPGA Benchmark for Unrolled DNNs with Fine-Grained Sparsity and Mixed Precision'
authors:
  - key: xilai
  - key: yuzong
  - key: mohamed
venue: fpl
year: 2024
date: 2024-09-01
teaser: ''
tags:
  - fpga
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2407.06033
    type: file-pdf
  - name: Code
    url: https://github.com/abdelfattah-lab/Kratos-benchmark
    type: code
---
FPGAs offer a flexible platform for accelerating deep neural network (DNN) inference, particularly for non-uniform workloads featuring fine-grained unstructured sparsity and mixed arithmetic precision. To leverage these redundancies, an emerging approach involves partially or fully unrolling computations for each DNN layer. That way, parameter-level and bit-level ineffectual operations can be completely skipped, thus saving the associated area and power. Regardless, unrolled implementations scale poorly and limit the size of a DNN that can be unrolled on an FPGA. This motivates the investigation of new reconfigurable architectures to improve the efficiency of unrolled DNNs, while taking advantage of sparsity and mixed precision. To enable this, we present Kratos: a focused FPGA benchmark of unrolled DNN primitives with varying levels of sparsity and different arithmetic precisions. Our analysis reveals that unrolled DNNs can operate at very high frequencies, reaching the maximum frequency limit of an Arria 10 device. Additionally, we found that substantial area reductions can be achieved through fine-grained sparsity and low bit-width. We build on those results to tailor the FPGA fabric for unrolled DNNs through an architectural case study demonstrating $\sim$2$\times$ area reduction when using smaller LUT sizes within current FPGAs. This paves the way for further exploration of new programmable architectures that are purpose-built for sparse and low-precision unrolled DNNs. Our source code and benchmark are available on github.com/abdelfattah-lab/Kratos-benchmark.
