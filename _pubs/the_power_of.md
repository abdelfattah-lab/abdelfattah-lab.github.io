---
title: 'The Power of Negative Zero: Datatype Customization for Quantized Large Language Models'
authors:
  - key: yuzong
  - key: xilai
  - key: chichih
  - key: yash
  - key: mohamed
venue: preprint
year: 2025
date: 2025-01-01
teaser: ''
tags:
  - dnn compression
  - llm
  - gpu
  - quantization
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2501.04052
    type: file-pdf
  - name: Code
    url: https://github.com/abdelfattah-lab/RaZeR
    type: code
preprint:
    server: arxiv
    id: 2501.04052
---
Large language models (LLMs) have demonstrated remarkable performance across various machine learning tasks, quickly becoming one of the most prevalent AI workloads. Yet the substantial memory requirement of LLMs significantly hinders their deployment for end users. Post-training quantization (PTQ) serves as one of the most hardware-efficient methods to mitigate the memory and computational demands of LLMs. Although the traditional integer (INT) datatype has received widespread adoption in PTQ methods, floating-point (FP) quantization has emerged as a viable alternative thanks to its effectiveness in fitting LLM numerical distributions. However, the FP datatype in sign-magnitude binary representation contains both positive and negative zero, which constrains its representation capability, particularly under low precision (3 and 4 bits). In this paper, we extend the basic FP datatype to perform Redundant Zero Remapping (RaZeR), which remaps the negative zero FP encoding to a set of pre-defined special values to maximally utilize FP quantization encodings and to better fit LLM numerical distributions. Through careful selection of special values, RaZeR outperforms conventional asymmetric INT quantization while achieving high computational efficiency. We demonstrate that RaZeR can be seamlessly integrated with quantization algorithms for both weights and KV-cache, including advanced methods with clipping and transformations, and consistently achieve better model accuracy. Additionally, we implement a fast GEMV kernel with fused dequantization that efficiently converts the 4-bit RaZeR value to FP16 through novel bit-level manipulation. On modern GPUs, our evaluation shows that RaZeR improves the GEMV speed by up to 7.56$\times$ compared to the FP16 implementation, while achieving up to 2.72$\times$ speedup in the LLM decoding throughput.
