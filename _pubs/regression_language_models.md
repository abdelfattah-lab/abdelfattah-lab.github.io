---
title: 'Regression Language Models for Code'
authors:
  - key: yash
  - key: xingyou
  - key: arissa
  - key: bryan
  - key: mohamed
venue: preprint
year: 2025
date: 2025-10-01
teaser: ''
tags:
  - machine learning
  - generative ai
  - inference-time techniques
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2509.26476
    type: file-pdf
  - name: Model
    url: https://huggingface.co/akhauriyash/RLM-GemmaS-Code-v0
    type: code
  - name: Code Dataset
    url: https://huggingface.co/datasets/akhauriyash/Code-Regression
    type: database
  - name: NAS Dataset
    url: https://huggingface.co/datasets/akhauriyash/GraphArch-Regression
    type: database
preprint:
    server: arxiv
    id: 2509.26476
---
We study code-to-metric regression: predicting numeric outcomes of code executions, a challenging task due to the open-ended nature of programming languages. While prior methods have resorted to heavy and domain-specific feature engineering, we show that a single unified Regression Language Model (RLM) can simultaneously predict directly from text, (i) the memory footprint of code across multiple high-level languages such as Python and C++, (ii) the latency of Triton GPU kernels, and (iii) the accuracy and speed of trained neural networks represented in ONNX. In particular, a relatively small 300M parameter RLM initialized from T5Gemma, obtains > 0.9 Spearman-rank on competitive programming submissions from APPS, and a single unified model achieves > 0.5 average Spearman-rank across 17 separate languages from CodeNet. Furthermore, the RLM can obtain the highest average Kendall-Tau of 0.46 on five classic NAS design spaces previously dominated by graph neural networks, and simultaneously predict architecture latencies on numerous hardware platforms.
