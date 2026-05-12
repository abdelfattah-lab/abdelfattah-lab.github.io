---
title: 'xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction'
authors:
  - key: chichih
  - key: chienyu
  - key: yash
  - key: weicheng
  - key: kaichiang
  - key: luis
  - key: mohamed
venue: icml
year: 2026
date: 2026-07-01
teaser: ''
tags:
  - llm
  - dnn compression
  - gpu
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2503.18893
    type: file-pdf
  - name: Code
    url: https://github.com/abdelfattah-lab/xKV
    type: code
preprint:
    server: arxiv
    id: 2503.18893
---
Long-context Large Language Models (LLMs) enable powerful applications but incur high memory costs due to the key–value states (KV-Cache). Recent studies attempt to share KV-Cache across layers, but these approaches either require expensive pretraining or rely on per-token cross-layer cosine similarity that is often limited in practice. We show, via Centered Kernel Alignment (CKA), that the dominant singular vectors of KV-Cache are well aligned across layers. Motivated by this observation, we propose xKV, a post-training compression method that jointly factorizes grouped-layer KV-Cache into a shared low-rank subspace, substantially reducing KV-Cache memory. Across widely used LLMs, xKV achieves up to 8× KV-Cache compression while preserving accuracy on long-context tasks and in multi-turn settings. To further improve efficiency, we introduce Selective Reconstruction (SR) at decode time. Combined with SR, xKV achieves up to 4.23× end-to-end speedup, surpassing notable baselines with 30% higher throughput under a similar accuracy level. Overall, xKV provides a plug-and-play approach to reduce both memory and latency for long-context LLM inference. Our code will be open-sourced. Our code is publicly available at: https://github.com/abdelfattah-lab/xKV.
