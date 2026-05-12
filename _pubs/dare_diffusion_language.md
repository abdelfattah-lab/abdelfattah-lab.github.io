---
title: 'DARE: Diffusion Language Model Activation Reuse for Efficient Inference'
authors:
  - key: natalia
  - key: bokun
  - key: hungyueh
  - key: chichih
  - key: mohamed
  - key: diana
venue: preprint
year: 2026
date: 2026-05-01
teaser: ''
tags:
  - machine learning
  - generative ai
  - inference-time techniques
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2605.08134
    type: file-pdf
preprint:
    server: arxiv
    id: 2605.08134
---
Diffusion Large Language Models (dLLMs) have emerged as a promising alternative to auto-regressive (AR) models, offering greater expressive capacity and potential for parallel generation and faster inference. However, open-source dLLMs remain immature, lagging behind AR models in both efficiency and quality. We identify an underexplored property of dLLMs: *token-wise redundancy* in bi-directional self-attention. Self-attention activations are highly correlated across tokens, and temporal changes in query representations can predict redundancy in corresponding key, value, and output activations. We introduce DARE, with two complementary mechanisms: DARE-KV, which reuses cached key-value (KV) activations, and DARE-O, which reuses output activations to reduce redundant computation while preserving quality. DARE achieves up to 1.20x per-layer latency reduction and reuses up to 87% of attention activations, with negligible degradation on reasoning and code-generation benchmarks. DARE-KV and DARE-O incur average performance drops of only 2.0% and 1.2%, respectively. Combined with techniques such as prefix caching and Fast-dLLM, DARE provides additive gains without retraining. These results establish token-wise reuse as an effective strategy for improving the efficiency of diffusion-based LLMs while preserving generation fidelity. Code: https://github.com/enyac-group/DARE
