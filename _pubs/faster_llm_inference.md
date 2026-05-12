---
title: 'Faster LLM Inference via Sequential Monte Carlo'
authors:
  - key: yahya
  - key: mau
  - key: chichih
  - key: cameron
  - key: tim
  - key: ryan
  - key: mohamed
venue: preprint
year: 2026
date: 2026-08-01
teaser: ''
tags:
  - machine learning
  - generative ai
  - inference-time techniques
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2604.15672
    type: file-pdf
  - name: Code
    url: https://github.com/abdelfattah-lab/smcsd
    type: code
preprint:
    server: arxiv
    id: 2604.15672
---
Speculative decoding (SD) accelerates language model inference by drafting tokens from a cheap proposal model and verifying them against an expensive target model via rejection sampling. Because rejection truncates the draft block at the first error, throughput degrades when draft and target diverge. Rather than rejecting draft tokens outright, we propose to reweight them. To this end, we introduce sequential Monte Carlo speculative decoding (SMC-SD), which replaces token-level rejection with importance-weighted resampling over a population of draft particles. SMC-SD is a principled approximate inference scheme that trades exactness for additional speed, while preserving theoretical bounds on its per-step approximation error. Because LLM inference is memory bandwidth-bound, the arithmetic needed to draft particles and to score them in parallel comes nearly for free -- SMC-SD uses idle compute to turn verification into a vectorized, fixed-size operation with no rollback. Empirically, SMC-SD achieves 2.36x speed-up over speculative decoding and a 5.2x speed-up over autoregressive decoding, while remaining within 3% of the target model's accuracy on reasoning, instruction-following, and coding benchmarks.
