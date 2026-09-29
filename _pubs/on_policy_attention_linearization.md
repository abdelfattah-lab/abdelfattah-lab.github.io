---
title: 'On-Policy Attention Linearization'
authors:
  - key: arian
    equal: true
  - key: anupam
    equal: true
  - key: anthony
  - key: akaash
  - key: mohamed
  - key: gauri
venue: preprint
year: 2026
date: 2026-09-01
teaser: ''
tags:
  - llm
  - dnn compression
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2609.31947
    type: file-pdf
  - name: Blog
    url: https://hunchings.github.io/late-stage-patching/
    type: blog
preprint:
    server: arxiv
    id: 2609.31947
---
Hybrid transformer architectures that replace most softmax attention layers with linear attention offer transformer-level quality at a fraction of the memory cost. Rather than pretraining such models, a growing body of work distills them from already trained full-attention transformers. However, these distilled models often collapse on long-context retrieval and reasoning tasks, particularly when operating in thinking mode, where the efficiency gains of hybrid architectures matter most. Since linear attention layers must compress context into a fixed-size state, their errors compound over long sequences. As off-policy distillation never teaches the student model to recover from this drift, tasks that necessitate longer sequence lengths become especially challenging. We introduce On-Policy Attention Linearization (OPAL) in which the hybrid attention student samples its own long-context trajectories and receives dense supervision from the frozen full-attention teacher. Applying OPAL to Qwen3-4B and MiMo-7B-RL-0530, we recover 87-94% of full-attention performance on commonsense reasoning, 100% on needle-in-a-haystack (NIAH) retrieval, and 83-93% on mathematical reasoning with only 3B training tokens. We achieve these results without supervised fine-tuning (SFT) or reinforcement learning with verifiable rewards (RLVR). Compared with the strongest prior linearization method, which recovers 68% of its teacher's retrieval performance and 21.6% absolute average mathematical reasoning accuracy, OPAL fully recovers retrieval and achieves 67.6-72.2% on math reasoning.
