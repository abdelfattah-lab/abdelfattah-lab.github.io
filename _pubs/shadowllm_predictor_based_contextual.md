---
title: 'ShadowLLM: Predictor-based Contextual Sparsity for Large Language Models'
authors:
  - key: yash
  - name: Ahmed F. AbouElhamayed
  - key: jordan
  - name: Zhiru Zhang
  - name: Alexander M. Rush
  - name: Safeen Huda
  - key: mohamed
venue: emnlp
year: 2024
date: 2024-11-01
teaser: ''
tags:
  - machine learning
  - generative ai
  - inference-time techniques
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2406.16635
    type: file-pdf
---
The high power consumption and latency-sensitive deployments of large language models (LLMs) have motivated efficiency techniques like quantization and sparsity. Contextual sparsity, where the sparsity pattern is input-dependent, is crucial in LLMs because the permanent removal of attention heads or neurons from LLMs can significantly degrade accuracy. Prior work has attempted to model contextual sparsity using neural networks trained to predict activation magnitudes, which can be used to dynamically prune structures with low predicted activation magnitude. In this paper, we look beyond magnitude-based pruning criteria to assess attention head and neuron importance in LLMs. We develop a novel predictor called ShadowLLM, which can shadow the LLM behavior and enforce better sparsity patterns, resulting in over 15% improvement in end-to-end accuracy compared to prior methods. In addition, ShadowLLM achieves up to a 20% speed-up over the state-of-the-art DejaVu framework. These enhancements are validated on Llama-2 and OPT models with up to 30 billion parameters. Our code is available at \href{https://github.com/abdelfattah-lab/shadow_llm/}{ShadowLLM}.
