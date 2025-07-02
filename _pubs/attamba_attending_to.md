---
title: 'Attamba: Attending To Multi-Token States'
authors:  
  - key: yash
  - name: Safeen Huda
  - key: mohamed
venue: preprint
year: 2024
date: 2024-11-01
teaser: ''
tags:
  - machine learning
  - generative ai
  - inference-time techniques
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2411.17685
    type: file-pdf
  - name: Code
    url: https://github.com/abdelfattah-lab/attamba
    type: code
preprint:
    server: arxiv
    id: 2411.17685
---
When predicting the next token in a sequence, vanilla transformers compute attention over all previous tokens, resulting in quadratic scaling of compute with sequence length. State-space models compress the entire sequence of tokens into a fixed-dimensional representation to improve efficiency, while other architectures achieve sub-quadratic complexity via low-rank projections or sparse attention patterns over the sequence. In this paper, we introduce Attamba, a novel architecture that uses state-space models to compress chunks of tokens and applies attention on these compressed key-value representations. We find that replacing key and value projections in a transformer with SSMs can improve model quality and enable flexible token chunking, resulting in 24% improved perplexity with transformer of similar KV-Cache and attention footprint, and ~4 times smaller KV-Cache and Attention FLOPs for 5% perplexity trade-off. Attamba can perform attention on chunked-sequences of variable length, enabling a smooth transition between quadratic and linear scaling, offering adaptable efficiency gains.
