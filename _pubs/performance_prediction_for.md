---
title: 'Performance Prediction for Large Systems via Text-to-Text Regression'
authors:
  - key: yash
  - key: bryan
  - key: chenghsi
  - key: adrian
  - key: grant
  - key: arissa
  - key: bangding
  - key: mohamed
  - key: sagi
  - key: xingyou
venue: preprint
year: 2025
date: 2025-06-01
teaser: ''
tags:
  - automl
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2506.21718
    type: file-pdf
  - name: Code
    url: https://github.com/google-deepmind/regress-lm
    type: code
preprint:
    server: arxiv
    id: 2506.21718
---
In many industries, predicting metric outcomes of large systems is a fundamental problem, driven largely by traditional tabular regression. However, such methods struggle on complex systems data in the wild such as configuration files or system logs, where feature engineering is often infeasible. We propose text-to-text regression as a general, scalable alternative. For predicting resource efficiency on Borg, Google's massive compute cluster scheduling system, a 60M parameter encoder-decoder, trained from random initialization, achieves up to a near perfect 0.99 (0.9 average) rank correlation across the entire fleet, and 100x lower MSE than tabular approaches. The model also easily adapts to new tasks in only 500 few-shot examples and captures the densities of complex outcome distributions. Ablation studies highlight the importance of using encoders, increasing sequence length, and the model's inherent uncertainty quantification. These findings pave the way for universal simulators of real-world outcomes.
