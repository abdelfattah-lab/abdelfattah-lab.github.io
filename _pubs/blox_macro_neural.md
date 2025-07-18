---
title: 'BLOX: Macro Neural Architecture Search Benchmark and Algorithms'
authors:
  - key: thomas
  - key: lukasz
  - key: hongkai
  - key: nicholas
  - key: mohamed
venue: nips
year: 2022
date: 2022-12-01
teaser: ''
tags:
  - automl
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2210.07271.pdf
    type: file-pdf
---
Neural architecture search (NAS) has been successfully used to design numerous high-performance neural networks. However, NAS is typically compute-intensive, so most existing approaches restrict the search to decide the operations and topological structure of a single block only, then the same block is stacked repeatedly to form an end-to-end model. Although such an approach reduces the size of search space, recent studies show that a macro search space, which allows blocks in a model to be different, can lead to better performance. To provide a systematic study of the performance of NAS algorithms on a macro search space, we release Blox - a benchmark that consists of 91k unique models trained on the CIFAR-100 dataset. The dataset also includes runtime measurements of all the models on a diverse set of hardware platforms. We perform extensive experiments to compare existing algorithms that are well studied on cell-based search spaces, with the emerging blockwise approaches that aim to make NAS scalable to much larger macro search spaces. The benchmark and code are available at https://github.com/SamsungLabs/blox.
