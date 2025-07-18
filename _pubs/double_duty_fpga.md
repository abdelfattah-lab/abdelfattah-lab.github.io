---
title: 'Double Duty: FPGA Architecture to Enable Concurrent LUT and Adder Chain Usage'
authors:
  - key: junius
  - key: xilai
  - key: grace
  - key: mahesh
  - key: boutros
  - key: vaughn
  - key: mohamed
venue: fpl
year: 2025
date: 2025-09-01
teaser: ''
tags:
  - fpga
  - hardware
materials:
  - name: PDF
    url: https://arxiv.org/pdf/2507.11709
    type: file-pdf
---
Flexibility and customization are key strengths of Field-Programmable Gate Arrays (FPGAs) when compared to other computing devices. For instance, FPGAs can efficiently implement arbitrary-precision arithmetic operations, and can perform aggressive synthesis optimizations to eliminate ineffectual operations. Motivated by sparsity and mixed-precision in deep neural networks (DNNs), we investigate how to optimize the current logic block architecture to increase its arithmetic density. We find that modern FPGA logic block architectures prevent the independent use of adder chains, and instead only allow adder chain inputs to be fed by look-up table (LUT) outputs. This only allows one of the two primitives -- either adders or LUTs -- to be used independently in one logic element and prevents their concurrent use, hampering area optimizations. In this work, we propose the Double Duty logic block architecture to enable the concurrent use of the adders and LUTs within a logic element. Without adding expensive logic cluster inputs, we use 4 of the existing inputs to bypass the LUTs and connect directly to the adder chain inputs. We accurately model our changes at both the circuit and CAD levels using open-source FPGA development tools. Our experimental evaluation on a Stratix-10-like architecture demonstrates area reductions of 21.6% on adder-intensive circuits from the Kratos benchmarks, and 9.3% and 8.2% on the more general Koios and VTR benchmarks respectively. These area improvements come without an impact to critical path delay, demonstrating that higher density is feasible on modern FPGA architectures by adding more flexibility in how the adder chain is used. Averaged across all circuits from our three evaluated benchmark set, our Double Duty FPGA architecture improves area-delay product by 9.7%.
