---
title: "Extending MXFP4 and NVFP4 with Redundant Zero Remapping (RaZeR) for Accurate 4-bit LLM Quantization"
authors:
  - key: yuzong
  - key: xilai
  - key: mohamed
tags:
  - llm
  - quantization
venue: none
year: 2025
date: 2025-08-13
stub: false
materials:
  - name: MXFP4
    url: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    type: file-pdf
  - name: NVFP4
    url: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
    type: code
  - name: BitMoD
    url: https://arxiv.org/abs/2411.11745
    type: file-pdf
  - name: RaZeR
    url: https://github.com/abdelfattah-lab/razer-llm
    type: code
---


# TL;DR
We extend the standard FP4 format with Redundant Zero Remapping (RaZeR), which adaptively remaps the negative zero encoding to one of some pre-defined special values. This allows to maximally utilize the available quantization encodings. Through the careful algorithm-hardware co-design of these special values, RaZeR can better fit LLM numerical distributions with minimal hardware overhead. By integrating RaZeR on top of two widely used FP4 variants—MXFP4 and NVFP4—we achieve better accuracy under 4-bit weight quantization across a range of large language models (LLMs). 


# Introduction
The substantial memory footprint and computational demand of LLMs present critical challenges for efficient inference. Recent advancements in quantization alleviate the memory bottleneck of LLMs while boosting the computational throughput. Specifically, 4-bit weight-only quantization offers a favorable trade-off between model accuracy and hardware performance, and can be effectively performed in a post-training manner. Motivated by this, the industry has adopted two custom 4-bit data types—MXFP4 and NVFP4—both based on the standard FP4 format, for LLM quantization. For instance, the recently released [GPT-OSS](https://github.com/openai/gpt-oss) is natively quantized to MXFP4 through quantization-aware training. 

However, one crucial aspect of FP4, that has been overlooked by these proposals, is the redundant negative zero encoding. Due to the inherent sign-magnitude representation, FP4 contains both positive and negative zeroes in its quantization values, despite one being redundant. In this post, we discuss how to intelligently repurpose this redundant zero as an additional, meaningful quantization value, which improves model accuracy compared to the basic FP4 format. 


# Recap of FP4
We briefly discuss the concept of FP4, MXFP4, and NVFP4, which are illustrated in Figure 1.
<div style="text-align:center;">
  <img src="/imgs/blog/razer_blog/fp4_variant.png" width="75%" />
  <figcaption style="font-size: 0.95em; margin-top: 8px;">
    <em>Figure 1: Visualization of FP4, MXFP4, and NVFP4.</em>
  </figcaption>
</div>  

## Definition
  According to the [OCP standard](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), the FP4 format contains 1 sign bit, 2 exponent bit, and 1 mantissa bits. It can represent 15 quantization values \[0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0\]. 

## Microscaling FP4 (MXFP4)
  According to the [OCP standard](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), the microscaling (MX) format specifies that a block of 32 elements shares an E8M0 scale factor, which is an unsigned representation of the conventional FP32 exponent. After scaling, each element is quantized to a specified data type, e.g., FP4.

## NVIDIA FP4 (NVFP4)
  The latest NVIDIA Blackwell GPU architecture introduces the [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) format, which differs from MXFP4 in two aspects. First, it reduces the quantization block size from 32 to 16, enabling finer-grained scaling than MXFP4. Second, it adopts a two-level scaling approach, where every block of 16 elements share an FP8-E4M3 scale factor, followed by an FP32 scale factor shared per tensor.  


# Redundant Zero Remapping (RaZeR)
One disadvantage of FP4 comes from its inherent sign-magnitude representation that introduces both positive and negative zeroes, despite one of them being redundant. In other words, the standard FP4 format wastes 1/16 = 6.25% of the available encoding. To address this, we propose Redundant Zero Remapping (RaZeR), originally published in [our HPCA 2025 paper](https://arxiv.org/abs/2411.11745). RaZeR allows the negative zero encoding to be replaced by one of some pre-defined ***special values*** (SVs), resulting in an extended set of quantization values: \[SV, 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0\]. Ideally, the SV may contain arbitary value, e.g., anything from FP16. But computing such a high-precision SV will incur considerable hardware overhead and offset the energy efficiency brought by FP4. Hence, we limit the choices of SV to fixed-point values with one fraction bit, i.e., an multiple of 0.5 as in FP4. Figure 2 illustrates an example design of RaZeR, which allows the negative zero to be remapped to six SVs: \[±2.5, ±5.0, ±8.0\]. 

<div style="text-align:center;">
  <img src="/imgs/blog/razer_blog/fp4_sv.png" width="95%" />
  <figcaption style="font-size: 0.95em; margin-top: 8px;">
    <em>Figure 2: Visualization of RaZeR.</em>
  </figcaption>
</div>  

In order to integrate RaZeR with MXFP4 and NVFP4, each block of elements can adaptively select its own optimal SV. Specifically, the quantization algorithm can iterate over all pre-defined SVs for a block, and selects the SV that minimizes the block's resulting quantization error. To identify which SV is selected, each block stores a *logN*-bit metadata, where *N* is the number of pre-defined SVs. In addition, a lightweight hardware decoder is required to perform computation with RaZeR, as illustrated in Figure 3. Assume there are four pre-defined SVs stored in dedicated registers, which are indexed via a 2-bit metadata. During computation, the FP4 operand is compared with negative zero, and if equal, the target SV associated with the current quantization block will be output.

<div style="text-align:center;">
  <img src="/imgs/blog/razer_blog/fp4_hardware.png" width="75%" />
  <figcaption style="font-size: 0.95em; margin-top: 8px;">
    <em>Figure 3: Hardware decoder of RaZeR.</em>
  </figcaption>
</div>  


# Experiment Results
To demonstrate the benefits of RaZeR, we implement six weight-only quantization algorithms to evaluate the quantization effects of FP4, MXFP4, NVFP4, FP4-RaZeR, MXFP4-RaZeR, and NVFP4-RaZeR. For RaZeR, we choose four pre-defined SVs: \[±5.0, ±8.0\]. Both FP4 and FP4-RaZeR adopt a block size of 128 with a per-block FP16 scale factor. While MXFP4-RaZeR and NVFP4-RaZeR use the same block size and scaling configuration as MXFP4 and NVFP4, respectively. The code to reproduce our experiments is available [here](https://github.com/abdelfattah-lab/razer-llm). 

The table below shows the perplexity of Wikitext2 and C4 datasets across a range of LLMs, all using the instruction-tuned versions. Adding RaZeR consistently outperforms FP4, MXFP4, and NVFP4.

<table><thead>
  <tr>
    <th rowspan="2"> <br>Method </th>
    <th rowspan="2"> <br>Block Size </th>
    <th rowspan="2"> <br>Scale Format </th>
    <th colspan="2">Llama-3.1-8B </th>
    <th colspan="2">Llama-3.2-3B </th>
    <th colspan="2">QWen2.5-3B </th>
    <th colspan="2"> QWen2.5-7B </th>
    <th colspan="2">QWen2.5-14B </th>
  </tr>
  <tr>
    <th>Wiki2</th>
    <th>C4</th>
    <th>Wiki2</th>
    <th>C4</th>
    <th>Wiki2</th>
    <th>C4</th>
    <th>Wiki2</th>
    <th>C4</th>
    <th>Wiki2</th>
    <th>C4</th>
  </tr></thead>
<tbody>
  <tr>
    <td> <br>FP16 </td>
    <td> <br>N/A </td>
    <td> <br>N/A </td>
    <td> <br>7.21 </td>
    <td> <br>10.38 </td>
    <td> <br>11.05 </td>
    <td> <br>14.48 </td>
    <td> <br>8.56 </td>
    <td> <br>12.03 </td>
    <td> <br>7.46 </td>
    <td> <br>10.88 </td>
    <td> <br>5.69 </td>
    <td> <br>9.38 </td>
  </tr>
  <tr>
    <td> <br>FP4 </td>
    <td> <br>128 </td>
    <td> <br>FP16 </td>
    <td> <br>7.71 </td>
    <td> <br>11.15 </td>
    <td> <br>11.98 </td>
    <td> <br>15.53 </td>
    <td> <br>9.27 </td>
    <td> <br>12.91 </td>
    <td> <br>7.79 </td>
    <td> <br>11.31 </td>
    <td> <br>6.17 </td>
    <td> <br>9.72 </td>
  </tr>
  <tr>
    <td> <br>FP4-RaZeR </td>
    <td> <br>128 </td>
    <td> <br>FP16 </td>
    <td> <strong><br>7.60 </strong></td>
    <td> <strong><br>10.97 </strong></td>
    <td> <strong><br>11.94 </strong></td>
    <td> <strong><br>15.40 </strong></td>
    <td> <strong><br>9.11 </strong></td>
    <td> <strong><br>12.71 </strong></td>
    <td> <strong><br>7.79 </strong></td>
    <td> <strong><br>11.20 </strong></td>
    <td> <strong><br>6.14 </strong></td>
    <td> <strong><br>9.68 </strong></td>
  </tr>
  <tr>
    <td> <br>MXFP4 </td>
    <td> <br>32 </td>
    <td> <br>E8M0 </td>
    <td> <br>8.14 </td>
    <td> <br>11.60 </td>
    <td> <br>12.31 </td>
    <td> <br>16.01 </td>
    <td> <br>9.44 </td>
    <td> <br>13.09 </td>
    <td> <br>8.48 </td>
    <td> <br>11.97 </td>
    <td> <br>6.41 </td>
    <td> <br>9.89 </td>
  </tr>
  <tr>
    <td> <br>MXFP4-RaZeR </td>
    <td> <br>32 </td>
    <td> <br>E8M0 </td>
    <td> <strong><br>7.68 </strong></td>
    <td> <strong><br>11.01 </strong></td>
    <td> <strong><br>11.76 </strong></td>
    <td> <strong><br>15.31 </strong></td>
    <td> <strong><br>9.14 </strong></td>
    <td> <strong><br>12.80 </strong></td>
    <td> <strong><br>7.77 </strong></td>
    <td> <strong><br>11.25 </strong></td>
    <td> <strong><br>6.22 </strong></td>
    <td> <strong><br>9.70 </strong></td>
  </tr>
  <tr>
    <td> <br>NVFP4 </td>
    <td> <br>16 </td>
    <td> <br>E4M3 + FP32 </td>
    <td> <br>7.56 </td>
    <td> <br>10.93 </td>
    <td> <br>11.73 </td>
    <td> <br>15.15 </td>
    <td> <br>9.12 </td>
    <td> <br>12.60 </td>
    <td> <br>7.73 </td>
    <td> <br>11.13 </td>
    <td> <br>6.05 </td>
    <td> <br>9.63 </td>
  </tr>
  <tr>
    <td> <br>NVFP4-RaZeR </td>
    <td> <br>16 </td>
    <td> <br>E4M3 + FP32 </td>
    <td> <strong><br>7.42 </strong></td>
    <td> <strong><br>10.71 </strong></td>
    <td> <strong><br>11.41 </strong></td>
    <td> <strong><br>14.89 </strong></td>
    <td> <strong><br>8.95 </strong></td>
    <td> <strong><br>12.45 </strong></td>
    <td> <strong><br>7.67 </strong></td>
    <td> <strong><br>11.05 </strong></td>
    <td> <strong><br>5.95 </strong></td>
    <td> <strong><br>9.56 </strong></td>
  </tr>
</tbody></table>


# Related Work
Although the focus of RaZeR is to address the redundant zero encoding of FP4, there is a recent work, [BlockDialect](https://arxiv.org/abs/2501.01144), that configures the two largest magnitude values of FP4 to capture various large magnitude distributions. In the meanwhile, most of the small magnitude values to remain consistent with FP4, reducing the hardware complexity to support diverse encodings. The proposed RaZeR is synergistic to BlockDialect. 


# Conclusion
In this post, we introduce RaZeR, which adaptively remaps the redundant zero encoding of FP4 to an additional, meaningful quantization value. We empirically validate that RaZeR significantly improves model accuracy compared to FP4, MXFP4, and NVFP4. This provides a guidance for future FP4 tensor core design.  


# Citing
```bibtex
@misc{abdelfattah2025_razer_blog,
      title={Extending MXFP4 and NVFP4 with Redundant Zero Remapping (RaZeR) for Accurate 4-bit LLM Quantization}, 
      author={Yuzong Chen and Xilai Dai and Mohamed Abdelfattah},
      year={2025},
      url={https://abdelfattah-lab.github.io/blog/razer_blog}, 
}
```