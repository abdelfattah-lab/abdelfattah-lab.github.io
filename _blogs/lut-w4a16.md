---
title: "From RaZeR to LUT-W4A16: Flexible 4-Bit Weights from Ampere Onward"
authors:
  - key: xilai
# DRAFT: confirm the author list and publication date before publishing.
tags:
  - gpu
  - llm
  - quantization
venue: none
year: 2026
date: 2026-09-23
stub: false
published: false
materials:
  - name: Kernel code (release link TBD)
    url: https://github.com/huoshenlaile/marlin-mod
    type: code
  - name: RaZeR paper
    url: https://arxiv.org/abs/2501.04052v3
    type: file-pdf
  - name: Marlin
    url: https://github.com/IST-DASLab/marlin
    type: code
  - name: FLUTE
    url: https://github.com/HanGuo97/flute
    type: code
---

> **Working draft.** This post describes the current compact-table implementation, including a measured comparison with FLUTE on RTX 5090. Placeholders identify a future full-FP16 extension, validation on more GPUs, and model-level results.

A four-bit weight can represent an integer, a small floating-point number, or an index into a table. We want to change that interpretation without rebuilding the matrix-multiplication kernel around it.

Starting from the W4A16 kernel in our [RaZeR paper](https://arxiv.org/abs/2501.04052v3), we have generalized the decoder into a family of kernels for **INT4, FP4, RaZeR, custom lookup tables, and per-group selection between two tables**. The computation uses FP16 activations and FP16 tensor-core instructions, with a common weight layout and a decoder built from instructions available from Ampere onward.

The central idea is to keep the codebook in registers and implement the lookup with byte-permutation instructions. In the current implementation, table entries must have a zero low byte in their FP16 representation. This includes every FP4-E2M1 value and every signed INT4 value, while allowing many other codebooks. We will explain both the mechanism and that precision constraint.

## Why this kernel started with RaZeR

FP4-E2M1 has sixteen encodings but only fifteen distinct numerical values:

$$
\{0,\ \pm 0.5,\ \pm 1,\ \pm 1.5,\ \pm 2,\ \pm 3,\ \pm 4,\ \pm 6\}.
$$

The duplicated value is zero: both `0000` and `1000` encode it. RaZeR assigns the redundant negative-zero encoding to a useful extra value, selected for each quantization group. For example, one group might benefit from an extra level at $+5$, while another needs $-8$. The quantizer chooses that value offline; inference needs to recover it cheaply.

The paper studies this idea at several levels, including NVFP4 quantization and a proposed hardware decoder. The ancestor of this kernel is specifically its **weight-only W4A16 implementation** in Section 4.3: packed four-bit weights, groups of 128 weights, an FP16 block scale, and fused decoding into FP16 tensor-core operands. The paper evaluates that path on Blackwell GPUs. Its native NVFP4, two-pass W4A4 experiment in Appendix D.3 is a separate implementation. [RaZeR, Sections 4.3 and D.3](https://arxiv.org/abs/2501.04052v3).

That W4A16 path provided a useful starting point. It already inherited [Marlin's](https://arxiv.org/abs/2408.11743) offline weight permutation, asynchronous loading, tensor-core execution, and work partitioning. Generalizing the decoder lets us reuse those mechanisms for more numerical formats.

## Keep the computation; change the meaning of four bits

Let $A$ have shape $M\times K$ and the weight matrix have shape $K\times N$. For a four-bit code $q_{k,n}$, a codebook $T$, and a scale shared by $G$ consecutive weights along $K$, the operation is conceptually

$$
\widehat W_{k,n}=s_{g,n}\,T[q_{k,n}],\qquad
g=\left\lfloor k/G\right\rfloor,\qquad C=A\widehat W.
$$

The implementation reconstructs small weight fragments inside the GEMM. It never writes a full dequantized weight matrix to global memory. Packed weights and scales move through a shared-memory pipeline; the decoder builds FP16 operands in registers; `mma.sync.aligned.m16n8k16` performs the matrix multiplication. MMA accumulators are FP32, and outputs are FP16. When several thread blocks contribute to one output tile, their global reduction also passes partial results through an FP16 buffer. [Kernel implementation][kernel-source].

The six modes share this structure:

| Mode | Meaning of a four-bit code | Per-group adaptation |
| --- | --- | --- |
| `fp4` | Standard FP4-E2M1 | FP16 scale |
| `lut4` | Index into a caller-supplied 16-entry table | FP16 scale |
| `razer_high_prec` | FP4 with code 8 remapped to 5 or 8 before scaling | Scale bit 0 chooses the magnitude; signed scale supplies the sign |
| `razer_fast` | FP4 with code 8 remapped through a scale-encoded nibble | Scale bits 3:0 identify the extra level; signed scale supplies the sign |
| `lut4mixed` | Custom table with entry 8 replaced per group | Same nibble mechanism as `razer_fast` |
| `lut4dual` | Index into one of two caller-supplied tables | Scale bit 0 selects the entire table |

Each mode is compiled as a separate specialization. Choosing a mode selects a kernel at launch; it does not introduce a per-weight switch over all six possibilities. INT4 is expressed through `lut4`, for example with $T[q]=q-8$. [Mode dispatch][kernel-source] and [Python API][python-source].

## A lookup table made of bytes

The useful observation is visible in the FP16 bit patterns of FP4 values:

| Value | FP16 bits | High byte |
| ---: | --- | --- |
| 0.5 | `0x3800` | `0x38` |
| 1.0 | `0x3C00` | `0x3C` |
| 3.0 | `0x4200` | `0x42` |
| 6.0 | `0x4600` | `0x46` |
| −6.0 | `0xC600` | `0xC6` |

All have a zero low byte. A complete FP4 codebook therefore fits in sixteen bytes: four 32-bit words.

NVIDIA's `prmt.b32` instruction selects four bytes from an eight-byte pool held in two registers. Each output byte has its own selector. That makes it a small parallel lookup engine. [PTX instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt).

Our decoder uses two eight-entry banks. For four packed codes at a time, it performs the following operations:

1. Use the low three bits of each code to look up one byte from each bank.
2. Expand bit 3 of each code into a byte mask, then select the correct bank's result.
3. Interleave the selected high bytes with zeros to form four FP16 values in two `half2` registers.
4. Multiply by the corresponding stored scale and feed the resulting fragment into tensor-core MMA.

In the CUDA helper, this takes five PRMT operations, a `LOP3` blend, and the selector/mask preparation for four values. The table lookup itself needs neither a shared-memory gather nor communication with another warp lane. Weight and scale loading still use shared memory. [The `dequant_lut` helper][decode-source].

<figure>
  <img src="/imgs/blog/lut-w4a16/register-lut-decoder.svg" alt="A four-bit code selects one of two eight-byte register banks. The selected high byte is combined with a zero low byte to reconstruct FP16. Dual LUT mode first chooses a whole codebook using a scale bit." style="width:100%;height:auto;" />
  <figcaption><em>Figure 1. Register-based decoding and per-group table selection. The two banks in the upper diagram together form one 16-entry codebook; the two codebooks in dual mode each contain both banks.</em></figcaption>
</figure>

### How general is the current table?

For FP4, code bit 3 happens to be a sign bit. For a custom LUT it is simply a bank selector. The entries can be asymmetric, non-monotonic, or arranged in any code order; the upper eight entries need not be the negatives of the lower eight.

There is, however, a numerical restriction: **the current decoder represents only FP16 values whose low eight bits are zero**, equivalent to the FP8-E5M2 value grid. Computation still uses FP16. The weights still occupy four bits; the eight-bit table entries are shared across the tensor.

This compact representation already covers useful families of four-bit formats: signed INT4 values $-8$ through $7$, the full FP4-E2M1 set, and additional levels such as $5$, $7$, $8$, and $10$. We can change the range, move levels, or build asymmetric tables while keeping the same small register footprint. It excludes $9$: its FP16 representation is `0x4880`, with a nonzero low byte. Standard NF4 levels also require more precision. The public `pack_lut` helper rejects such entries rather than silently rounding them. [Packing contract][python-source].

> **Future work — full-FP16 LUT entries.** Explore a second byte-lookup path for values such as 9.0 and NF4, then measure its instruction and register cost. A possible extension would look up both byte planes and combine them into FP16 operands. The current compact-table implementation and the measurements below use only the high-byte path.

## From one remapped value to two complete codebooks

RaZeR changes one table entry. In `razer_fast`, the low nibble of the stored scale determines the high byte of entry 8:

```text
entry_8_high_byte = 0x40 | (scale_bits & 0xF)
```

A nibble of `0x5` produces FP16 `0x4500`, or $5$; `0x8` produces `0x4800`, or $8$. The table update can be expressed as one `LOP3`. Once that entry is set, code 8 goes through the ordinary lookup, with no separate per-weight compare-and-replace step. The scale's sign supplies the sign of the special value during multiplication. The offline RaZeR packer compensates by flipping ordinary nonzero codes in negative-scale groups. [Table construction][decode-source] and [quantizer][python-source].

This encoding differs from the paper's original W4A16 encoding, which used the scale's sign and most significant exponent bit for two metadata bits. The current fast mode constrains four low mantissa bits; `razer_high_prec` constrains only one. These are tradeoffs in the scale's available precision, even though they add no per-group storage.

`lut4mixed` applies the same one-entry adaptation to a custom base table. Dual mode goes further: each quantization group chooses between **two complete 16-entry codebooks**:

$$
b_{g,n}=\operatorname{bits}(s_{g,n})\mathbin{\&}1,
\qquad
\widehat W_{k,n}=s_{g,n}\,T_{b_{g,n}}[q_{k,n}].
$$

For example, a quantizer could offer a table with more levels near zero and another with a wider range, then choose the better table for each group. This is a mechanism for implementing such a quantizer; an accuracy improvement still needs to be demonstrated.

The CUDA code selects four table words for each output column's fragment and then calls the same decoder. Selection is local to a group along $K$ and an output column. It adds no second GEMM and no separate selector tensor. The union of the tables can contain 32 values, but **each group still chooses among only 16 levels**: the selector is shared, not an additional bit per weight. [Dual-mode implementation][dual-source].

There is a subtle packing rule: the kernel multiplies by the **exact stored FP16 scale, including its selector bits**. It does not clear the low bit before multiplying. A dual-LUT quantizer must choose a representable scale with the desired bit, then quantize against that stored value. For a fixed selector this leaves every other FP16 bit pattern available. The fast RaZeR mode similarly uses a grid with a fixed low nibble. [Scale encoding][python-source].

## Why Ampere is enough

The generalized path uses byte permutations and logical operations for decoding, followed by conventional FP16 MMA. It has no dependency on native FP4 tensor-core arithmetic or the newer FP4 conversion instruction. Its asynchronous copy pipeline uses `cp.async`, whose target requirement is SM80 or later. [NVIDIA PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async).

The repository records why the native FP4 conversion path was removed: below its supported targets, the CUDA helper fell back to expensive software conversion, and the native path also required a different packed-nibble order. A single PRMT decoder removes that packing distinction. The current packed layout is shared across architectures; older checkpoints produced by the removed layout need the corresponding repacking. [Porting notes][native-decode].

Architecture-specific details still matter. For example, the kernel uses a simpler `cp.async` form on Blackwell to avoid a problem with the earlier cache-hint sequence. Tile selection and the number of participating thread blocks also affect performance. Supporting the instruction set does not establish the same performance on every device. [Blackwell notes][blackwell-notes].

For this draft, the current CUDA device source compiled successfully with CUDA toolkit 13.2 to cubins for all seven targets below. Compilation establishes that the device code builds for a target; execution and performance evidence are listed separately:

| Architecture | Device-code compilation targets | Execution/performance evidence |
| --- | --- | --- |
| Ampere | `sm_80`, `sm_86` | - |
| Ada | `sm_89` | RTX 4000 Ada measurements shown below|
| Hopper | `sm_90` | - |
| Blackwell | `sm_120` | RTX 5090 measurements shown below |

The build should name the target explicitly. For example, from the kernel repository:

```bash
TORCH_CUDA_ARCH_LIST="8.0" ./compile.sh   # Ampere SM80
TORCH_CUDA_ARCH_LIST="12.0a" ./compile.sh   # Blackwell SM120
```

The current `setup.py` defaults to `8.0;8.6;8.9;12.0a+PTX`; it does not include every target in the table. Older toolkits also cannot build its Blackwell default, so they need an appropriate target override. [Build configuration][build-source].

## What the existing measurements show

The repository contains steady-state sweeps recorded on August 9–10, 2026. We replot those archived measurements here; these are **kernel throughput measurements**, not newly collected timings or end-to-end model speedups.

Both sweeps use group size 128, unlocked boost clocks, rotated variant order, and three rounds. The benchmark warms up and soaks the GPU, submits calls for a nominal six-second window, synchronizes, and divides the elapsed wall time by completed calls. Figure 2 uses median latency across rounds. The square matrix sizes differ: $N=K=12{,}288$ on RTX 4000 Ada and $N=K=43{,}520$ on RTX 5090. [Benchmark driver][benchmark-driver], [Ada data][ada-data], and [RTX 5090 data][blackwell-data].

<figure>
  <img src="/imgs/blog/lut-w4a16/archived-mode-benchmarks.svg" alt="Archived Ada and RTX 5090 speedups for single LUT, RaZeR fast, dual LUT, and Marlin INT4. Each dashed ideal roofline curve starts near 3.88 times dense FP16, falls as W4A16 becomes compute bound, and reaches 1 times when both paths are compute bound. Lower panels show relative speed on a zero-to-110-percent scale, with plain FP4 at 100 percent." style="width:100%;height:auto;" />
  <figcaption><em>Figure 2. Top: speedup over dense FP16, including Marlin INT4; dashed curves show an ideal roofline model calibrated separately for each GPU, transitioning from approximately 3.88× at small batches to 1× when both paths are compute bound. Bottom: relative speed versus this kernel's plain FP4 mode (100%); higher is faster. Each column uses its own GPU and matrix shape. Measured curves use ratios of median latencies from three recorded rounds; model assumptions are described below.</em></figcaption>
</figure>

| GPU | $M$ | Single-LUT speedup over dense FP16 | Single-LUT speed relative to FP4 | Dual-LUT speed relative to FP4 |
| --- | ---: | ---: | ---: | ---: |
| RTX 4000 Ada | 1 | 3.804× | 100.00% | 99.54% |
| RTX 4000 Ada | 128 | 1.858× | 99.80% | 94.86% |
| RTX 4000 Ada | 1024 | 1.101× | 99.99% | 94.64% |
| RTX 5090 | 1 | 3.881× | 99.93% | 99.79% |
| RTX 5090 | 128 | 1.136× | 99.98% | 97.61% |
| RTX 5090 | 1024 | 1.049× | 99.77% | 97.25% |

Relative speed is $100\,t_{\mathrm{FP4}}/t_{\mathrm{mode}}$ percent: a kernel taking 6% longer runs at $100/1.06\approx94.3\%$ of FP4's speed. The lower axes start at zero and extend to 110% to include Marlin's peak of 105.8%. The Marlin INT4 baseline comes from the same archived sweeps; on Blackwell, it includes the same `cp.async` compatibility fix used by our kernel. [Benchmark driver][benchmark-driver] and [Blackwell baseline notes][blackwell-notes].

For these workloads, passing the codebook as kernel parameters adds little measured time relative to the hardcoded FP4 table. Selecting an entire second table has a clearer cost as $M$ grows. The archived driver uses E2M1 for the single-LUT table and a distinct second table for dual mode; it uses synthetic packed weights for timing. These measurements isolate execution cost, not the quality of a trained quantized model. [Benchmark inputs][benchmark-driver].

The dashed curves adapt the roofline reasoning in [Marlin, Sections 3.1, 3.4, and 5.1](https://arxiv.org/abs/2408.11743v1). Both paths perform $F=2MKN$ floating-point operations. Assuming one read of weights, scales, and activations, plus one output write, their minimum traffic in bytes is

$$
D_{16}=2KN+2M(K+N),\qquad
D_4=\left(\frac{1}{2}+\frac{2}{G}\right)KN+2M(K+N).
$$

With bandwidth $\beta$ and a common FP16 compute rate $P$, perfect overlap of memory and compute gives

$$
S_{\mathrm{ideal}}(M)=
\frac{\max(D_{16}/\beta,\ F/P)}
     {\max(D_4/\beta,\ F/P)}.
$$

At small $M$, both paths are memory bound and weights dominate, giving approximately $16/(4+16/128)=3.88\times$. As the batch grows, W4A16 reaches its compute limit first while FP16 remains memory bound, so the speedup falls. Once both paths are compute bound, their identical arithmetic work gives $1\times$.

For these unlocked-clock sweeps, we calibrate $\beta$ from the dense $M=1$ median and $P$ from the dense $M=1024$ median, then hold both rates fixed across modes and batch sizes. These are effective rates from the archived runs, not advertised hardware peaks or fits to the quantized kernels:

| GPU | Effective bandwidth $\beta$ | Effective FP16 rate $P$ | W4A16 becomes compute bound | FP16 becomes compute bound |
| --- | ---: | ---: | ---: | ---: |
| RTX 4000 Ada | 333.49 GB/s | 70.13 TFLOP/s | $M\approx56$ | $M\approx218$ |
| RTX 5090 | 1670.25 GB/s | 218.34 TFLOP/s | $M\approx34$ | $M\approx132$ |

The model assumes no decoding, launch, reduction, or repeated-load cost and no cache benefit. It compares two idealized latencies, so it is not an upper bound on speedup against measured dense FP16. In particular, the archived runs have mode-dependent boost clocks under a power cap; measured speedups can sit above the common-rate curve. [Archived clock and power analysis][blackwell-notes].

These large square matrices amortize launch and reduction costs. The repository's separate study of real projection shapes finds lower achieved bandwidth for smaller cold weight matrices and benefits from tuning the launched block count. The draft should retain that distinction when adding model-level results. [Small-batch analysis][lowbatch-notes].

## Where this sits relative to FLUTE

[FLUTE](https://arxiv.org/abs/2407.10960) is an important baseline for LUT-quantized matrix multiplication. Its current generated kernels use vectorized lookup tables: for four-bit weights, a table of $16\times16$ pairs can return two FP16 values in one 32-bit lookup. The base pair table occupies 1 KiB before replication; shared-memory layout and duplication help manage access conflicts. Its source also contains a warp-shuffle lookup alternative. [Pair-table construction][flute-table] and [decode implementation][flute-decode].

Our decoder makes a different tradeoff. Constraining each table entry to one significant byte lets a lane reconstruct values through PRMT and logical operations, without a data-dependent shared-memory lookup for the codebook. Dual mode then adds per-group selection of those register tables.

| Capability | Current LUT-W4A16 implementation | FLUTE checkout reviewed here |
| --- | --- | --- |
| Weight bit widths | 4 | 2, 3, 4 |
| Activation/compute input type | FP16 | FP16 and BF16 |
| LUT entry precision | FP16 values with a zero low byte | Full FP16 or BF16 table entries |
| Codebook lookup in the discussed path | Register byte permutations | Vectorized shared-memory pair lookup |
| Per-group selection between two complete codebooks | `lut4dual`, encoded in scale bit 0 | Not exposed by the reviewed scalar-LUT API |
| RaZeR-style per-group entry replacement | `razer_*` and `lut4mixed` | Not exposed by that API |

This is broader than the original fixed-format RaZeR kernel in the formats and group adaptations it supports. FLUTE remains more general in table-entry precision, activation type, and weight bit width. The useful performance question is how the two compare on the **same supported codebook and workload**. [FLUTE API][flute-api] and [our API][python-source].

### A matched comparison on RTX 5090

We measured our single-table **`lut4` mode** against FLUTE's four-bit FP16 kernel using identical integer codes, an E2M1 lookup table, stored FP16 scales, and activations. The dense baseline multiplies the same explicitly decoded FP16 weights. Both quantized kernels are tuned for each shape, batch size, and group size before measurement.

We focus here on two shapes: $N=14{,}336$, $K=4096$, and $N=K=43{,}520$. On the large square matrix, our LUT kernel is **1.06× faster than FLUTE at $M=1$ and 1.53× faster at $M=1024$**. For $N=14{,}336$, $K=4096$, the kernels are essentially tied at $M=1$ ($21.12\,\mu s$ for ours and $21.09\,\mu s$ for FLUTE). At $M=1024$, our kernel takes $520.28\,\mu s$ versus FLUTE's $700.87\,\mu s$, a **1.35× speedup**.

<figure>
  <img src="/imgs/blog/lut-w4a16/flute-comparison-speedup.svg" alt="Two plots compare our LUT-W4A16 kernel and FLUTE by speedup over dense FP16 on RTX 5090 for N=14336, K=4096 and N=K=43520. Higher is faster; a dashed line marks dense FP16 at 1×. Both panels use group size 128 and rotate weights beyond twice L2 capacity." style="width:100%;height:auto;" />
  <figcaption><em>Figure 3. Speedup over dense FP16 on RTX 5090 with the same E2M1-quantized weights and group size 128. Higher is faster; the dashed line marks dense FP16 at 1×. Distinct weight allocations rotate beyond twice the 96 MiB L2 capacity to approximate cold weights. Lines show dense FP16 median latency divided by each kernel's median latency; shading spans the speedups from three CUDA-graph timing rounds. These measurements use a different timing method from Figure 2.</em></figcaption>
</figure>

The table below gives absolute latencies for selected points from Figure 3 and our speedup over FLUTE, $t_{\mathrm{FLUTE}}/t_{\mathrm{ours}}$. Values above $1\times$ in the final column favor our LUT kernel; Figure 3 uses dense FP16 as its baseline. Here $N$ is output width and $K$ is reduction width.

| $N$ | $K$ | $M$ | FLUTE latency (µs) | Our LUT latency (µs) | Our speedup over FLUTE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 14336 | 4096 | 1 | 21.09 | 21.12 | 0.998× |
| 14336 | 4096 | 128 | 91.80 | 73.86 | 1.243× |
| 14336 | 4096 | 1024 | 700.87 | 520.28 | 1.347× |
| 43520 | 43520 | 1 | 622.38 | 585.51 | 1.063× |
| 43520 | 43520 | 128 | 3112.45 | 2070.49 | 1.503× |
| 43520 | 43520 | 1024 | 25373.04 | 16542.61 | 1.534× |

Figure 3 shows these two shapes at group size 128, with batches from 1 to 1024. We also collected [repeated-weight latency measurements](/imgs/blog/lut-w4a16/flute-comparison-repeated.svg). Reusing one allocation changes cache behavior for the $14{,}336\times4096$ shape; the large square packed matrix still exceeds L2. Both regimes use the settings selected with rotating weights, so the repeated-weight regime is not independently tuned. The [full results CSV](/assets/data/lut-w4a16/flute-comparison.csv) contains all 118 timing rows from the broader 59-case sweep, including group size 64, selected configurations, and numerical errors.

For these two shapes, our performance advantage is clearest at larger batches. These measurements compare complete kernels, including their scheduling and accumulation choices; they do not isolate the cost of register lookup versus shared-memory lookup. Near-ties should be read in light of the unlocked clocks and variation across rounds.

### How we measured the comparison

The platform is an RTX 5090 with 170 SMs, 96 MiB L2, and its existing 600 W power cap. We used PyTorch 2.9.1 with CUDA 13.0 runtime libraries. FLUTE 0.4.2 was built from commit `9eb83a1` with CUTLASS 3.4.1 and CUDA toolkit 13.2, targeting SM120; our kernel uses the existing extension from the inspected `0261159` checkout. The FLUTE build enables FP16 W4G64/W4G128 and retains all 144 upstream four-bit templates. It required build configuration changes, but no changes to its GEMM algorithm or device instructions.

FLUTE tuning screens all 144 templates; 138 launch successfully for these shapes. The fastest eight candidates are timed again and checked for correctness. Our tuning sweeps the legal tile configurations and 12 launch block counts, then refines and checks the fastest four candidates. This gives each implementation a chance to choose a suitable configuration instead of relying on a fixed default.

A two-second dense GEMM warmup precedes each shape/group sweep. Timings use CUDA events around unrolled CUDA graphs, with at least 30 ms per variant per round, three rounds, and rotated variant order. Figure 3 reports ratios of median latencies. Packing, quantization, and CPU dispatch are outside the timed region. FLUTE uses its upstream mixed accumulation scheme; our kernel uses FP32 MMA accumulators with FP16 storage for cross-block partial reductions. These are kernel measurements, not end-to-end model speedups.

All 59 matched cases passed numerical checks against FP32 matmul over explicitly decoded FP16 weights. The worst relative L2 error was $7.68\times10^{-4}$ for FLUTE and $5.07\times10^{-4}$ for our LUT kernel, below the $2\times10^{-3}$ threshold. Twelve additional FLUTE checks using full-FP16 NF4 entries also passed; the timed comparison uses E2M1, which both implementations represent exactly.

> **Still to measure:** repeat the matched comparison on Ampere and Ada, and evaluate learned single/dual codebooks in real models. The scalar single-codebook FLUTE benchmark above does not measure the per-group dual-LUT operation; that mode needs its own overhead and accuracy evaluation.

## Using the implementation

The supplied demonstration quantizer makes it easy to try RaZeR:

```python
import torch
import marlin_razer

linear = torch.nn.Linear(4096, 4096, bias=False).cuda().half()
layer = marlin_razer.Layer(
    infeatures=4096, outfeatures=4096,
    groupsize=128, mode="razer_fast",
).cuda()
layer.quick_quantize_razer4(linear)
y = layer(torch.randn(16, 4096, device="cuda", dtype=torch.float16))
```

For custom codebooks, an integration supplies its own quantizer and packs the resulting codes and scales. `pack_lut` packs only the codebook; it does not quantize the weights. At the low-level interface, INT4 decoding can be selected as follows:

```python
int4_table = marlin_razer.pack_lut(list(range(-8, 8)))

# A, packed_codes, C, packed_scales, and workspace are prepared beforehand.
# Each packed code q represents q - 8, in this implementation's weight layout.
marlin_razer.mul(
    A, packed_codes, C, packed_scales, workspace,
    mode="lut4", lut=int4_table,
)
```

The layer currently supports group sizes 32, 64, and 128; `fp4` and `lut4` also support per-column scales. Its public shape constraints require $K$ divisible by 128 and $N$ divisible by 256, or 512 for group size 32. Packing, mode, codebooks, and scale metadata must agree. The existing `Layer.pack()` method emits plain FP4 codes, so it should not be used as a generic LUT quantizer. [API and packing implementation][python-source].

## Checks completed, and results still to add

During preparation of this draft, all six modes passed 61 focused GEMM cases on an RTX 5090 using PyTorch 2.9.1 and CUDA 13.0. The cases cover group sizes 32/64/128, per-column scaling where supported, signed scales, all code values, asymmetric tables, and changing dual-table selectors. Against explicitly decoded FP16 weights and an FP32 reference matmul, the worst normalized mean absolute error was $8.05\times10^{-4}$. This checks numerical agreement with the decoded weights, not accuracy relative to the original unquantized model.

A separate CPU model checked every combination of four packed codes, $16^4=65{,}536$, for three tables, including exact signed-zero bit patterns. These are targeted checks; broader shape coverage, GPU sanitizers, and execution on the other architectures remain part of release validation.

> **Before publication:** extend the matched FLUTE benchmark to more GPU architectures; add model accuracy and end-to-end decode results for learned single/dual codebooks; complete hardware validation beyond the devices measured here; and confirm the project name, authors, and date. The kernel source links use the inspected checkout's configured remote and commit, which were not accessible in an unauthenticated check; confirm the public release location and release the comparison benchmark harness. Full-FP16 table entries remain a separate extension to investigate.

The decoder gives quantization experiments a common execution path: changing a format can mean changing sixteen table entries, and adapting a format across groups can mean selecting a second table. The next step is to use that flexibility on real models and measure where the additional choices pay off.

[kernel-source]: https://github.com/huoshenlaile/marlin-mod/blob/0261159896e848983c72eea7d24360cecb7943f1/marlin_razer/csrc/marlin_razer_cuda_kernel.cu
[decode-source]: https://github.com/huoshenlaile/marlin-mod/blob/0261159896e848983c72eea7d24360cecb7943f1/marlin_razer/csrc/marlin_razer_cuda_kernel.cu#L275
[dual-source]: https://github.com/huoshenlaile/marlin-mod/blob/0261159896e848983c72eea7d24360cecb7943f1/marlin_razer/csrc/marlin_razer_cuda_kernel.cu#L762
[python-source]: https://github.com/huoshenlaile/marlin-mod/blob/0261159896e848983c72eea7d24360cecb7943f1/marlin_razer/__init__.py
[build-source]: https://github.com/huoshenlaile/marlin-mod/blob/0261159896e848983c72eea7d24360cecb7943f1/setup.py
[native-decode]: https://github.com/huoshenlaile/marlin-mod/blob/0261159896e848983c72eea7d24360cecb7943f1/docs/native_fp4_dequant.md
[blackwell-notes]: https://github.com/huoshenlaile/marlin-mod/blob/0261159896e848983c72eea7d24360cecb7943f1/docs/rtx5090_blackwell_benchmark.md
[benchmark-driver]: https://github.com/huoshenlaile/marlin-mod/blob/0261159896e848983c72eea7d24360cecb7943f1/benchmarks/benchmark_boost.py
[ada-data]: https://github.com/huoshenlaile/marlin-mod/blob/0261159896e848983c72eea7d24360cecb7943f1/results/boost/boost_sweep.json
[blackwell-data]: https://github.com/huoshenlaile/marlin-mod/blob/0261159896e848983c72eea7d24360cecb7943f1/results/boost_rtx5090/boost_sweep.json
[lowbatch-notes]: https://github.com/huoshenlaile/marlin-mod/blob/0261159896e848983c72eea7d24360cecb7943f1/docs/lowbatch_analysis.md
[flute-table]: https://github.com/HanGuo97/flute/blob/9eb83a12d56949bbe7fe9c836ba97a67bd1e3761/flute/utils.py#L15
[flute-decode]: https://github.com/HanGuo97/flute/blob/9eb83a12d56949bbe7fe9c836ba97a67bd1e3761/flute/csrc/packbits_utils.hpp
[flute-api]: https://github.com/HanGuo97/flute/blob/9eb83a12d56949bbe7fe9c836ba97a67bd1e3761/flute/ops.py
