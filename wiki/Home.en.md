# Wiki Overview

> 🌐 [中文](Home.md) | **English**

This wiki documents the repository's **CPU inference optimizations for Bonsai 1-bit (Q1_0)
models**: principles, implementation, build, delivery, testing and troubleshooting.

> **Current build: Build 24 (Based on llama.cpp b-10991).** Every console tool (`cli` /
> `server` / `train` / `bench` / `perplexity` / `quantize` / `imatrix` / `tokenize` /
> `gguf-split` / `rpc-server`) prints the same Lumina-Engine startup screen at launch
> (ASCII logo + build number + GitHub / Afdian / Bilibili links). The build number comes from
> `LLAMA_BUILD_NUMBER` and always matches `--version`.

## Pages

| Page | Content |
|------|------|
| [Home](Home.en.md) | This page — overview and quick navigation |
| [Background](Background.en.md) | Bonsai model, Q1_0 format, GGUF, the bottlenecks of 1-bit inference |
| [Optimizations](Optimizations.en.md) | Principles and implementation of the core optimizations (including Q1LUT) |
| [Building](Building.en.md) | Single / multi-variant / dual-toolchain builds, artifact layout |
| [Delivery](Delivery.en.md) | **Delivery directory `Lumina-Engine-Bin` and the generation config (scripts, assembly, conventions)** |
| [Benchmarks](Benchmarks.en.md) | Methodology, raw data, A/B comparisons, micro-benchmarks |
| [Correctness](Correctness.en.md) | Numerical equivalence, perplexity, sampling, backend tests |
| [Compatibility](Compatibility.en.md) | 14 CPU variants, selection logic, distribution notes |
| [Changes](Changes.en.md) | File-by-file summary of the main code changes |
| [ModelCompression](ModelCompression.en.md) | **User guide: how to compress a model and load it** |
| [LosslessCompression](LosslessCompression.en.md) | **Lossless GGUF container (Build 8): field split + context sort + DEFLATE, `.gguf.lumina`** |
| [Training](Training.en.md) | **Training engine: QLoRA fine-tuning of quantized models / `.gguf.lumina` in C++** |
| [Troubleshooting](Troubleshooting.en.md) | FAQ and diagnostics |

## One-page summary

### The problem
Every weight in Bonsai-8B is 1-bit (`GGML_TYPE_Q1_0`: 128 weights share one fp16 scale,
1.125 bit/weight). On CPU:
- **Token generation (tg)** must stream ~1.06 GB of weights per token → memory-bandwidth bound.
- **Prompt processing (pp)** reuses weights → compute bound, dominated by the `Q1_0 × Q8_0`
  dot-product kernel.

### The approach
1. **Per-128 activation quantization (Q8_128)**: one activation scale per 128 weights lets the four
   int32 sub-block dot products of a Q1_0 block accumulate before a single float conversion
   (4 → 1 conversions).
2. **Register tiling (2 rows × 4 cols)**: expand the sign bits once and reuse across 4 columns,
   with no extra memory traffic.
3. **VNNI**: `vpdpbusd` fuses `vpmaddubsw + vpmaddwd`.
4. **`vpsignb`**: one instruction for “keep +1 / negate −1”.
5. **Single-column GEMV (tg)**: reuses per-128 activations, one float conversion per 128 weights,
   with vectorized, block-parallel activation quantization.
6. **Binary lookup table (Q1LUT, enabled by default)**: an activation-side table GEMV for binary
   weights; tg and pp with `ne11 >= 8` use it.
7. **Multi-microarchitecture build**: one source tree compiled into 14 CPU backends, selected at
   runtime by CPUID.

### Results (local i5-10400, 6 threads)

> ⚠️ **The test machine uses single-channel memory** (8 GB DDR4-2666; measured streaming read bandwidth
> ~14–17 GB/s): tg (token generation) is capped by this bandwidth (theoretical ceiling ~15.7 tok/s), so the
> tg absolute numbers are low; on dual-channel / higher-bandwidth platforms the tg gain approaches the
> ~30% kernel-level value. Prefer same-run A/B **ratios** over absolute numbers.

- **pp +29% ~ +43%** vs. the generic path (Q1LUT adds roughly +26% ~ +57% at `ne11 >= 8`).
- **tg ~ +12% ~ +14%** (single-column GEMV, 12 threads); **Q1LUT enabled by default adds ~+30%**
  (~9.8 → ~13.0 t/s).
- Perplexity: optimized 5.7173; Q1LUT 5.7049 (both on par with generic 5.7160 — no accuracy loss).
- 47/47 backend tests pass.

### Quick commands
```bat
REM Delivery artifacts live in Lumina-Engine-Bin\ at the repo root (the only delivery dir; see Delivery.en.md)
cd C:\Lumina-Engine\Lumina-Engine-Bin
Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 512 -n 128 -t 6

set GGML_NO_Q1_TILE=1   & REM fall back to the generic kernel for A/B (pp+tg)
set GGML_NO_Q1_GEMV=1   & REM fall back for tg GEMV only
set GGML_Q1_GEMV_ROWS=2 & REM tune tg row grouping (1/2/4)
set GGML_Q1_LUT=0       & REM disable the binary LUT (enabled by default)
set GGML_Q1_LUT_PP=0    & REM disable only the pp LUT
set GGML_Q1_LUT_DEBUG=1 & REM print LUT cache allocation
```
