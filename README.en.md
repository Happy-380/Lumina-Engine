# Lumina-Engine

> 🌐 [中文](README.md) | **English**

> A Windows x86-64 inference / compression / training engine built on llama.cpp b-10991: dedicated CPU kernels for **Bonsai Q1_0**, plus **Vulkan acceleration, automatic device policy, lossless compression, and QLoRA training** for a wider range of models.

![Build](https://img.shields.io/badge/build-24-blue)
![llama.cpp](https://img.shields.io/badge/llama.cpp-b10991-green)
![Platform](https://img.shields.io/badge/platform-Windows%20x86--64-lightgrey)
![License](https://img.shields.io/badge/license-AGPL--3.0-orange)
[![Support on Afdian](https://img.shields.io/badge/Support-Afdian-ff69b4)](https://afdian.com/a/lumi_380)

[Quick start](#quick-start) · [Optimization scope](#optimization-scope) · [Benchmarks](#benchmarks) · [Docs](wiki/Home.en.md) · [Contributors](CONTRIBUTORS.md#english) · [❤ Support](#support-this-project)

**At a glance:**

- **What it is**: an enhanced fork of llama.cpp b-10991. The core is a highly optimized pure-CPU path for `GGML_TYPE_Q1_0` (1.125 bits/weight); the same delivery also includes general inference acceleration, lossless GGUF compression, and QLoRA training for quantized models.
- **Who it is for**: anyone who wants to run 1-bit models cheaply on a regular Windows machine; anyone using Q4_K / Q8_0 models for inference, compression, or fine-tuning; and anyone who needs long-context training on an 8 GB machine.
- **What is different**: it goes beyond "it runs" — Q1_0 prefill is up to **+43%** faster and token generation **+30%** faster; a Q4_K model can be compressed losslessly to **94.28%** of its original size; quantized models can be QLoRA-tuned on pure CPU, and 32K context training fits in 8 GB of RAM.

**Fastest way to run it (model already available):**

```bat
cd /d C:\Lumina-Engine\Lumina-Engine-Bin
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf -p "Hello, introduce yourself." -n 128 -t 6
```

> No model yet? Read the prerequisites in [Quick start](#quick-start).

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Features](#features)
- [Optimization scope](#optimization-scope)
- [Quick start](#quick-start)
- [Common workflows](#common-workflows)
- [Benchmarks](#benchmarks)
- [Automatic device policy (CPU / GPU)](#automatic-device-policy-cpu--gpu)
- [Compatibility](#compatibility)
- [Build from source](#build-from-source)
- [Environment variables](#environment-variables)
- [Architecture and layout](#architecture-and-layout)
- [FAQ](#faq)
- [Changelog](#changelog)
- [Documentation](#documentation)
- [Support this project](#support-this-project)
- [Contributing](#contributing)
- [License and acknowledgements](#license-and-acknowledgements)

---

## Why this exists

Every weight in Bonsai-8B is 1-bit (`Q1_0`: 128 weights share one fp16 scale), so the model is only **1.07 GiB** and looks lightweight. On a CPU, however:

- **Token generation (tg)**: every token requires reading all 1.07 GiB of weights once → **memory-bandwidth bound**. On single-channel DDR4 the theoretical ceiling is only ~15.7 tok/s.
- **Prompt processing (pp)**: weights are reused → **compute-bound**, with almost all the time spent in the `Q1_0 × Q8_0` dot-product kernel.

The upstream generic Q1_0 kernel converts to float every 32 weights and does not exploit the structure of binary weights. Lumina-Engine redesigns this path:

| Optimization | Result (local i5-10400, single-channel 8 GB) |
|---|---|
| per-128 activation quantization + register-tiled GEMM | pp **+28.5% ~ +42.5%** |
| Single-column GEMV (token generation) | tg **+12% ~ +14%** (12 threads) |
| Binary lookup table Q1LUT (enabled by default) | tg **+30%**; pp512 **+26%**, pp2048 **+57%** |
| 14 CPU microarchitecture variants | one prebuilt package picks the best kernel for almost any x86-64 CPU |

> All numbers are **same-session A/B ratios**. The test machine uses single-channel memory, which caps absolute tg throughput; on dual-channel / higher-bandwidth platforms the gain gets closer to the kernel-level improvement.
>
> The above focuses on the Q1_0 path. See [Optimization scope](#optimization-scope) for general improvements such as Vulkan, lossless compression, and QLoRA training.

---

## Features

- **1-bit-specific kernels**: per-128 activations (Q8_128), 2×4 register tiling, single-column GEMV, VNNI (`vpdpbusd`), and `vpsignb` — **strictly isolated** to the Q1_0 + AVX2 path, with zero impact on other models.
- **Optimizations beyond Q1_0**: 14 CPU variants, the Vulkan backend and automatic device policy, lossless GGUF compression, QLoRA for quantized models, and long-context training memory optimizations all cover a wider range of models and quantization types.
- **True binary lookup (Q1LUT)**: exploits the symmetry `T[c ^ 0xf] = -T[c]` to build only half the table. tg uses the lookup path by default; pp switches to it automatically when `ne11 >= 8`. Disable everything with `GGML_Q1_LUT=0`.
- **Numerically lossless**: Q1_0 bit operations are exact; 47/47 backend tests pass, perplexity is in the same range as the generic path (difference < 0.2%), and greedy decoding matches.
- **One package, 14 CPUs**: SSE4.2 / Sandy Bridge / Haswell / Skylake-X / Ice Lake / Alder Lake / Zen 4 / Sapphire Rapids… selected automatically via CPUID at runtime.
- **Automatic CPU / GPU decisions**: pure CPU without a GPU; operator offload on integrated GPUs; full-layer offload on discrete GPUs; MoE models automatically avoid the "more offload, slower" trap.
- **Lossless model compression**: the `.gguf.lumina` container separates the `d/dmin/scales` fields of Q4_K / Q6_K from the weight streams and then applies DEFLATE (F32 tensors are compressed separately). For `SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf`, 2.63 GB → 2.48 GB (**94.28%**), with a byte-identical SHA-256 after decompression.
- **Built-in training engine**: `Lumina-Engine-train` performs **QLoRA** directly on Q4_K / Q8_0 / Q1_0 quantized models, with JSONL datasets, checkpoints, resume, MoE expert adapters, and 32K long-context training on an 8 GB machine (attention recomputation during backward + dataset mmap; stable at ~3.5 GiB).
- **One-command delivery**: `build-bin.bat` builds the MSVC + Clang trees and assembles the runtime directory; no manual DLL copying.

---

## Optimization scope

Not every change is 1-bit-only: the Q1_0 kernels are strictly isolated, while the remaining optimizations apply to more models and quantization types.

| Optimization | Scope |
|---|---|
| per-128 activations, tiled GEMM, single-column GEMV, Q1LUT | **Q1_0 only** (Bonsai family); non-Q1_0 paths are untouched |
| Vulkan Q1_0 DP4A GEMV | **Q1_0 + discrete Vulkan GPU only** (disabled on integrated GPUs by default) |
| 14 CPU variants, automatic device policy, Vulkan operator offload | **All models** (dense / MoE; policy selected automatically per model type) |
| QLoRA training, checkpoints / resume, long-context memory optimization | **Q4_K / Q8_0 / Q1_0**, dense + MoE |
| `.gguf.lumina` lossless compression | **GGUF files containing Q4_K / Q6_K / F32 tensors** (currently supported by the packer) |
| Console tooling, one-command build, training ETA / thread tuning | All tools and training jobs |

### Measured gains on other models

| Scenario | Result |
|---|---|
| Qwen3-0.6B-Q8_0 + Intel iGPU operator offload (`-ngl 0`) | pp512 **~2.4×** (420 vs 207 t/s), tg unchanged |
| phi-3-mini Q4_K_M + iGPU operator offload | pp256 **~1.6×** (76 vs 48 t/s) |
| Qwen3-0.6B automatic device policy | PPL 23.7397 vs 23.7639 pure CPU, **~1.9×** faster |
| QLoRA on quantized models | SmallThinker MoE loss **1.93 → 1.44**; Qwen2.5-0.5B **2.55 → 1.99** |
| 32K long-context training (8 GB machine) | Private memory stable at **~3.5 GiB**, no swapping, bit-identical loss |

> General-optimization data comes from an i5-12500H / Intel Iris Xe (Vulkan) machine; Q1_0 data comes from an i5-10400. None of these capabilities depend on Q1_0, and the Q1_0-specific kernels still only affect the Q1_0 + AVX2 path.

---

## Quick start

### Prerequisites

| Item | Requirement |
|---|---|
| OS | Windows 10 / 11 x86-64 |
| CPU | SSE4.2 required; **AVX2 recommended** (Haswell 2013+ / Zen+). VNNI / AVX-512 is faster |
| RAM | ≥ 8 GB recommended for 8B models; 4 GB is enough for 1.7B |
| Model | **Bonsai-8B GGUF (Q1_0)**, ~1.07 GiB. Download it yourself from [PrismML](https://prismml.com); it is not bundled. Other Q4_K / Q8_0 GGUF files also load directly |
| Runtime | Inference needs **no Python / Node / CUDA**; the prebuilt package ships every DLL. The compression scripts require Python 3 |

### Option 1: use the prebuilt binaries (recommended)

`Lumina-Engine-Bin/` already contains 14 CPU variants, all tools, and the runtime libraries (54 files). Put the model in the repository root, then:

```bat
cd /d C:\Lumina-Engine\Lumina-Engine-Bin

REM Single-turn inference
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf -p "The capital of France is" -n 32 -st -t 6
```

Expected output:

```text
The capital of France is Paris.
...
llama_perf_context_print:        eval time = ...
```

Interactive chat (omit `-p` to enter multi-turn mode):

```bat
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf -t 6
```

Start an OpenAI-compatible HTTP server:

```bat
Lumina-Engine-server.exe -m ..\Bonsai-8B.gguf -c 4096 -t 6
```

```bat
curl.exe http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain quantum computing in one sentence.\"}]}"
```

> **Thread count defaults to physical cores (Hyper-Threading disabled).** 1-bit GEMV shares L1 and memory bandwidth under SMT and is usually slower. SMT is only enabled when you explicitly pass `-t <logical-core-count>`.

### Option 2: build from source

Requires CMake ≥ 3.14, Ninja, and VS 2022 / 2026 (MSVC); Clang 20 is optional (5 extra variants). See [wiki/Building.en.md](wiki/Building.en.md).

```bat
cd /d C:\Lumina-Engine

REM 1) Configure the multi-variant build tree (first time only)
configure-release.bat

REM 2) Build the MSVC tree
build-release.bat

REM 3) Assemble the runtime directory
bench\assemble_bin.bat
```

> The root `.bat` files hard-code the author's paths (`D:\VS2026`, `D:\llvm`, `C:\Lumina-Engine`, and optionally `D:\VulkanSDK\sdk`). On another machine or in another directory, edit the paths at the top of the scripts or use the CMake commands below directly.

`build-bin.bat` builds and assembles once `build-release` (and optionally `build-clang`) is **already configured**. If `clang` is not on `PATH`, it skips the Clang build and restores the 5 Clang-only variants from `bench\clang_variants_backup\`.

Build for this machine only (fastest, single variant):

```bat
cmake -S llama.cpp-b10991 -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
cmake --build build -j
```

The output is in `build\bin\`.

---

## Common workflows

> The commands below assume the current directory is `Lumina-Engine-Bin\`; step 1 changes to it.

### 1. Benchmark

```bat
cd /d C:\Lumina-Engine\Lumina-Engine-Bin
Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 512 -n 128 -t 6 -r 3
```

### 2. Perplexity (verify numerical equivalence)

```bat
Lumina-Engine-perplexity.exe -m ..\Bonsai-8B.gguf -f ..\wiki\Background.md -c 512 -t 6 --chunks 4
```

> `-f` accepts any UTF-8 text file; substitute your own corpus.

### 3. A/B against the upstream generic kernel

```bat
set GGML_NO_Q1_TILE=1
Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 512 -n 128 -t 6
set GGML_NO_Q1_TILE=
```

To compare the lookup path separately:

```bat
set GGML_Q1_LUT=0   && REM Disable Q1LUT (saves ~0.95 GiB of memory)
set GGML_Q1_LUT_PP=0 && REM Disable pp lookup only, keep tg lookup
```

### 4. Lossless model compression (optional; currently Q4_K / Q6_K / F32 tensors)

```bat
REM Back to the repository root (the scripts live in tools\)
cd /d C:\Lumina-Engine

REM Example: a Q4_K_M model (measured at 94.28%)
python tools\lumina_gguf_pack.py SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf
REM -> SmallThinker-...gguf.lumina (original file kept)

python tools\lumina_gguf_unpack.py verify SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf.lumina SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf

Lumina-Engine-Bin\Lumina-Engine-cli.exe -m SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf.lumina -p "Hello" -n 16 -st -t 6
```

The engine recognizes the `LGGC` magic automatically, so both `.gguf` and `.gguf.lumina` load directly. The first load decompresses to `%TEMP%` and reuses the result. The current packer supports Q4_K / Q6_K / F32 tensors; other quantization types fail with a clear error instead of producing a corrupt container. See [wiki/ModelCompression.en.md](wiki/ModelCompression.en.md).

### 5. QLoRA fine-tuning of quantized models

`Lumina-Engine-train` trains low-rank adapters directly on a quantized base and writes a standard LoRA GGUF (replace `data\...` with your dataset paths):

```bat
cd /d C:\Lumina-Engine\Lumina-Engine-Bin

Lumina-Engine-train.exe -m ..\Bonsai-8B.gguf ^
  --dataset data\train.jsonl --dataset-val data\val.jsonl ^
  -c 8192 -b 2048 -ub 128 --cpu-cores 0 ^
  -lr 5e-5 --epochs 2 -lora-rank 64 -lora-alpha 128 ^
  --save-steps 500 --keep-last 3 ^
  -o out\stage2.lora.gguf
```

```bat
REM Attach the adapter at inference time (no merge required)
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf --lora out\stage2.lora.gguf -p "Hello" -n 128 -t 6
```

- JSONL datasets: each line is auto-detected as `{"text": ...}`, `{"messages": [...]}`, or `{"conversations": [...]}`; by default only assistant replies contribute to the loss.
- Supports **dense and MoE** models (including 3D expert adapters and the router).
- Continue from an existing adapter with `--init-lora`; resume with `--resume`; manage checkpoints with `--save-steps` / `--save-every-epochs` / `--keep-last`.
- `--cpu-cores 0` (default) uses **all logical cores**, about **1.3×** faster than physical cores alone, with bit-identical loss.
- Long-context training runs at `-c 32768` even on 8 GB (attention recomputation during backward + dataset mmap).

A typical three-stage chain is `base → stage1 (continued pretraining) → stage2 (instruction tuning, --init-lora stage1) → stage3 (persona, --init-lora stage2)`. Keep `-lora-rank` / `-lora-alpha` / `-lora-targets` identical across stages; the final adapter already contains the earlier stages.

The full three-stage pipeline is in **[TRAINING_PIPELINE.md](TRAINING_PIPELINE.md)** (Chinese); parameters and implementation details are in **[wiki/Training.en.md](wiki/Training.en.md)**.

---

## Benchmarks

> The data below covers the **Q1_0-specific path**. See [Optimization scope](#optimization-scope) for Vulkan, multi-CPU-variant, training, and compression gains.

**Machine**: Intel Core i5-10400 (6 cores / 12 threads) · 8 GB single-channel DDR4-2666 (streaming read ~14–17 GB/s) · Windows 10 19045
**Model**: Bonsai-8B.gguf (Qwen3-8B architecture, all weights Q1_0, 1.07 GiB) · `-r 3`

### Prompt processing (pp, 6 threads)

| Test | Generic path | Optimized | Gain |
|------|-------------:|----------:|-----:|
| pp32  | 13.18 | **18.78** | **+42.5%** |
| pp64  | 13.14 | **18.46** | **+40.5%** |
| pp128 | 13.11 | **17.94** | **+36.8%** |
| pp256 | 12.98 | **17.31** | **+33.4%** |
| pp512 | 12.83 | **16.49** | **+28.5%** |

> Generic path = `GGML_NO_Q1_TILE=1` (still includes `vpsignb` and VNNI).

### Token generation (tg, 12 threads, interleaved A/B)

| Test | Old | New (single-column GEMV) | Gain |
|------|----:|-------------------------:|-----:|
| tg32  | 11.27 | **12.67** | **+12.4%** |
| tg64  | 11.22 | **12.83** | **+14.3%** |
| tg128 | 11.17 | **12.69** | **+13.6%** |

### Binary lookup Q1LUT (local AVX2, no VNNI, controlled A/B)

| Scenario | Packed kernel | Q1LUT | Gain |
|----------|--------------:|------:|-----:|
| tg (`-p 64 -n 128 -t 6`) | 9.70 / 9.99 | **13.04 / 13.01** | **~+30%** (at the single-channel bandwidth ceiling) |
| pp512 | ~15.8 | **~19.9** | **+26%** |
| pp1024 | 19.68 | **24.82** | **+26%** |
| pp2048 | 14.58 | **22.86** | **+57%** |

Default policy: **tg uses the lookup path; pp uses it when `ne11 >= 8` and the tiled GEMM for `2..7`**. The cost is about 0.7 s to build the table on first use and ~0.95 GiB of extra memory.

### Correctness

| Check | Result |
|-------|--------|
| `test-backend-ops` Q1_0 MUL_MAT | **47 / 47 pass** |
| Perplexity (Bonsai whitepaper corpus, 4 chunks, c=512) | Q1LUT **5.7049** / tiled path **5.7173** (generic 5.7160) |
| Greedy decoding | Matches the generic path (only float summation order differs, ~1e-6) |
| Sampling parameters | `--temp / --top-k / --top-p / --seed` work normally |

Recommended sampling parameters (same as the model publisher): `temperature 0.5–0.7`, `top-k 20–40`, `top-p 0.85–0.95`, `repetition penalty 1.0`.

Full data, methodology, and reproduction commands: [wiki/Benchmarks.en.md](wiki/Benchmarks.en.md), [wiki/Correctness.en.md](wiki/Correctness.en.md).

---

## Automatic device policy (CPU / GPU)

One prebuilt package decides between CPU and GPU automatically; an explicit `-ngl` always wins.

| Machine / model | Automatic policy |
|---|---|
| No GPU | Pure CPU |
| Integrated GPU + dense model | Layers stay on CPU + Vulkan operator offload (pp ~2.4×) |
| Integrated GPU + **MoE model** | Pure CPU (operator offload makes MoE ~30% slower) |
| Discrete GPU | Full-layer offload according to VRAM |
| **Q1_0 + discrete GPU with DP4A** | Automatic offload with Vulkan Q1_0 GEMV enabled |

The Vulkan Q1_0 DP4A GEMV measures **3.7× (Bonsai-1.7B) / 7.1× (Bonsai-8B)** tg versus native Vulkan GEMV, and the author verified correct output on a discrete Turing GPU. The kernel produces incorrect end-to-end output on Intel UMA integrated GPUs, so it is **disabled on integrated GPUs by default**. Manual switch: `LUMINA_Q1_VK_GEMV=0/1`.

Training is forced onto pure CPU by default (the Vulkan backend has no backward kernels); `LUMINA_TRAIN_ALLOW_GPU=1` is experimental only.

---

## Compatibility

`Lumina-Engine-Bin/` ships 14 CPU backend DLLs and selects one automatically via CPUID at runtime:

```text
x64  sse42  sandybridge  ivybridge  piledriver  haswell  skylakex
cannonlake  cascadelake  icelake  cooperlake  alderlake  zen4  sapphirerapids
```

- AVX2 CPUs use the register-tiled GEMM / single-column GEMV / Q1LUT.
- CPUs with VNNI additionally use `vpdpbusd`.
- Older CPUs fall back to the generic Q1_0 kernel (still correct).
- 9 common variants are built with MSVC and 5 Clang-only variants with Clang 20; mixing them has been verified.

See [wiki/Compatibility.en.md](wiki/Compatibility.en.md) for details.

---

## Build from source

### Multi-variant (for distribution)

```bat
cmake -S llama.cpp-b10991 -B build-release -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ^
  -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON ^
  -DGGML_NATIVE=OFF -DGGML_RPC=ON ^
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build build-release -j
```

Then assemble into `Lumina-Engine-Bin\` with `bench\assemble_bin.bat` (it uses `build-clang\bin` as the base; without a Clang build it uses the MSVC base plus the backed-up variants).

> If `D:\VulkanSDK\sdk` is detected, the build enables Vulkan automatically; otherwise it falls back to CPU-only.

Build trees, artifact layout, and troubleshooting: [wiki/Building.en.md](wiki/Building.en.md), [wiki/Delivery.en.md](wiki/Delivery.en.md).

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GGML_NO_Q1_TILE=1` | off | Disable tiled / GEMV / lookup paths and fall back to the generic Q1_0 kernel (A/B or troubleshooting) |
| `GGML_NO_Q1_GEMV=1` | off | Disable only the tg single-column GEMV |
| `GGML_Q1_GEMV_ROWS=1\|2\|4` | `1` | Row-group size for single-column GEMV |
| `GGML_Q1_LUT=0` | on | Disable the entire binary lookup path; saves ~0.95 GiB of memory |
| `GGML_Q1_LUT_PP=0` | on | Disable pp lookup only (pp falls back to the 2×4 tiled GEMM) |
| `GGML_Q1_LUT_KB=N` | `0` | Lookup K-block size (0 = whole table, fastest measured; >0 is experimental) |
| `GGML_Q1_LUT_DEBUG=1` | off | Print lookup-cache allocation / fill information |
| `LUMINA_Q1_VK_GEMV=0\|1` | auto | Vulkan Q1_0 DP4A GEMV: force off / on (default: on for discrete GPUs, off for integrated GPUs) |
| `LUMINA_Q1_VK_CACHE_MB=N` | `2048` | Vulkan Q1_0 weight-repack cache limit (MB, `0` = unlimited) |
| `LUMINA_Q1_VK_INT8=1` | off | Experimental int8 prefill path |
| `LUMINA_TRAIN_ALLOW_GPU=1` | off | Allow training on GPU (experimental, not recommended) |
| `LUMINA_OP_PROF=1` | off | Print training operator time profile |

---

## Architecture and layout

```text
Bonsai-8B.gguf / .gguf.lumina
        │
        ▼
┌──────────────────────────────────┐
│ Model load (transparent LGGC)    │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│ Backend selection (CPUID/Vulkan) │
└──────────────────────────────────┘
        │
        ├── Q1_0 + AVX2 ──► Q1LUT / 2×4 tiled GEMM / single-column GEMV
        └── other models ─► upstream llama.cpp generic kernels
        │
        ▼
 cli / server / bench / perplexity / train / quantize ...
```

```text
Lumina-Engine/
├─ llama.cpp-b10991/          # Modified source tree
│  ├─ ggml/src/ggml-cpu/q1t.cpp      # Q1_0 tiled GEMM + single-column GEMV + Q1LUT
│  ├─ src/lumina-gguf.cpp            # .gguf.lumina lossless container decoder
│  └─ tools/lumina-train/            # C++ QLoRA training engine
├─ Lumina-Engine-Bin/         # ★ Prebuilt delivery (54 files / 14 variants)
├─ bench/                     # Kernel microbenchmarks + build / assemble scripts
├─ tools/                     # GGUF compression, unpacking, inspection scripts
├─ wiki/                      # Detailed docs in Chinese and English
├─ TRAINING_PIPELINE.md       # Three-stage training pipeline (Chinese)
├─ OPTIMIZATION_LOG.md        # Per-build optimization log and A/B data (Chinese)
├─ HANDOFF.md                 # Development handoff notes (Chinese)
├─ README.md                  # Chinese README
├─ README.en.md               # English README
└─ CONTRIBUTORS.md            # Contributors and upstream acknowledgements
```

---

## FAQ

**Do I need a GPU?**
No. The default is the pure-CPU path; Vulkan is optional acceleration, and the engine falls back automatically when no Vulkan device is present.

**Is NVIDIA CUDA supported?**
The prebuilt delivery is **CPU + Vulkan**. For CUDA, build on top of upstream llama.cpp yourself; the Q1_0-specific CPU kernels are x86 paths, and there is also a Vulkan Q1_0 path.

**Do the optimizations change model output?**
No. Weights are byte-identical and Q1_0 bit operations are exact; only the float accumulation order differs, at the 1e-6 level, and perplexity stays in the same range.

**Why do I only get a dozen tokens per second?**
Each token for an 8B model requires reading about 1.07 GiB of weights, so speed is determined by memory bandwidth. Single-channel DDR4 tops out around 15.7 tok/s; dual-channel or faster memory helps the most.

**Will it run on an old CPU?**
Yes. The minimum is SSE4.2, with an automatic fallback to the generic kernel; AVX2 is needed for the full benefit of the tiled GEMM / Q1LUT paths.

**Where do I download the model?**
The repository does not bundle models. Get the Bonsai GGUF (Q1_0) series from [PrismML](https://prismml.com) and place it in the repository root.

**Is Linux supported?**
The source is cross-platform, but the prebuilt delivery and performance testing here focus on Windows x86-64; on Linux, build it yourself with CMake.

**Can I fine-tune a 1-bit model?**
Yes. The quantized base stays frozen and only LoRA adapters are trained (QLoRA), supporting Q1_0 / Q4_K / Q8_0 and MoE.

**Are these optimizations 1-bit-only?**
No. The Q1_0-specific kernels only affect 1-bit models; the 14 CPU variants, Vulkan, automatic device policy, lossless compression, and QLoRA training all apply to other models / quantization types. See [Optimization scope](#optimization-scope).

---

## Changelog

| Build | Highlights |
|------:|------------|
| **24** | Unified startup banner across all console tools (ASCII logo + build number + project links) |
| 23 | Training uses all logical cores by default, ~**1.3×** faster with bit-identical loss |
| 22 | Fixed ETA growing after `--resume`; added `--dataset-stride`, which can halve epoch time |
| 21 | Fixed `--resume` restarting from step 1; resumes at the checkpoint's datapoint / microbatch |
| 20 | Long-context training RAM optimization: `-c 32768` runs stably on 8 GB without swapping |
| 19 | Vulkan Q1_0 DP4A GEMV port + automatic backend / model selection |
| 12–16 | Vulkan backend, machine-aware device policy, training acc fix, unified banner |
| 6–8 | `.gguf.lumina` lossless container (v1 → v2, 94.28%) |

See [HANDOFF.md](HANDOFF.md) and [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md) (both Chinese) for the full per-build notes.

---

## Documentation

| Document | Contents |
|----------|----------|
| [wiki/Home.en.md](wiki/Home.en.md) | Documentation overview and one-page summary |
| [wiki/Background.en.md](wiki/Background.en.md) | Bonsai / Q1_0 / GGUF background and bottleneck analysis |
| [wiki/Optimizations.en.md](wiki/Optimizations.en.md) | Principles and implementation of each optimization (including Q1LUT) |
| [wiki/Building.en.md](wiki/Building.en.md) | Single-variant / multi-variant / dual-toolchain builds |
| [wiki/Delivery.en.md](wiki/Delivery.en.md) | Delivery directory, assembly flow, and conventions |
| [wiki/Benchmarks.en.md](wiki/Benchmarks.en.md) | Full benchmark data and methodology |
| [wiki/Correctness.en.md](wiki/Correctness.en.md) | Numerical equivalence and backend tests |
| [wiki/Compatibility.en.md](wiki/Compatibility.en.md) | 14 CPU variants and distribution |
| [wiki/ModelCompression.en.md](wiki/ModelCompression.en.md) | Compressing models / using compressed models (user guide) |
| [wiki/LosslessCompression.en.md](wiki/LosslessCompression.en.md) | Lossless container format and principles |
| [wiki/Training.en.md](wiki/Training.en.md) | QLoRA training engine parameters and implementation |
| [wiki/Changes.en.md](wiki/Changes.en.md) | Per-file code changes |
| [wiki/Troubleshooting.en.md](wiki/Troubleshooting.en.md) | Common issues and troubleshooting |

Every page also has a Chinese version (`wiki/*.md`). The Chinese README is [README.md](README.md).

---

## Support this project

Every performance number in Lumina-Engine comes from repeated, interleaved A/B runs on real hardware; every "enabled by default" decision is backed by correctness, perplexity, and compatibility checks. The project is maintained in the author's own time: no ads, and every core feature stays open source. A full 14-variant build and regression takes hours, and adapting to a new CPU / GPU platform often requires dozens of A/B runs.

If it has saved you time, please consider supporting its continued development:

### [❤️ Support Lumina-Engine on Afdian →](https://afdian.com/a/lumi_380)

Your support is prioritized for:

- **Broader hardware coverage**: testing and tuning on AVX-512 / VNNI / AMD / NVIDIA platforms
- **Longer test runs**: not just one pass, but enough repetitions, 32K-context runs, and full-model regressions
- **Faster upstream tracking**: llama.cpp updates, model format changes, and issue reproduction
- **Better documentation**: benchmark data, build scripts, and troubleshooting notes

You can also help without spending money:

- ⭐ Star the repository so more people can find it
- 📊 Share your hardware benchmarks or reproduction results
- 🐛 File high-quality issues / PRs
- 🔔 Follow updates on [Bilibili](https://space.bilibili.com/3546376302496335)

> Sponsorship is entirely optional. Lumina-Engine stays AGPL-3.0 open source either way.

---

## Contributing

Issues and PRs are welcome. The project's core principle is **correctness first, then speed**: every optimization must pass A/B and perplexity checks and must not affect non-Q1_0 paths. See [CONTRIBUTORS.md](CONTRIBUTORS.md#english) for the contributor list and acknowledgements.

```bat
git clone https://github.com/Happy-380/Lumina-Engine.git
cd Lumina-Engine

REM Configure + build + assemble (edit the toolchain and repo paths in the scripts for your machine)
configure-release.bat
build-release.bat
bench\assemble_bin.bat

REM Basic verification
Lumina-Engine-Bin\Lumina-Engine-cli.exe -m Bonsai-8B.gguf -p "The capital of France is" -n 8 -st
Lumina-Engine-Bin\Lumina-Engine-perplexity.exe -m Bonsai-8B.gguf -f wiki\Background.md -c 512 -t 6 --chunks 4
```

`test-backend-ops` requires building with `-DLLAMA_BUILD_TESTS=ON` (the default delivery scripts disable tests).

Please include as much of the following as possible:

- **Same-session A/B** data before and after the change (`GGML_NO_Q1_TILE=1` as the baseline), with CPU, memory, thread count, and model noted.
- Correctness evidence: `test-backend-ops` (Q1_0 MUL_MAT), perplexity, and a fixed-seed greedy output comparison.
- If the change touches build configuration, the MSVC / Clang / CMake versions used.

Kernel microbenchmarks live in `bench/` (for example `bench_lut2.cpp`, `bench_gemv128.cpp`); the build and assembly scripts are there as well.

---

## License and acknowledgements

- Modifications and new code in this repository are released under **GNU AGPL-3.0**; see [LICENSE](LICENSE).
- Upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) is **MIT licensed**; upstream files retain their original copyright and license notices.
- The Q1_0 quantization format and Bonsai models come from [PrismML](https://prismml.com).
- The built-in DEFLATE decoder uses [miniz](https://github.com/richgel999/miniz) (MIT / public domain).
- See [CONTRIBUTORS.md](CONTRIBUTORS.md#english) for the complete contributor list and upstream acknowledgements.

> If you make a modified version available to users over a network, AGPL-3.0 requires you to offer them the corresponding source code.
