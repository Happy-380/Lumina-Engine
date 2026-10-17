# Benchmark Results & Methodology

> 🌐 [中文](Benchmarks.md) | **English**

## 1. Test environment

| Item | Value |
|------|----|
| CPU | Intel Core i5-10400 (Comet Lake, 6C/12T, 2.90 GHz, no AVX-512 / VNNI) |
| Memory | 8 GB **single-channel** DDR4 (SPD 3200, running at 2666 MT/s) |
| OS | Windows 10 22H2 (build 19045) |
| Compilers | MSVC 14.51 (VS2026) + Clang 20.1.8 (LLVM) |
| Model | `Bonsai-8B.gguf` (Qwen3-8B, all Q1_0, 1.07 GiB) |
| Threads | 6 (physical cores) |
| Repetitions | `-r 3` (mean ± stddev per item) |

> This machine uses the `haswell` variant (AVX2, no VNNI). The VNNI path cannot run here, but its
> instruction generation was verified by disassembly.

> ⚠️ **The test machine uses single-channel memory** (8 GB DDR4-2666; measured streaming read bandwidth
> ~14–17 GB/s). **tg is memory-bandwidth bound**, so every tg absolute figure here is capped by the
> single-channel ceiling (theoretical ~15.7 tok/s); pp is compute bound and less affected. When comparing
> across platforms, focus on same-run A/B **ratios**, not local absolutes.
> (The machine also runs other applications, which can further perturb A/B measurements.)

---

## 2. Main result: tiled path vs. generic path

`Lumina-Engine-bench.exe -m Bonsai-8B.gguf -p 32,64,128,256,512 -n 32,64,128 -t 6 -r 3`

**Generic path** (`GGML_NO_Q1_TILE=1`, still with `vpsignb` + VNNI):

| Test | t/s |
|------|----:|
| pp32  | 13.18 ± 0.29 |
| pp64  | 13.14 ± 0.35 |
| pp128 | 13.11 ± 0.18 |
| pp256 | 12.98 ± 0.06 |
| pp512 | 12.83 ± 0.05 |
| tg32  | 9.60 ± 0.17 |
| tg64  | 9.58 ± 0.09 |
| tg128 | 9.53 ± 0.09 |

**Optimized** (Q8_128 tiled path):

| Test | t/s | vs. generic |
|------|----:|--------:|
| pp32  | **18.78 ± 0.55** | **+42.5%** |
| pp64  | **18.46 ± 0.44** | **+40.5%** |
| pp128 | **17.94 ± 0.40** | **+36.8%** |
| pp256 | **17.31 ± 0.27** | **+33.4%** |
| pp512 | **16.49 ± 0.18** | **+28.5%** |
| tg32  | **10.68 ± 0.55** | **+11%** |
| tg64  | **11.01 ± 0.42** | **+15%** |
| tg128 | **10.19 ± 0.25** | **+7%** |

> tg uses a separate single-column GEMV path (see §2.1). The table above is for 6 threads and is very
> sensitive to machine state; for reliable comparisons use same-run alternating A/B.

### 2.1 Token generation (tg) same-run alternating A/B

Old and new CPU backend DLLs run alternately (`-p 64 -n 32,64,128 -t 12 -r 3`, mean of two rounds):

| Test | Old | **New** | Gain |
|------|-----:|---------:|-----:|
| tg32  | 11.27 | **12.67** | **+12.4%** |
| tg64  | 11.22 | **12.83** | **+14.3%** |
| tg128 | 11.17 | **12.69** | **+13.6%** |

At 6 threads the gain is about +4–10% (closer to the single-channel bandwidth ceiling). The row group
size can be tuned with `GGML_Q1_GEMV_ROWS=1\|2\|4`.

### Controlled A/B (3 same-run alternating rounds, to remove machine drift)

`Lumina-Engine-bench -p 32,128,512 -n 1 -t 6 -r 2`

| Round | pp32 generic → tiled | pp128 generic → tiled | pp512 generic → tiled |
|------|------------------|-------------------|-------------------|
| 1 | 13.46 → 16.57 | 13.08 → 16.28 | 12.74 → 16.09 |
| 2 | 13.23 → 16.43 | 13.05 → 16.21 | 12.79 → 16.10 |
| 3 | 13.15 → 16.81 | 12.85 → 16.51 | 12.72 → 16.09 |
| **Mean gain** | **+25%** | **+26%** | **+26%** |

---

## 3. Tile-size sweep

Local AVX2, pp128 (`t/s`):

| R×T | pp128 |
|----:|------:|
| 4×1 | 12.30 ± 0.20 |
| 4×2 | 14.55 ± 0.34 |
| 4×3 | 15.27 ± 0.34 |
| 3×3 | 15.67 ± 0.23 |
| 4×4 | 15.85 ± 0.24 |
| **2×4** | **16.36 ± 0.12** |

**R=2, T=4** was chosen. Re-confirmed (3 same-run rounds):

| Config | Run 1 | Run 2 | Run 3 |
|------|--------:|--------:|--------:|
| 2×4 | 16.41 | 16.39 | 16.30 |
| 4×4 | 15.85 | 15.84 | 15.86 |

---

## 4. VNNI instruction confirmation

```
dumpbin /disasm build-release\bin\ggml-cpu-<arch>.dll | findstr vpdpbusd
```

| Variant | `vpdpbusd` | `vpmaddubsw` |
|------|-----------:|-------------:|
| cascadelake | 797 | 0 |
| alderlake (AVX-VNNI) | 473 | 0 |
| haswell (AVX2 fallback) | 0 | 993 |

The VNNI path is generated correctly, and non-VNNI variants fall back correctly.

---

## 5. Standalone kernel micro-benchmarks

`bench_kern.cpp` (single thread, mean of 8, hot data):

| Kernel | GMAC/s |
|------|-------:|
| `xor+sub` (original) | 16.76 |
| **`vpsignb` (current)** | **18.53** (+10.6%) |
| `vpsignb` + unroll2 | 18.23 |
| `vpsignb` + prefetch | 17.6 (no benefit) |

`bench_tile.cpp` (single thread, MSVC `-O2 -arch:AVX2`):

| Kernel | GMAC/s |
|------|-------:|
| Generic (untiled) | 17.1 |
| Tiled 2×4 | 27.6 |
| Tiled 4×2 | 26.2 |
| Tiled 2×8 | 29.9 |
| Tiled 4×4 | **30.8** |

> The tiled kernel is ~1.8× faster than the generic one in isolation; in the full model the net gain
> is about +25% because the framework already tiles 16×16.

### Memory bandwidth probe

`bench_mem2.cpp` (vectorized read, 1 GB array):

| Threads | Read bandwidth |
|-----:|-------:|
| 1 | 14.1 GB/s |
| 6 | 16.7 GB/s |
| 12 | 16.1 GB/s |

Copy bandwidth is ~4.3–4.8 GB/s. This explains the tg ceiling: `1.06 GB / 16.7 GB/s ≈ 63 ms`
→ theoretical ~15.7 tok/s, actual ~9.5 (the rest is other ops and synchronization). With Q1LUT the
machine reaches ~13 tok/s, closer to the ceiling.

---

## 6. Perplexity

`Lumina-Engine-perplexity.exe -m Bonsai-8B.gguf -f docx/whitepaper.txt -c 512 -t 6 --chunks 4`

| Config | PPL |
|------|----:|
| Generic (Q8_0 activations) | 5.7160 ± 0.507 |
| **Optimized (Q8_128 activations)** | **5.7111 ± 0.507** |
| **Binary LUT (Q1LUT, default)** | **5.7049** |

Per-128 quantization did not increase perplexity; it is marginally lower (within noise).

---

## 7. Machine noise

This is an office machine, so absolute performance drifts with temperature/load (pp512 can range from
13.0 to 16.5 for the same config at different times). Therefore:

- **Only trust same-run alternating A/B ratios**, not cross-session absolutes.
- Micro-benchmarks are more stable than `Lumina-Engine-bench`, but still vary ~10%.
- All conclusions are based on same-run alternating measurements.

---

## 8. Binary lookup table (Q1LUT, enabled by default)

> The 9-23 re-check reversed the earlier “no benefit” conclusion in this section: that was an artifact
> of downclocking/background load. The data below is controlled alternating A/B.

tg (`-p 64 -n 128 -t 6 -r 3`, off/lut alternating):

| Config | Packed GEMV (t/s) | Q1LUT (t/s) |
|---|---:|---:|
| Local (AVX2, no VNNI) | 9.70 / 9.99 | **13.04 / 13.01 (+30%)** |

pp (controlled A/B: `GGML_Q1_LUT=0` vs. default):

| Test | Tiled GEMM (t/s) | Q1LUT pp (t/s) | Change |
|---|---:|---:|---:|
| pp8 | 16.9 ~ 18.3 | 20.8 ~ 23.7 | ~ +27% |
| pp64 | ~ 17.0 | ~ 21.4 | +26% |
| pp512 | ~ 15.8 | ~ 19.9 | +26% |
| pp1024 | 19.68 | 24.82 | +26% |
| pp2048 | 14.58 | 22.86 | +57% |

K-blocking comparison (`GGML_Q1_LUT_KB`, `-p 64 -n 128 -t 6`):

| KB | tg128 (t/s) |
|---:|---:|
| 0 (full table, default) | ~12.0 |
| 1024 | ~11.0 |
| 2048 | ~11.1 |
| 4096 | ~11.9 |

Single-thread kernel micro-benchmark (`bench/bench_lut2.cpp`, K=4096, M=65536, correctness max err ~1e-8):

| Kernel | GMAC/s |
|---|---:|
| Existing sign GEMV (Q8_128) | 25.9 |
| **Binary activation-LUT GEMV (32 rows)** | **46.2 (+78%)** |

Numerical correctness: PPL on the same level (LUT 5.7049 vs. disabled 5.7173; difference from float
summation order); greedy output over long prompts byte-identical.

Conclusion: tg and pp (`ne11>=8`) use Q1LUT by default, while pp (`2..7`) uses the tiled GEMM;
`GGML_Q1_LUT=0` disables it entirely (~0.95 GiB extra memory, ~0.7 s one-time permutation).

---

## 9. Reproduction commands

```bat
cd C:\Lumina-Engine\Lumina-Engine-Bin

REM Binary LUT A/B (on by default; disable with GGML_Q1_LUT=0)
set GGML_Q1_LUT=0
Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 64 -n 128 -t 6 -r 3
set GGML_Q1_LUT=
Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 64 -n 128 -t 6 -r 3

REM Main result
Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 32,64,128,256,512 -n 32,64,128 -t 6 -r 3

REM Generic-path baseline
set GGML_NO_Q1_TILE=1
Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 32,64,128,256,512 -n 32,64,128 -t 6 -r 3
set GGML_NO_Q1_TILE=

REM tg single-column GEMV comparison (old/new DLL alternating)
set GGML_NO_Q1_GEMV=1
Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 64 -n 32,64,128 -t 12 -r 3
set GGML_NO_Q1_GEMV=
Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 64 -n 32,64,128 -t 12 -r 3

REM tg row-group sweep
for %R in (1 2 4) do set GGML_Q1_GEMV_ROWS=%R && Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 64 -n 128 -t 12 -r 3
set GGML_Q1_GEMV_ROWS=

REM Perplexity
Lumina-Engine-perplexity.exe -m ..\Bonsai-8B.gguf -f ..\docx\whitepaper.txt -c 512 -t 6 --chunks 4
```
