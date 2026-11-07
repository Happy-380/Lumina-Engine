# Optimization Details

> 🌐 [中文](Optimizations.md) | **English**

This page describes the core optimizations added/changed in this project (7 in total). All of them
apply only on the `GGML_TYPE_Q1_0` path and do not change the behavior of any other quantization
type or backend.

> ⚠️ All performance figures here come from a **single-channel memory** test machine (8 GB DDR4-2666,
> streaming read bandwidth ~14–17 GB/s). tg is bound by memory bandwidth, so its absolutes are low;
> on dual-channel / higher-bandwidth platforms the tg gain approaches the kernel-level value.

---

## Optimization 1: Per-128 activation quantization (Q8_128)

### Problem

The generic Q1_0 path uses **Q8_0** activations (one fp16 scale per 32 values). A Q1_0 block is
128 weights = 4 Q8_0 sub-blocks, and the four scales `d1_0..d1_3` differ. Every sub-block therefore
needs its own float conversion:

```
Q8_0:  per 128 weights
  4 × ( vpsignb + dot + cvtdq2ps + fmadd + scale )    ← 4 float conversions
```

### Approach

Change the tile path's activation quantization to **one scale per 128 values** (Q8_128):

```c
#define QK8_128 128
typedef struct {
    ggml_half d;        // 2 bytes
    int8_t    qs[128];  // 128 bytes
} block_q8_128;         // 130 bytes / 128 values
```

The four sub-blocks now share one `d1`, so their **int32 dot products can be accumulated first**,
followed by a single float conversion:

```
Q8_128: per 128 weights
  4 × ( vpsignb + dot )  +  1 × ( cvtdq2ps + fmadd + scale )
                          ↑ float conversions 4 → 1
```

### Why a local format instead of a new GGML type

Adding `GGML_TYPE_Q8_128` would require touching `ggml.h`, `ggml.c` and the type registries of
**175+ backend files**, a large risk that also affects CUDA/Metal/Vulkan, etc.

The implementation therefore uses Q8_128 **only inside the q1t path**:

- `ggml_compute_forward_mul_mat_q1t()` quantizes `src1->data` (F32) into Q8_128 in parallel, writing
  into shared scratch (in `params->wdata`, after the Q8_0 conversion region), followed by a
  `ggml_barrier` sync.
- The generic path (tg, tail columns, tail rows) still uses the framework's pre-quantized Q8_0 with
  identical numerics.

**Overhead is negligible**: quantization touches `ne11 × ne00` elements while the matmul performs
`ne01 × ne11 × ne00` MACs — a ratio of `1 / ne01` (a few thousandths).

**Accuracy**: per-128 is coarser than per-32, but measured perplexity did not rise (see
[Correctness](Correctness.en.md)). This matches the K-quant practice of using per-256 scales.

---

## Optimization 2: Register-tiled GEMM (2 rows × 4 columns)

### Problem

The default `ggml_compute_forward_mul_mat` calls `vec_dot` once per (output row, input column),
**re-expanding the 1-bit sign bits every time**. For an N-column prompt the same bit expansion is
repeated N times.

### Approach

Compute R rows and T columns in registers simultaneously; the sign bits are **expanded once per R
rows per 32 weights** and reused across T columns, **without spilling to memory** (which is exactly
why the earlier “memory panel” approach failed — it wrote an extra byte per weight and read it back).

Core loop (`q1t_dot<R,T>`):

```c
for (ib : each 128-weight block) {
    for (k : 4 sub-blocks) {
        sg[R] = expand(x[r][ib].qs[k]);        // expand once
        for (t : T columns) {
            qy = load(y[t][ib].qs + k*32);
            for (r : R rows)
                t32[r][t] += vpdpbusd(ones, sign(qy, sg[r]));   // int32 accumulate
        }
    }
    for (t,r)
        acc[r][t] = fmadd(d0[r]*d1[t], cvtdq2ps(t32[r][t]), acc[r][t]);  // once per 128 weights
}
```

### Choosing the tile size

`t32[R][T]` (int32) and `acc[R][T]` (float) each use `R×T` vector registers, plus `sg[R]`, `d0v[R]`,
etc., so register pressure grows with `R×T`. Measured (local AVX2, pp128):

| R×T | pp128 (t/s) |
|----:|------------:|
| 4×1 | 12.30 |
| 4×2 | 14.55 |
| 4×3 | 15.27 |
| 3×3 | 15.67 |
| 4×4 | 15.85 |
| **2×4** | **16.36** |

The final choice is **R=2, T=4**: T=4 amortizes the sign expansion, R=2 keeps register pressure in check.

---

## Optimization 3: VNNI (`vpdpbusd`)

### Principle

On AVX2, summing four int8 products takes two instructions:

```
vpmaddubsw  ones, sy   → int16 pairwise sums
vpmaddwd    ., ones16  → int32
```

VNNI's `vpdpbusd(acc, a_unsigned, b_signed)` computes `acc += a·b` directly (accumulating int32 per
group of 4 bytes). With `a = ones` (unsigned 1) and `b = sy` (signed ±q), one instruction suffices:

```
vpdpbusd  acc, ones, sy   → acc += Σ of 4 sy values
```

### Coverage

| ISA | Macro | Intrinsic | CPU |
|--------|----|-----------|-----|
| AVX512-VNNI + VL | `__AVX512VNNI__` | `_mm256_dpbusd_epi32` | Cascade Lake / Ice Lake / Cooper Lake / Zen 4 / Sapphire Rapids |
| AVX-VNNI | `__AVXVNNI__` | `_mm256_dpbusd_avx_epi32` | Alder Lake |
| none | — | fall back to `vpmaddubsw + vpmaddwd` | Haswell / Skylake-X / Cannon Lake, etc. |

Confirmed by disassembly (see [Benchmarks](Benchmarks.en.md#4-vnni-instruction-confirmation)):
`cascadelake` has 797 `vpdpbusd`, `alderlake` 473, and `haswell` 0 (it uses 993 `vpmaddubsw`).

---

## Optimization 4: `vpsignb` (“keep +1 / negate −1”)

The core semantics of a 1-bit weight are: **sign bit 1 → keep the activation; 0 → negate it**. That
is exactly what `vpsignb` does.

### Before

```c
sm = cmpeq(and(shuffle(...), bit_masks), zero);   // 0xFF where bit=0
sy = sub(xor(qy, sm), sm);                         // two instructions, dependency chain 2
```

### After

```c
sg = or(sm, ones);                     // 0xFF (-1) or 0x01 (+1)
sy = _mm256_sign_epi8(qy, sg);         // one instruction
```

The dependency chain shrinks from 2 to 1. Standalone micro-benchmark (single thread, hot data):

| Kernel | GMAC/s |
|------|-------:|
| `xor+sub` (original) | 16.76 |
| **`vpsignb`** | **18.53** (+10.6%) |

Applied to both:
- the generic kernel `ggml_vec_dot_q1_0_q8_0` (`arch/x86/quants.c`, AVX2 branch)
- the tiled kernel `q1t_dot` (`q1t.cpp`)

---

## Optimization 5: Multi-microarchitecture build

With `-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON` the CPU backend is compiled into **one DLL per
microarchitecture**, and the `ggml` loader picks the best match at runtime based on CPU features:

```
x64  sse42  sandybridge  ivybridge  piledriver  haswell  skylakex
cannonlake  cascadelake  icelake  cooperlake  alderlake  zen4  sapphirerapids
```

- AVX2 and above: the register-tiled GEMM is enabled.
- VNNI: additionally enables `vpdpbusd`.
- Older CPUs: fall back to the generic Q1_0 kernel (correct but slower).

See [Compatibility](Compatibility.en.md).

---

## Instruction count comparison (per 128 weights, AVX2)

| Stage | Generic Q8_0 | Tiled Q8_128 | Note |
|------|----------:|------------:|------|
| Sign expansion | 4 × 5 = 20 / T | 4 × 5 = 20 (amortized over 4 columns) | once per 4 columns |
| Apply signs | 4 | 2 | R=2 |
| Dot product | 4×(1~2) | 2×(1~2) | VNNI saves 1 each |
| Float conversion | 4 | 1 | **the key per-128 win** |
| Scale computation | 4 | 1 | |

Net measured pp gain: **29%–43%**.

---

## Optimization 6: Single-column GEMV (token generation)

### Problem

tg decodes one token at a time (`src1->ne[1] == 1`) and does not enter the tiled path above; it goes
through the generic `ggml_vec_dot_q1_0_q8_0`. That kernel uses Q8_0 activations (one scale per 32
weights), so a 128-weight Q1_0 block needs **4 float conversions** (one per sub-block), each with an
fmadd dependency chain.

### Approach

Reuse the same per-128 activation (Q8_128) for tg: accumulate the four sub-blocks' int32 dot products
first and perform **one float conversion** per block. The row group size `R` defaults to 1
(`q1t_gemv<1>`) and can be tuned with `GGML_Q1_GEMV_ROWS=2|4` to share activation loads across R rows.

Activation quantization is also made **AVX2-vectorized and distributed across threads by 128-value
block**: with a single column there is only one activation row, so a row-based split would leave one
thread working while the rest idle at the barrier; block-based splitting parallelizes all threads.

### Results

| Platform | Kernel (micro-bench, single thread) | Full-model tg (12 threads, same-run A/B) |
|------|----------------------:|------------------------------:|
| i5-10400 (AVX2, no VNNI) | ~18.5 → ~24 GMAC/s (+30%) | ~11.2 → ~12.8 t/s (+13%) |

Single-channel memory bandwidth is the hard ceiling for tg on this machine (`1.06 GiB / bandwidth`);
on platforms with more bandwidth the gain approaches the ~30% kernel-level improvement.

---

## Optimization 7: Binary lookup table (Q1LUT, enabled by default)

> Enabled by default; `GGML_Q1_LUT=0` disables the whole LUT path, `GGML_Q1_LUT_PP=0` disables only
> the prompt-processing (pp) LUT. All code lives in `q1t.cpp`; no third-party implementation is used.

### Idea (inspired by BitNet TL, adapted to binary)

BitNet's weights are ternary `{-1,0,+1}`, which a single `vpsignb` cannot express, so the TL kernel
precomputes weights into “lookup by activation”. Bonsai's Q1_0 weights are strictly binary
`{-1,+1}`: the sign bit directly means keep/negate, so a literal copy of BitNet's “one 16-byte table
per weight” would cost 16 bytes/weight (~128× bloat) and lose on a bandwidth-bound CPU.

The correct direction for binary weights is an **activation-side** table: a group of 4 consecutive
weights has only `2^4 = 16` sign combinations, so for a given activation vector we precompute, per
group of 4 activations, a 16-entry table

```
T[c] = Σ_{i=0..3} ( bit_i(c) ? +a[i] : -a[i] )
```

The partial dot product of those 4 weights is then a single `vpshufb` (indexed by the packed 4-bit
weight code). The table is built once per activation vector and reused across all output rows of the
layer — that is where a binary LUT actually amortizes cost.

### Implementation notes

- Table entries are `int16`, stored as “16 low bytes + 16 high bytes”; one lookup = two `vpshufb`.
- **True binary symmetry (generation side)**: negating all 4 weight bits flips every entry's sign,
  i.e. `T[c ^ 0xf] = -T[c]`. So only `c = 0..7` are summed; the other 8 entries are filled by
  negation, roughly halving generation work.
- Weights are permuted once into the layout the kernel needs: per 32 rows × per 4 K positions, 16
  bytes holding 32 4-bit codes (2 codes per byte). This is a pure permutation; storage stays at
  1 bit/weight.
- The permutation runs lazily per tensor and is filled in parallel across row groups (first use
  ≈ 0.7 s); the result is cached in-process.
- The permutation also collects the per-128 fp16 scale `d0` into a contiguous array so the GEMV no
  longer touches the original tensor (otherwise it would read it in 576 B strides, doubling DRAM traffic).
- The activation table accumulates `int16` per 128-weight block (upper bound `32×4×127 = 16256`, no
  overflow) and is scaled by `d0·d1` at the end.

### L1 constraint and K blocking

The full table is `K/4 × 32 B = 8 B/weight`: at K=4096 that is 32 KiB (fills the 32 KiB L1d), at
K=12288 it is 96 KiB (spills to L2). Following BitNet TL, an optional K blocking
(`GGML_Q1_LUT_KB`, block table ≤ 16 KiB) was implemented. K-blocking comparison
(`-p 64 -n 128 -t 6`): KB=1024 ~11.0, KB=2048 ~11.1, KB=4096 ~11.9, full table (KB=0) ~12.0 — the
**full-table direct-write default is fastest**; K blocking is kept only as an experiment.

### End-to-end measurements (re-check, 2025-09-23)

An earlier run concluded “tg flat, pp LUT slower”; a controlled re-check (same-run alternating A/B,
with warm-up and ordering controlled) showed that was an artifact of downclocking/background load.
The correct conclusion:

- **tg**: the packed GEMV is compute bound (~9.8 t/s); the LUT kernel pushes it to the single-channel
  DRAM bandwidth ceiling (~13 t/s). Controlled A/B on this machine (`-p 64 -n 128 -t 6`, off/lut
  alternating): **9.70/9.99 → 13.04/13.01 t/s (~+30%)**.
- **pp**: from `ne11 >= 8` the LUT is uniformly faster than the 2×4 tiled GEMM, with a larger margin
  as batch grows:

| Test | Tiled GEMM (t/s) | Q1LUT pp (t/s) | Change |
|---|---:|---:|---:|
| pp8    | 16.9 ~ 18.3 | 20.8 ~ 23.7 | ~ +27% |
| pp64   | ~ 17.0 | ~ 21.4 | +26% |
| pp512  | ~ 15.8 | ~ 19.9 | +26% |
| pp1024 | 19.68 | 24.82 | +26% |
| pp2048 | 14.58 | 22.86 | +57% |

The default policy therefore selects **by the actual number of tokens**: `ne11 == 1` (tg) and
`ne11 >= 8` (pp) use the LUT; `ne11 = 2..7` uses the 2×4 tiled GEMM (the table is not yet amortized
there, and the generic path also uses less memory). `GGML_Q1_LUT_PP=0` forces pp back to tiled GEMM.
Cost: a one-time per-tensor table generation (≈0.7 s) and ≈0.95 GiB extra memory.

### Single-thread kernel micro-benchmark

`bench/bench_lut2.cpp` (self-contained independent implementation with correctness checks, plus a
K-blocking variant):

| Kernel | GMAC/s |
|---|---:|
| Existing sign GEMV (Q8_128, single column) | 25.9 |
| **Binary activation-LUT GEMV (32 rows, full table)** | **46.2 (+78%)** |

The kernel is fast; end-to-end tg is bandwidth bound, so the kernel speed is masked by memory.

### Conclusion

- **tg uses the LUT by default**: the packed GEMV is compute bound, and the LUT gives ~**+30%**
  (reaching the single-channel bandwidth ceiling).
- **pp uses the LUT by default for `ne11 >= 8`**, uniformly faster than 2×4 tiled GEMM (pp2048 +57%);
  `2..7` still uses tiled GEMM.
- Numerically correct: PPL is on the same level (LUT 5.7049 vs. disabled 5.7173; the difference comes
  from float summation order); greedy output over long prompts is byte-identical.
- K blocking is implemented and verified to have no benefit; L1 table overflow is not the bottleneck.
- Cost: ~0.95 GiB extra memory, ~0.7 s one-time permutation; `GGML_Q1_LUT=0` disables it entirely.
- Do not use more threads than physical cores; this repo defaults to physical cores (no SMT).

---

## Optimization 8: whole-block prefetch in the Q4_K GEMV (Build 31)

This one applies to **Q4_K / Q4_K_M quantized models** (not Q1_0). `ggml_gemv_q4_K_8x8_q8_K()` is the
repacked Q4_K decode kernel (`ggml/src/ggml-cpu/arch/x86/repack.cpp`) and CPU single-token generation
spends virtually all of its time there.

**Diagnosis**: it only reaches 6.0-7.5 GB/s per thread, while a plain streaming read on the same thread
tops out at 11.6-13.1 GB/s. That gap means it is **memory-latency bound**, not bandwidth bound: the
eight 32 B loads of one `block_q4_Kx8` (1152 B) wait for the arithmetic of the previous block, and the
out-of-order window (~224 uops) does not cover a 200-300 cycle memory latency.

**Fix**: at the top of the block loop, pull the whole block that will be consumed **two iterations
later** (18 cache lines) into L1:

```c
{
    const char * pf = (const char *) (b_ptr + b + 2);
    for (size_t pf_off = 0; pf_off < sizeof(block_q4_Kx8); pf_off += 64) {
        _mm_prefetch(pf + pf_off, _MM_HINT_T0);
    }
}
```

**Pitfall: do not add an `if (b + 2 < nb)` guard.** The branch inside the hot loop (or the loop
optimisation it prevents) costs much more than the few useless prefetches at the end of the matrix -
measured **-13%** (5.49 vs 6.34 GB/s). `PREFETCHh` never faults on an invalid address, so running past
the end of the tensor is harmless.

**Effect** (i5-10400 / 8 GB, a training job in the background, medians of interleaved in-process runs):

| Metric | Before | After |
|---|---:|---:|
| kernel bandwidth, single thread (18 MiB matrix, DRAM resident) | 6.00-6.47 GB/s | 6.63-6.78 GB/s (**+13.0%**) |
| end-to-end decode `tg32` (Qwen3.5-0.8B Q4_K_M, `-t 4`) | 13.66 t/s | 15.53 t/s (**+13.7%**) |
| prefill `pp128` | 79-89 t/s | 87-89 t/s (unchanged, the GEMM path is untouched) |
| L3 resident small matrix (0.56 MiB) | 10.07 GB/s | 10.30 GB/s (within noise) |

The output is **bit-identical** to the previous kernel (`memcmp`; greedy-decoded text has the same
md5). Measurement/reproduction: `llama.cpp-b10991/tests/bench-cpu-kern.cpp` and
`bench/build_bench_cpu_kern.bat`.

---

## Why there is no AVX-512 512-bit tile

Considered and rejected:

1. The Q8_0/Q8_128 block layout is 34/130 bytes, so 64-byte-aligned loads are impossible and would
   need 2 loads + insert.
2. Each 512-bit half-block needs a `[d1_0 ×8, d1_1 ×8]` vector plus an extra `mul`, offsetting the
   width advantage.
3. This machine has no AVX-512, so runtime correctness cannot be verified; introducing untested
   crash risk violates the “must run correctly” requirement.

AVX-512 CPUs use the **VNNI 256-bit path**, which already gains at the instruction level.
