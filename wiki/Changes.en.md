# Main Code Changes

> 🌐 [中文](Changes.md) | **English**

This page lists all changes relative to upstream llama.cpp (b10991). The core changes touch **5
files**: 2 new, 3 modified (the full binary lookup table Q1LUT lives inside the new `q1t.cpp`). All
changes affect only the `GGML_TYPE_Q1_0` + AVX2 path; other models/quantizations/backends are
unaffected.

---

## 0. Unified console startup screen (Build 24)

Every console tool now prints the same Lumina-Engine startup screen (ASCII logo + build number +
project links):

| File | Type | Description |
|------|------|-------------|
| `common/build-info.cpp.in` / `.h` | modified | adds `lumina_ascii_logo()`; `llama_print_lumina_banner()` prints the full startup screen; version bumped to **Build 24** |
| `common/arg.cpp` | modified | prints the startup screen once `common_params_parse()` succeeds (except `cli`, which draws its own richer banner) |
| `tools/cli/cli-context.cpp` | modified | drops the local logo copy, reuses `lumina_ascii_logo()` |
| `tools/server/server.cpp`, `tools/cvector-generator/cvector-generator.cpp` | modified | removed duplicate banner printing |
| `tools/gguf-split/gguf-split.cpp` | modified | prints the startup screen; `--version` reuses `llama_print_build_info()` |
| `tools/rpc/rpc-server.cpp` / `tools/rpc/CMakeLists.txt` | modified | does not link `llama-common`; prints the same screen from `LUMINA_RPC_BUILD_NUMBER` + the same ASCII art |

---

## 1. `ggml/src/ggml-cpu/q1t.cpp` (new, ~480 lines)

The dedicated Q1_0 inference kernel. Key contents:

### 1.1 Per-128 activation format

```c
#define QK8_128 128
typedef struct {
    ggml_half d;        // one fp16 scale per 128 values
    int8_t    qs[128];
} block_q8_128;
static_assert(sizeof(block_q8_128) == sizeof(ggml_half) + QK8_128, "...");
```

### 1.2 Activation quantization (AVX2-vectorized + block-parallel across threads)

```c
static void q1t_quantize_block(const float * x, block_q8_128 * y) {
    // 4 × 32 values: andnot+max for |amax|, round-to-nearest, packs to int8
    ...
}
```

Quantization is distributed across threads by 128-value block:

```c
for (int64_t blk = ith; blk < total_blocks; blk += nth)
    q1t_quantize_block(...);
```

This lets single-column (tg) work be parallelized across all threads. An earlier row-based split left
only one thread working when `ne11 == 1`, with the rest idling at the barrier, significantly slowing tg
(fixed in an early version).

### 1.3 VNNI selection

```c
#if defined(__AVX512VNNI__)
#  define Q1T_VNNI 1
#  define Q1T_DPBUSD(acc, a, b) _mm256_dpbusd_epi32(acc, a, b)
#elif defined(__AVXVNNI__)
#  define Q1T_VNNI 1
#  define Q1T_DPBUSD(acc, a, b) _mm256_dpbusd_avx_epi32(acc, a, b)
#endif
```

### 1.4 Register-tiled kernel (pp)

```c
#define Q1T_ROWS 2
#define Q1T_COLS 4

template <int R, int T>
Q1T_NOINLINE void q1t_dot(int n, float * out,
                          const block_q1_0 * const * x,
                          const block_q8_128 * const * y) {
    // ... initialize acc[R][T]
    for (int ib = 0; ib < nb; ++ib) {
        __m256 d0v[R];                          // broadcast weight scale per row
        for (int r = 0; r < R; ++r)
            d0v[r] = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[r][ib].d));

        __m256i t32[R][T];                      // int32 accumulators (across 4 sub-blocks)
        // ... zero them

        for (int k = 0; k < 4; ++k) {
            __m256i sg[R];
            for (int r = 0; r < R; ++r)         // expand signs once
                sg[r] = q1t_expand(((const uint32_t *) x[r][ib].qs)[k]);

            for (int t = 0; t < T; ++t) {       // reused for T columns
                const __m256i qy = _mm256_loadu_si256(
                        (const __m256i *) (y[t][ib].qs + k * 32));
                for (int r = 0; r < R; ++r) {
                    const __m256i sy = _mm256_sign_epi8(qy, sg[r]);
#if defined(Q1T_VNNI)
                    t32[r][t] = Q1T_DPBUSD(t32[r][t], ones, sy);       // VNNI
#else
                    t32[r][t] = _mm256_add_epi32(t32[r][t],
                            _mm256_madd_epi16(_mm256_maddubs_epi16(ones, sy), ones16));
#endif
                }
            }
        }

        for (int t = 0; t < T; ++t) {           // one float conversion per 128 weights
            const __m256 d1v = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[t][ib].d));
            for (int r = 0; r < R; ++r) {
                const __m256 dscale = _mm256_mul_ps(d0v[r], d1v);
                acc[r][t] = _mm256_fmadd_ps(dscale, _mm256_cvtepi32_ps(t32[r][t]), acc[r][t]);
            }
        }
    }
    // ... horizontal sum and output
}
```

### 1.5 Single-column GEMV kernel (tg)

```c
#define Q1T_GEMV_ROWS_DEFAULT 1   // tune with GGML_Q1_GEMV_ROWS=1|2|4
#define Q1T_GEMV_ROWS_MAX     4

template <int R>
Q1T_NOINLINE void q1t_gemv(int n, float * out,
                           const block_q1_0 * const * x,
                           const block_q8_128 * y) {
    for (int ib = 0; ib < nb; ++ib) {
        __m256i t32[R] = {0};
        for (int k = 0; k < 4; ++k) {
            const __m256i qy = _mm256_loadu_si256((const __m256i *) (y[ib].qs + k * 32));
            for (int r = 0; r < R; ++r) {
                const __m256i sg = q1t_expand(((const uint32_t *) x[r][ib].qs)[k]);
                t32[r] = Q1T_DPBUSD(t32[r], ones, _mm256_sign_epi8(qy, sg));
            }
        }
        // accumulate the 4 sub-blocks in int32; one float conversion per 128 weights
        const __m256 d1v = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib].d));
        for (int r = 0; r < R; ++r)
            acc[r] = _mm256_fmadd_ps(_mm256_mul_ps(d0v[r], d1v),
                                     _mm256_cvtepi32_ps(t32[r]), acc[r]);
    }
}
```

Condition: `src1->ne[1] == 1` and `GGML_NO_Q1_GEMV` unset.

### 1.6 Dispatch, scratch and compute

```c
bool ggml_cpu_q1t_supports_mul_mat(const struct ggml_tensor * dst);  // Q1_0 + AVX2 + ...

// shared scratch: per-128 activations for all src1 rows
size_t ggml_cpu_q1t_scratch_size(const struct ggml_tensor * dst);

// quantize src1 → barrier → tiled matmul
void ggml_compute_forward_mul_mat_q1t(const struct ggml_compute_params * params,
                                      struct ggml_tensor * dst);
```

`compute` key flow:

```c
// 1. quantize F32 activations to Q8_128 in parallel (128-value blocks per thread)
for (int64_t blk = ith; blk < total_blocks; blk += nth)
    q1t_quantize_block(src + bi*QK8_128, scratch + row*qrow_size + bi);

// 2. barrier
ggml_barrier(params->threadpool);

// 3a. ne11 == 1: single-column GEMV (tg)
// 3b. otherwise: 2-row × 4-column dynamically tiled GEMM (pp)
//     row/column tails fall back to the generic vec_dot (Q8_0)
```

---

## 2. `ggml/src/ggml-cpu/q1t.h` (new)

```c
bool   ggml_cpu_q1t_supports_mul_mat(const struct ggml_tensor * dst);
size_t ggml_cpu_q1t_scratch_size   (const struct ggml_tensor * dst);
void   ggml_compute_forward_mul_mat_q1t(const struct ggml_compute_params * params,
                                        struct ggml_tensor * dst);
```

---

## 3. `ggml/src/ggml-cpu/ggml-cpu.c` (modified)

### 3.1 include

```c
#include "traits.h"
#include "iqp.h"
#include "q1t.h"      // ← new
```

### 3.2 mul_mat hook (after the src1→Q8_0 conversion + barrier)

```c
// IQ panel gemm (iqp.h)
if (ggml_cpu_iqp_supports_mul_mat(dst) && !params->use_ref) {
    ggml_compute_forward_mul_mat_iqp(params, dst);
    return;
}

// 1-bit Q1_0 register-tiled gemm (q1t.h)   ← new
if (ggml_cpu_q1t_supports_mul_mat(dst) && !params->use_ref) {
    ggml_compute_forward_mul_mat_q1t(params, dst);
    return;
}
```

### 3.3 work buffer size

```c
// the IQ panel path needs one scratch panel per thread past the q8_K rows
if (ggml_cpu_iqp_supports_mul_mat(node)) {
    cur = GGML_PAD(cur, 64) + n_tasks * ggml_cpu_iqp_scratch_size(node);
}

// the Q1_0 register-tiled path needs the q8_128 activation past the q8_0 rows  ← new
if (ggml_cpu_q1t_supports_mul_mat(node)) {
    cur = GGML_PAD(cur, 64) + ggml_cpu_q1t_scratch_size(node);
}
```

---

## 4. `ggml/src/ggml-cpu/CMakeLists.txt` (modified)

```cmake
ggml-cpu/iqp.cpp
ggml-cpu/iqp.h
ggml-cpu/q1t.cpp      # ← new
ggml-cpu/q1t.h        # ← new
```

---

## 5. `ggml/src/ggml-cpu/arch/x86/quants.c` (modified)

### 5.1 VNNI macros (file scope)

```c
#if defined(__AVX512VNNI__)
#  define GGML_Q1_0_VNNI 1
#  define GGML_Q1_0_DPBUSD(acc, a, b) _mm256_dpbusd_epi32(acc, a, b)
#elif defined(__AVXVNNI__)
#  define GGML_Q1_0_VNNI 1
#  define GGML_Q1_0_DPBUSD(acc, a, b) _mm256_dpbusd_avx_epi32(acc, a, b)
#endif
```

### 5.2 AVX2 branch of `ggml_vec_dot_q1_0_q8_0`

**Before:**

```c
const __m256i sm = _mm256_cmpeq_epi8(
        _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) qs32[K]), byte_shuf), bit_masks), zero);
const __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
```

**After:**

```c
// 0xFF where the bit is 0, 0x01 where it is 1; vpsignb then negates the corresponding lanes
const __m256i sg = _mm256_or_si256(_mm256_cmpeq_epi8(
        _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) qs32[K]), byte_shuf), bit_masks), zero), ones_8);
const __m256i sy = _mm256_sign_epi8(qy, sg);
#if defined(GGML_Q1_0_VNNI)
const __m256i s32 = GGML_Q1_0_DPBUSD(_mm256_setzero_si256(), ones_8, sy);
#else
const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
#endif
```

The same change is applied to the first sub-block and the `K=1..3` loop (2 places in total).

---

## 6. Binary lookup table inside `q1t.cpp` (Q1LUT, enabled by default)

Added inside the same file (no new file):

- `q1lut_get_or_alloc` / `q1lut_fill_group`: lazily permute Q1_0 weights into lookup codes and collect
  the per-128 `d0` scale (copying the original fp16 directly, so tg weight traffic equals the packed
  format); cached per tensor in-process, filled in parallel by row group.
- `q1lut_build_group`: build the 16-entry `int16` table (low bytes + high bytes) from activations.
  Using true binary symmetry `T[c ^ 0xf] = -T[c]`, only `c = 0..7` are summed and the rest filled by
  negation, halving generation work.
- `q1lut_acc128` / `q1lut_gemv32`: AVX2 kernel processing 32 output rows at a time via `vpshufb` (full
  table direct-write, the default).
- `q1lut_gemv32_add`: experimental K-blocked accumulation kernel enabled by `GGML_Q1_LUT_KB`
  (default 0 = full table); measured slower.
- Batched pp path (enabled by default, disabled with `GGML_Q1_LUT_PP=0`): builds tables for 32-column
  blocks; `q1lut_gemv32` computes one column at a time.
- `ggml_cpu_q1t_scratch_size` / `ggml_compute_forward_mul_mat_q1t`: `ne11 == 1` takes the tg LUT path;
  `ne11 >= Q1LUT_PP_MIN(8)` takes the pp LUT (both default-on, controlled by `GGML_Q1_LUT` /
  `GGML_Q1_LUT_PP`).
- `q1t_env_bool`: boolean environment parsing (`0/false/off/no` are false) for the switches above and
  `GGML_Q1_LUT_DEBUG`.

Controlled by `GGML_Q1_LUT` (master) and `GGML_Q1_LUT_PP` (pp only); when disabled there is zero
impact on upstream and other paths. Test tool: `bench/bench_lut2.cpp`.

---

## Build / delivery scripts

Besides the kernel changes, the repo root and `bench/` gained/updated build and delivery scripts (no
inference code involved; see [Delivery](Delivery.en.md)):

| Script | Purpose |
|------|------|
| `build-bin.bat` | one-click: MSVC + Clang build + assemble into `Lumina-Engine-Bin` |
| `bench/build_all_msvc.bat` / `bench/build_all_clang.bat` | build the two build trees |
| `bench/assemble_bin.bat` | clang base + 9 MSVC overrides + runtimes → `Lumina-Engine-Bin` |
| `bench/build_haswell.bat` / `bench/build_x64.bat` | single-variant incremental builds (dev) |
| `bench/bench_lut2.cpp` | standalone Q1LUT kernel micro-benchmark (with correctness checks) |

**Convention**: `Lumina-Engine-Bin` is the only delivery directory; the old `C:\Lumina-Engine\llama` entry has
been removed.

---

## 7. Lossless model compression container (Build 6/7/8, new)

Shrinks the model file **without quantization and without pruning**. New/changed files:

| File | Type | Notes |
|------|------|-------|
| `src/lumina-gguf.cpp` / `.h` | new | detects the `LGGC` container, streams it to a temporary `.gguf`; byte-exact restore |
| `src/miniz/miniz.{c,h}` | new | embedded single-file DEFLATE (MIT / public domain) |
| `src/llama.cpp` | modified | `llama_model_load_from_file()` calls `lumina_gguf_resolve_path()` (plain GGUF unchanged) |
| `src/CMakeLists.txt` | modified | builds `lumina-gguf.cpp` and `miniz.c` |
| `tools/lumina_gguf_pack.py` | new | packer: k-quant field split + context sort + DEFLATE (container v2) |
| `tools/lumina_gguf_unpack.py` | new | reference unpacker / `verify` byte-exact check |
| `common/build-info.cpp.in` | modified | version bumped to **Build 8** |
| `common/preset.cpp` | modified | model-dir listing recognises `.gguf.lumina` (strips suffix for the name) |
| `common/arg.cpp` | modified | `-m` shell completion matches both `.gguf` and `.gguf.lumina` |

**Build 7 (container v2)**: on top of the Build 6 field split, Q4_K `d/dmin` are split into fp16
byte-planes, the six-bit `scale`/`min` fields are unpacked to byte streams, and the 4-bit quants are
stably sorted by the context key `(scale << 6 | min)`; Q6_K `d` is byte-plane split and `scales`
are split even/odd.  The decoder reads both v1 and v2.

**Build 8 (extension convention)**: the packer now writes `A.gguf` to `A.gguf.lumina` by default
(the original is kept), so the original model and the compressed container are easy to tell apart.
The engine keys off the magic, so both `.gguf` and `.gguf.lumina` load directly; the model-directory
listing and the CLI completion also treat `.gguf.lumina` as a model.

Measured on this model: `2,630,213,408 B -> 2,479,779,960 B` (**94.28%**, 143.5 MiB saved;
container v1 was 95.30% / 117.9 MiB); the restored SHA-256 matches the original.  See
[LosslessCompression](LosslessCompression.en.md).

---

## 8. Training engine and dataset support (Build 9/10/11)

A pure C++ training tool, `tools/lumina-train/` (delivered as `Lumina-Engine-train`), trains LoRA
adapters (QLoRA) on top of a frozen quantized base. Build 10 added the ggml backward ops needed
for MoE training (`MUL_MAT_ID`, `REGLU`/`SIGMOID`/`TANH`/`CLAMP`/broadcast `DIV`, batched
`GET_ROWS_BACK`); Build 11 adds dataset and checkpoint support.

| File | Type | Description |
|------|------|-------------|
| `tools/lumina-train/lumina-train.cpp` | new/modified | QLoRA trainer; Build 11 adds `.jsonl` datasets (`text`/`messages`/`conversations`), a separate validation set, checkpoints and `--cpu-cores` |
| `include/llama.h` / `src/llama-adapter.cpp` | modified | `llama_adapter_lora_init_trainable()` / `_save()` / `_copy_weights()` (Build 11) |
| `src/llama-context.cpp` | modified | `opt_init()` registers LoRA params, doubled graph budget; Build 11 negative-label loss masking |
| `common/common.h` / `common/arg.cpp` | modified | `-lora-*` options; Build 11 `--dataset*` / `--mask-history` / `--cpu-cores` / `--save-steps` / `--keep-last` / `--resume` |
| `ggml/include/ggml.h` / `ggml-rpc.h` / `ggml.c` / backends | modified | MoE backward ops (Build 10) |
| `common/build-info.cpp.in` / `CMakeLists.txt` | modified | version bumped to **Build 11** |

---

## 9. Vulkan backend and default device policy (Build 12)

| File | Type | Description |
|------|------|-------------|
| `configure-release.bat` | modified | detect `D:\VulkanSDK\sdk` and pass `-DGGML_VULKAN=ON`, otherwise fall back to CPU-only |
| `common/common.h` | modified | `n_gpu_layers` default `-1` (auto) → **`0`** (layers on CPU + Vulkan op-offload) |
| `common/arg.cpp` | modified | drop the old `GGML_ASSERT(n_gpu_layers < 0)` guard and fix the `-ngl` help text |
| delivery | modified | adds `ggml-vulkan.dll` (55 files) |

Measured on this machine (Intel Iris Xe): with the default `-ngl 0`, Qwen3-0.6B pp512 is **≈2.4×** and
phi-3-mini pp256 **≈1.6×** versus pure CPU, with unchanged tg; `-ngl auto/99` (full layer offload)
raises pp further but slows tg; perplexity differs by **+0.11%** (CPU vs op-offload).
See `OPTIMIZATION_LOG.md` for the full A/B data.

**Build 13**: `tools/lumina-train/` now forces `n_gpu_layers = 0` and `no_op_offload = true`
before training — otherwise Vulkan op-offload corrupts the training accuracy metric (acc ≈ 8.25e8%);
after the fix it returns to normal (~10%).

**Build 14**: `common/common.cpp` adds a machine- and model-aware automatic device policy
(no GPU / integrated GPU → layers on CPU; discrete GPU → offload layers; MoE model → op-offload
disabled). One prebuilt binary decides at runtime. MoE with op-offload is ~30% slower for pp
(143 vs 207 t/s), so it is disabled automatically.

**Build 15**: GPU training experiment concluded — not feasible (the Vulkan backend has no backward
kernels). `LUMINA_TRAIN_ALLOW_GPU` is kept as an experimental, off-by-default escape hatch;
training defaults to pure CPU.

**Build 16**: fixed the broken training `acc` metric on the GPU path — the Vulkan `count_equal`
kernel writes 32-bit values while `ggml_count_equal` produces I64; Vulkan is now restricted to I32
destinations and I64 falls back to the CPU. The build number and project links are now shown by
`--version`, the CLI banner and the server startup. (Note: the standard Vulkan backward ops were
already implemented; dense GPU training loss matches the CPU.)

---

## Change size

| File | Type | Lines changed (approx.) |
|------|------|---------------|
| `q1t.cpp` | new | +480 (base) + ~190 (Q1LUT) |
| `q1t.h` | new | +30 |
| `ggml-cpu.c` | modified | +10 |
| `CMakeLists.txt` | modified | +2 |
| `arch/x86/quants.c` | modified | +20 |
| `bench/bench_lut2.cpp` | new | standalone micro-benchmark |
| `lumina-gguf.cpp` / `.h` | new | ~330 (lossless container decoder, v1 + v2) |
| `miniz/miniz.{c,h}` | new | embedded DEFLATE (vendored) |
| `tools/lumina_gguf_pack.py` / `_unpack.py` | new | pack / verify scripts |

**Total < 620 lines**, almost all concentrated in new files, with minimal intrusion into upstream.

---

## Rejected exploratory changes

The following were evaluated and **not kept**; nothing remains in the repo:

| Approach | Conclusion |
|------|------|
| Memory panel (IQP style) | Extra 1 byte/weight read+write cancels the gain; measured slower |
| AVX-512 512-bit tile | Layout mismatch + d1 vector construction overhead; theoretical gain ~1.1×, and unverifiable locally |
| Multi-row GEMV + Q8_0 activations (tg) | Activation scale still per-32, float conversions unchanged, no measured gain; tg moved to the Q8_128 single-column GEMV |
| New GGML type Q8_128 | Would require touching 175+ backend files, too risky; used a q1t-local format instead |
| Software prefetch | No benefit |
| 2-block unroll | Unstable gain, higher register pressure |
