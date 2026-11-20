## 0. PrismML `prism.hadamard.*` weight rotation (Build 33)

PrismML's ternary models (e.g. `Ternary-Bonsai-2-2B-Q2_0.gguf`, `qwen35` architecture + `Q2_0` ternary
quantization) ship a `prism.hadamard.*` metadata block: **every quantized weight is stored rotated**,

```text
W' = W * R,    R = block(512) normalized Sylvester-Walsh-Hadamard  o  diag(explicit +/-1 signs)
```

so inference must rotate the activation back into the same basis: `y = W x = W' * (R^T x)`. Before Build 33
the engine ignored those keys and used `W'` as if it were `W` -- the model loaded, decoded and even ran
`perplexity` to completion, but the output was gibberish (`Hello` -> `věd三等ieri fitting vastuмот')`).

Build 33 reads the metadata and inserts the rotation exactly where it belongs:

| Position | Transform |
|---|---|
| input of every rotated quantized weight (`weight_names`, 150 here) | forward `H(diag(s) x)`, block 512 |
| `token_embd.weight` lookup output (`inverse_weight_names`) | inverse `diag(s) H(x)`; the **raw (vision) embedding branch is left untouched** |
| tied output head (`tied_output = true`) | forward `H(diag(s) x)` |

| File | Type | Change |
|------|------|--------|
| `src/llama-hparams.h` | modified | new `struct llama_prism_hadamard` (`enabled` / `tied_output` / `block_size` / `signs[width]` / `fwd` / `inv`); **kept out of `llama_hparams`** because that struct must stay trivially copyable |
| `src/llama-model.h` / `.cpp` | modified | `llama_model::prism` + `llama_load_prism_hadamard()`: parses and validates the metadata with the raw gguf API; anything unexpected only warns and disables the feature |
| `src/llama-context.cpp` | modified | `graph_params()` passes `&model.prism` |
| `src/llama-graph.h` / `.cpp` | modified | `llm_graph_input_prism` (constant rotation matrix + sign vectors), `build_inp_prism()` / `build_prism_rotate()` / `build_prism_embd()` / `build_prism_output()` / `has_prism_rot()`; hooked into `build_lora_mm()` and the token branch of `build_inp_embd()` |
| `src/models/qwen35.cpp` / `qwen35moe.cpp` | modified | `build_inp_prism()` at graph construction; `build_prism_output()` before the tied head (incl. the MTP draft head) |
| `CMakeLists.txt` / `tools/rpc/rpc-server.cpp` | modified | Build 32 -> 33 |

The rotation reuses the engine's existing FWHT fast path (`GGML_HINT_SRC0_IS_HADAMARD` ->
`ggml_compute_forward_fwht`): `O(n log n)` on CPU / Vulkan / CUDA / Metal / SYCL; backends without the hint
fall back to a real matrix product (same result, slower). The rotation tensors are constant graph inputs
copied once on the first `set_input`. **Models without the metadata are completely unaffected.**

Implementation caveats (both were hit for real):

1. **Graph input tensors must be re-filled on every ubatch.** The first version treated the rotation
   matrix / sign vectors as constants and copied them only once per graph object - result: *prefill
   correct, token-by-token generation broken from the first step* (the `ggml_backend_sched` allocator
   recycles an input tensor's memory as soon as its last consumer has been computed). Symptoms:
   `-ub 512` gets the first token right and then garbage; `perplexity -b 1` returns `nan`; batched
   perplexity looks perfect (19.11 vs 1.8e6 for the wrong configuration). Fix: `set_input()` memcpy's
   unconditionally (1 MB per step, negligible).
2. **The tied output head has to be detected by tensor name** (`name in inverse_weight_names`), not by
   comparing `model.output == model.tok_embd`. With `-ngl -1` the `TENSOR_DUPLICATED` output tensor is a
   *different* `ggml_tensor` instance (symptom: CPU PPL 19.11 but Vulkan PPL 1,236,995).

Measured (`wiki/Background.en.md`, `-c 512 --chunks 2`): this implementation gives **PPL 19.11**
(identical on CPU and `-ngl -1` Vulkan); token-by-token (`-b 1 -ub 1`) gives **13.11**; swapping the
sign/Hadamard order gives **1,801,563**; a wrong embedding/head direction gives 29,055 - 1,236,995.
Generation is **identical for `-ub 1 / 2 / 4 / 512`** (`...France is` -> ` Paris.`; CLI `Hello` -> a
coherent reasoning paragraph), and the mmproj turns a 768px image into 256 vision tokens and describes it
correctly. Debug switches: `LUMINA_PRISM_DISABLE` / `LUMINA_PRISM_ORDER` / `LUMINA_PRISM_EMBD` /
`LUMINA_PRISM_OUT` / `LUMINA_PRISM_NO_FWHT` / `LUMINA_PRISM_DEBUG` / `LUMINA_NO_FUSED_GDN`.
Details in `HANDOFF.md` § Build 33.

---

## 0.1 Training progress bar loses `tok/s` (Build 32)

Build 27 appended a live throughput field `%5.1f tok/s` to the training progress line. It was a
**cumulative average over the whole pass**: the first micro-batches spend their time on disk reads and
cache building, permanently dragging the average down; the numerator also counted the tokens of
**fully masked prompt micro-batches** (`loss=0`, one loss row per ubatch); and with background load the
number drifted from 14.4 to 26.4 *within one run*. Build 32 removes it — the progress line keeps only
`loss` / `acc` / elapsed `t=` / `ETA=`:

```text
train: [███████▉] data=0000008/0000008 loss=1.94035±0.22760 acc=58.20±2.18%   t=00:01:05 ETA=00:00:00
```

To compare throughput, use a fixed token budget plus an external stopwatch (or `--dataset-limit` for a
short run of equal size); see the throughput table in `TRAINING_PIPELINE.md`.

The change touches only the `fprintf` in `ggml/src/ggml-opt.cpp` (no arithmetic), so **training is
bit-identical to Build 30/31**: for the same small-sample command all three builds produce the adapter
SHA-256 `04690bf8…69bd72` (details in `HANDOFF.md` § Build 32).

---

# Main Code Changes

> 🌐 [中文](Changes.md) | **English**

This page lists all changes relative to upstream llama.cpp (b10991). The Q1LUT (binary lookup
table) work touches **5 files**: 2 new, 3 modified (all inside the new `q1t.cpp`), and affects
only the `GGML_TYPE_Q1_0` + AVX2 path; on top of that there is the Q4_K decode-kernel prefetch
(Build 31, see §0) plus the per-build CLI/training fixes.

---

## 0.2 Q4_K GEMV software prefetch (Build 31)

CPU single-token generation spends almost all of its time in `ggml_gemv_q4_K_8x8_q8_K` (the repacked
Q4_K decode kernel). On this machine that kernel only reaches 6.0-7.5 GB/s per thread while a plain
streaming read on the same thread tops out at 11.6-13.1 GB/s - so it is **memory-latency bound**, not
bandwidth bound. Fix: at the top of the block loop, pull the whole block that will be consumed two
iterations later (1152 B = 18 cache lines) into L1 with `_mm_prefetch`.
**Do not add an `if (b + 2 < nb)` guard**: the branch inside the hot loop costs more than the few
useless prefetches at the end (measured -13%), and `PREFETCHh` never faults on an invalid address.

| File | Type | Description |
|------|------|-------------|
| `ggml/src/ggml-cpu/arch/x86/repack.cpp` | modified | `ggml_gemv_q4_K_8x8_q8_K()`: whole-block prefetch at +2 blocks (`sizeof` driven, unrolled) |
| `CMakeLists.txt` / `tools/rpc/rpc-server.cpp` | modified | Build 30 -> 31 |
| `tests/bench-cpu-kern.cpp` | new | kernel-level A/B benchmark: real kernel entry point + independent scalar reference + interleaved in-process measurement (built only with `LLAMA_BUILD_TESTS=ON`, not shipped) |
| `tests/q4k_gemv_body.inc` / `tests/CMakeLists.txt` | new / modified | baseline copy of the kernel body extracted from `repack.cpp`, plus the benchmark target |
| `bench/gen_q4k_body.py` / `bench/build_bench_cpu_kern.bat` | new | regenerate the baseline kernel body / build the separate `build-bench` tree in one step |

Measured on this machine (i5-10400 / 8 GB, a training job running in the background; every number is a
median of interleaved in-process runs):

| Metric | Before | After |
|---|---:|---:|
| kernel bandwidth, single thread (18 MiB matrix, DRAM resident) | 6.00-6.47 GB/s | 6.63-6.78 GB/s (**+13.0%**) |
| end-to-end decode `tg32` (Qwen3.5-0.8B Q4_K_M, `-t 4`, Clang) | 13.66 t/s | 15.53 t/s (**+13.7%**) |
| end-to-end decode `tg32` (MSVC delivery variant, paired median) | - | **+13%** |
| prefill `pp128` | 79-89 t/s | 87-89 t/s (unchanged, the GEMM path is untouched) |
| L3 resident small matrix (0.56 MiB) | 10.07 GB/s | 10.30 GB/s (within noise) |

Correctness: the patched kernel produces **bit-identical** output to the previous one (`memcmp`);
greedy decoding (`--temp 0`) of the real engine yields exactly the same text as Build 30; all 7
experimental variants match an independent scalar reference (`max_abs_err = 4.8e-07`, tolerance 2.9e-06).

## 0.3 CLI multi-turn fix (Build 30)

MoE chat answered `None` from the 2nd turn on: the CLI unconditionally appended the assistant message
(possibly an empty string) to its history, so a turn that produced no visible output left a dangling
empty assistant message there. Every later request then looked like `[user, assistant(""), user]` and
strict MoE chat templates made the model emit a placeholder token for good. Also fixed the infinite
`> ` loop after stdin was closed.

| File | Type | Change |
|------|------|--------|
| `tools/cli/cli-context.cpp` | modified | empty / failed turns are no longer stored in the history (`[empty response: ...]` is printed, `--output` gets no empty section); an empty read is treated as EOF and exits; empty model selection returns failure; `a.finish()` is called at the end of every turn |
| `tools/cli/cli-ui.h` | modified | `ui::assistant_turn::finish()` closes `[End thinking]` and resets colour / trailing-newline state at the end of a turn |
| `CMakeLists.txt` / `tools/rpc/rpc-server.cpp` | modified | Build 29 -> 30 |

Measured: `printf '你好\n1+1=?\n' | cli` produced an endless `> ` stream on Build 29
(1,483,183 lines in 300 s) and exits normally with `Exiting...` (37 lines, exit 0) on Build 30.

## 0.4 `--mask-history` training-path issue + normalisation (Build 29)

Measured: same token stream, `-c 4096`, per-batch loss ~11.9 with `--mask-history` vs ~1.47 with
`--no-mask-history` (two runs each, bit-identical; labels verified correct; inference PPL of the same
text is 2.03). **Train with `--no-mask-history`.**

| File | Type | Change |
|------|------|--------|
| `ggml/src/ggml-opt.cpp` / `ggml/include/ggml-opt.h` | modified | new `ggml_opt_result_add_rows()`; loss/accuracy normalise by the evaluated row count `nrows` instead of `ndata` (dataset points) |
| `src/llama-context.cpp` | modified | report `n_ubatch` rows per evaluated ubatch |
| `tools/lumina-train/lumina-train.cpp` | modified | note about the `--mask-history` path and the recommended configuration |
| `CMakeLists.txt` / `tools/rpc/rpc-server.cpp` | modified | Build 28 -> 29 |

## 0.5 Training-loss reporting note (Build 28)

The progress-bar loss is a cumulative mean; the CE kernel averages over all rows while
`ggml_opt_result_loss()` compensates with `ndata/nlabeled` (dataset points vs labelled rows - different
units), so right after a long masked prompt the value is inflated ~8x (same tokens, same positions:
masked 11.88 vs unmasked 1.44). Gradients are unaffected. A note is printed when `--mask-history` is on;
judge progress with `--dataset-val`.

| File | Type | Change |
|------|------|--------|
| `tools/lumina-train/lumina-train.cpp` | modified | removed the incorrect `-c > 2048` warning; added the `--mask-history` note |
| `CMakeLists.txt` / `tools/rpc/rpc-server.cpp` | modified | Build 27 -> 28 |

## 0.6 Disk-backed training dataset (Build 27)

The trainer used to accumulate the entire dataset in two `std::vector`s
(`int32` token ids + `uint8` trainability mask = **5 bytes per token**) and only then serialise it
to a temp file for mmap. A 1.18 B token corpus therefore needed ~**5.9 GiB** of extra RAM before
training even started.

| File | Type | Change |
|------|------|--------|
| `tools/lumina-train/lumina-train.cpp` | modified | disk sink on `lumina_token_stream` (`FILE*` + `sink_n` / `sink_trainable` / `sink_dir`); `lumina_append()` writes ids and mask with `fwrite`; new `lumina_stream_begin/finish/discard()` and `lumina_append_padding()`; `lumina_stream_view` gained `map_file()` + `map_streamed()`, and `size()` / `n_trainable()` cover both modes |
| same | modified | temp files are per process: `%TEMP%\lumina-train\<tag>-<pid>\` (Build 26 let two trainers share `train.stream.bin`) |
| `ggml/src/ggml-opt.cpp` | modified | progress bar reports `tok/s`; `ggml_opt_result_loss()` returns `NAN` when the result holds no batch yet (prints `loss=n/a` instead of 0.0). A `loss=0.00000` on the first validation micro-batch has a different cause (that micro-batch only contains masked prompt tokens); Build 26 behaves the same |
| build scripts (10 `.bat`) | modified | hardcoded `C:\Lumina-Engine` replaced with `%~dp0`; `build-clang.bat` takes `--target` |
| `CMakeLists.txt` / `tools/rpc/rpc-server.cpp` | modified | Build 26 -> 27 |

The on-disk layout is unchanged (`[n x int32][pad to 64 B][n x uint8]`), so the trainer reads
byte-identical data:

| Version | peak RSS while loading 300 M tokens | dataset stats |
|---|---|---|
| Build 26 | 2074 MB | 29282 rec / 300,021,789 tok / 278,340 datapoints |
| Build 27 | **645 MB** | identical |

## 0.7 Training gradient-accumulator fix (Build 26)

For the `llama-finetune` / `llama_opt_epoch()` path (`Lumina-Engine-train`) with
`n_batch / n_ubatch > 1`, the old code never cleared the gradient accumulators: every optimizer
step applied the **sum of all gradients since the start of training**, so the update magnitude
grew without bound and the loss climbed (approximately linearly).

| File | Type | Description |
|------|------|-------------|
| `ggml/src/ggml-opt.cpp` | modified | adds `ggml_opt_zero_grad_accs()` + `grad_acc_is_loss`: clears the persistent accumulators directly (loss set to 1) instead of relying on `gb_grad`, which is `nullptr` in the non-static-graph path; called from `ggml_opt_alloc()` when `backward && opt_period > 1 && opt_i == 0` |
| same | modified | `ggml_opt_eval()` only advances `opt_i` for graphs that compute gradients, so forward-only evaluations (validation / KV-cache rebuild on `--resume`) no longer shift the accumulation phase |
| same | modified | `ggml_opt_reset(opt_ctx, false)` uses `ggml_opt_zero_grad_accs()` |
| `tests/test-opt.cpp` | modified | adds the `test_gradient_accumulation_nonstatic` regression test (non-static graphs + `opt_period > 1`) |
| `CMakeLists.txt` / `tools/rpc/rpc-server.cpp` | modified | Build 25 → 26 |

> LoRA checkpoints/adapters produced before Build 26 were trained with contaminated gradients;
> discard them and retrain.

---

## 0.8 Unified console startup screen (Build 24)

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

**Convention**: `Lumina-Engine-Bin` is the only delivery directory; the old `D:\Lumina-Engine\llama` entry has
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
