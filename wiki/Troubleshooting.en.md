# FAQ & Troubleshooting

> 🌐 [中文](Troubleshooting.md) | **English**

## Performance

### Q: Why didn't tg (generation) get faster?

First confirm the binary lookup table (Q1LUT) is active: it is **enabled by default** and gives about
**+30%** in controlled A/B (~9.8 → ~13.0 t/s). If it is not faster, check whether `GGML_Q1_LUT=0` is
set, whether the CPU has AVX2, and whether the DLL is current.

> ⚠️ This machine uses **single-channel memory** (8 GB DDR4-2666; measured streaming read bandwidth
> ~14–17 GB/s). tg is ultimately **memory-bandwidth bound**: each token must read ~1.06 GB of Q1_0
> weights, so the theoretical ceiling is ~15.7 tok/s; Q1LUT already pushes the kernel close to that
> ceiling. Dual-channel / higher-bandwidth platforms will show higher tg.

This matches the model vendor's whitepaper: 1-bit mainly reduces tg's bandwidth demand rather than
raising CPU arithmetic throughput.

**Ways to improve tg further**: dual-channel memory, no background load, more physical cores (but no
more than the physical core count).

### Q: Why is pp512 only 28% faster while pp32 is 42%?

The tiled kernel's gain comes from “reusing the sign expansion across 4 columns”. With fewer columns
there are fewer absolute reuses, but the generic path is inherently slower at those small batches
(the fixed overhead of 16×16 tiling dominates), so the **relative** gain is larger. Absolute
throughput still grows with batch size.

### Q: How do I confirm the tiled path is really enabled?

```bat
set GGML_NO_Q1_TILE=1
Lumina-Engine-bench.exe -m Bonsai-8B.gguf -p 128 -n 1 -t 6 -r 2
set GGML_NO_Q1_TILE=
Lumina-Engine-bench.exe -m Bonsai-8B.gguf -p 128 -n 1 -t 6 -r 2
```

If the two differ noticeably (should be ~25%), the tiled path is working.

### Q: How do I confirm the tg GEMV path is enabled?

```bat
set GGML_NO_Q1_GEMV=1
Lumina-Engine-bench.exe -m Bonsai-8B.gguf -p 64 -n 128 -t 12 -r 3
set GGML_NO_Q1_GEMV=
Lumina-Engine-bench.exe -m Bonsai-8B.gguf -p 64 -n 128 -t 12 -r 3
```

At 12 threads the new version should be ~13% faster; at 6 threads the single-channel bandwidth limit
makes the difference small. The row group size can be tuned with `GGML_Q1_GEMV_ROWS` (1/2/4, default 1).

### Q: Results fluctuate a lot?

This is an office machine, so performance drifts with temperature/load. Measure **same-run
alternating**, and increase `-r`:
```bat
Lumina-Engine-bench.exe -m Bonsai-8B.gguf -p 128 -n 1 -t 6 -r 5
```

---

## Correctness

### Q: Greedy decoding occasionally differs from the optimized-off run?

The tiled kernel sums floats in a different order (four sub-blocks accumulated in int32 first, then a
single scale multiply), producing ~1e-6 differences that can cause greedy divergence on long texts.
However:

- **No perplexity loss** (5.7111 vs. 5.7160).
- Both continuations are coherent and of comparable quality.
- Upstream repack/GEMM optimization paths show the same phenomenon.

If you need bit-for-bit identity, set `GGML_NO_Q1_TILE=1`.

### Q: `iq1_s` / `iq1_m` CPU tests fail?

This is a **pre-existing upstream phenomenon** (IQP panel-path CPU-vs-CPU tolerance), unrelated to
these changes:
```bat
set GGML_NO_IQ_PANEL=1
test-backend-ops.exe test -b CPU -o MUL_MAT
```
Everything should pass afterwards.

---

## Building

### Q: MSVC only built 9 CPU variants?

Expected. `ivybridge/piledriver/cooperlake/zen4/sapphirerapids` require Clang (see
[Compatibility](Compatibility.en.md#2-why-msvc-can-only-build-9-variants)). Build `build-clang` with
Clang 20 to complete the set.

### Q: Clang reports `error STL1000: Unexpected compiler version, expected Clang 20 or newer`?

VS2026's MSVC STL requires Clang ≥ 20. Install LLVM 20+ (this repo uses 20.1.8).

### Q: `GGML_CPU_ALL_VARIANTS requires GGML_BACKEND_DL`?

Enable all three together:
`-DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON`.

### Q: Link error `undefined symbol g_q1p_*`?

That is leftover from an earlier reverted experiment. Make sure there is **no** `q1p.cpp/h` under
`ggml/src/ggml-cpu/` and no `q1p` reference in `ggml-cpu.c`.

### Q: How do I quickly rebuild after a change?

Rebuild only the variant this machine uses:
```bat
cmake --build build-release --target ggml-cpu-haswell -j
```

---

## Running

### Q: `unknown model architecture` or load failure?

Make sure the model is Q1_0 (the `ftype` column of `Lumina-Engine-bench` should show `Q1_0`). This
project only optimizes Q1_0; other models are handled by upstream code.

### Q: Memory usage?

Bonsai-8B weights are ~1.07 GiB; with the KV cache and runtime, 8 GB is enough. The Q8_128 scratch at
pp512 is ~6.4 MB (negligible). The binary LUT (enabled by default) needs ~0.95 GiB extra;
`GGML_Q1_LUT=0` releases it.

### Q: Can I use a GPU?

This project's optimizations target CPU only. Upstream CUDA/Metal backends are unaffected; `-ngl`
still works.

### Q: How do I switch CPU variants?

Selection is automatic at runtime. To force one, keep only the desired `ggml-cpu-*.dll`; or configure
CMake without `GGML_CPU_ALL_VARIANTS` and compile for the host.

---

## Chat / CLI

### Q: An MoE model answers `None` from the second turn on?

Fixed in Build 30. Root cause: the CLI appended every assistant reply to its history without checking
whether it was empty. A turn without visible output (first token is EOS, interrupted stream, empty
`content`, failed request) left a dangling empty assistant message in the history, so every later
request looked like `[user, assistant(""), user]` and strict MoE chat templates made the model emit a
placeholder token for good. Since Build 30, empty / failed turns are not added to the history
(`[empty response: nothing was added to the chat history]`). On an older build, `/clear` recovers.

### Q: `echo "..." | Lumina-Engine-cli.exe` (or a closed stdin) never exits and keeps printing `> `?

Fixed in Build 30 (an empty read is EOF and exits, matching upstream llama.cpp #29722). Older builds
spin forever and need Ctrl+C / kill.

### Q: The reasoning trace never gets its closing `[End thinking]`?

Since Build 30 every turn closes it (including interrupted streams and errors); older builds only
closed it on a reasoning -> content transition.

---

## Sampling

Recommended parameters (matching the model vendor):

```
--temp 0.5        (0.5 ~ 0.7)
--top-k 20        (20 ~ 40)
--top-p 0.9       (0.85 ~ 0.95)
--repeat-penalty 1.0
```

Example:
```bat
Lumina-Engine-cli.exe -m Bonsai-8B.gguf -p "Write a haiku about autumn." ^
  -n 64 --temp 0.7 --top-k 20 --top-p 0.9 --seed 5 -t 6 -st
```

---

## Diagnostic checklist

| Symptom | Checks |
|------|------|
| pp not faster | is `GGML_NO_Q1_TILE`/`GGML_Q1_LUT_PP` set; does the CPU have AVX2 (x64/sse42/sandybridge have no tiling/LUT) |
| tg not faster | is `GGML_Q1_LUT`/`GGML_NO_Q1_GEMV`/`GGML_NO_Q1_TILE` set; does the CPU have AVX2; is the DLL current |
| tg very slow | bandwidth bound; check whether memory is dual-channel; tune with `GGML_Q1_GEMV_ROWS` |
| garbled output | is the model Q1_0; does the CPU variant match; try the `haswell` variant |
| crash / load failure | are the runtime libraries present (`libomp140/vcomp140/msvcp140/vcruntime140*`) |
| unstable results | measure same-run alternating and increase `-r` |

---

## Binary lookup table (Q1LUT)

### Q: The first token got slower?

The LUT is **enabled by default**: on first use of a weight tensor it is permuted into lookup codes
(~0.7 s for the whole model) and ~0.95 GiB extra memory is allocated. This is a **one-time** cost;
every token afterwards uses the cache, so long generations amortize it. If you do not accept the
memory/first-use latency, set `GGML_Q1_LUT=0` to disable the whole LUT path.

### Q: Does the (default-on) LUT actually speed things up?

Yes. Controlled alternating A/B (`-p 64 -n 128 -t 6`): packed GEMV 9.70/9.99 → LUT 13.04/13.01 t/s
(about **+30%**, at the single-channel bandwidth ceiling); for pp, from `ne11>=8` it is uniformly
faster than the 2×4 tiled GEMM (pp512 +26%, pp2048 +57%). K blocking (`GGML_Q1_LUT_KB`) brings no
benefit; keep the default 0. See
[Optimizations](Optimizations.en.md#optimization-7-binary-lookup-table-q1lut-enabled-by-default).

### Q: How do I confirm the LUT is really active?

```bat
set GGML_Q1_LUT_DEBUG=1
Lumina-Engine-bench.exe -m Bonsai-8B.gguf -p 64 -n 128 -t 6 -r 2
```
It prints `q1lut alloc` lines; no output means it is not active (e.g. no AVX2, `GGML_Q1_LUT=0`, or
`ne01 < 32`).
