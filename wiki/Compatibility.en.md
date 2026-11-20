# Compatibility & Distribution

> 🌐 [中文](Compatibility.md) | **English**

## 1. CPU variants

The `Lumina-Engine-Bin/` directory contains 14 CPU backend DLLs. At runtime, `ggml` picks the
**highest-featured variant whose requirements are satisfied** based on CPUID.

| DLL | Key features | Compiler | Tiled GEMM | VNNI | Typical CPU |
|-----|----------|--------|:---------:|:----:|----------|
| `ggml-cpu-x64` | baseline (SSE2) | MSVC | — | — | very old x64 |
| `ggml-cpu-sse42` | SSE4.2 | MSVC | — | — | Nehalem+ |
| `ggml-cpu-sandybridge` | AVX | MSVC | — | — | Sandy Bridge |
| `ggml-cpu-ivybridge` | AVX + F16C | Clang | — | — | Ivy Bridge |
| `ggml-cpu-piledriver` | AVX + F16C + FMA | Clang | — | — | AMD Piledriver |
| `ggml-cpu-haswell` | AVX2 + FMA + BMI2 | MSVC | ✅ | — | Haswell ~ Comet Lake (**this machine**) |
| `ggml-cpu-skylakex` | + AVX-512 | MSVC | ✅ | — | Skylake-X |
| `ggml-cpu-cannonlake` | + AVX-512 VBMI | MSVC | ✅ | — | Cannon Lake |
| `ggml-cpu-cascadelake` | + AVX-512 VNNI | MSVC | ✅ | ✅ | Cascade Lake |
| `ggml-cpu-icelake` | + AVX-512 VBMI + VNNI | MSVC | ✅ | ✅ | Ice Lake |
| `ggml-cpu-cooperlake` | + AVX-512 VNNI + BF16 | Clang | ✅ | ✅ | Cooper Lake |
| `ggml-cpu-alderlake` | AVX2 + AVX-VNNI | MSVC | ✅ | ✅ | Alder Lake ~ Rocket Lake |
| `ggml-cpu-zen4` | AVX-512 + VNNI + BF16 | Clang | ✅ | ✅ | AMD Zen 4/5 |
| `ggml-cpu-sapphirerapids` | AVX-512 + VNNI + BF16 + AMX | Clang | ✅ | ✅ | Sapphire Rapids |

> “Tiled GEMM” requires AVX2; “VNNI” additionally enables `vpdpbusd` when supported. Older CPUs fall
> back to the generic Q1_0 kernel — still **correct**, just slower.

---

## 2. Why MSVC can only build 9 variants

In CMake, `GGML_CPU_ALL_VARIANTS` skips 5 variants for MSVC:

```cmake
if (NOT MSVC)
    # __FMA__ / __F16C__ are undefined in MSVC (implied by AVX2/AVX512)
    ggml_add_cpu_backend_variant(ivybridge      SSE42 AVX F16C)
    ggml_add_cpu_backend_variant(piledriver     SSE42 AVX F16C FMA)
    ...
    # MSVC does not support BF16 / AMX intrinsics
    ggml_add_cpu_backend_variant(cooperlake     ... AVX512_BF16)
    ggml_add_cpu_backend_variant(zen4           ... AVX512_BF16)
    ggml_add_cpu_backend_variant(sapphirerapids ... AVX512_BF16 AMX_TILE AMX_INT8)
endif()
```

The full 14 variants therefore require Clang. This repo uses **Clang 20.1.8** to fill the gap.
(Also, VS2026's MSVC STL requires Clang ≥ 20; the Clang 16 Emscripten version fails with
`error STL1000: Unexpected compiler version, expected Clang 20 or newer`.)

---

## 3. Mixed assembly

Because:

- MSVC generates better code for the tiled kernel (pp128 ≈ 16.4 vs. Clang 14.9);
- only Clang can build all 14 variants;

the delivery uses a mix: **Clang toolchain + 14 variants, with the 9 common variants replaced by MSVC
builds**.

The CPU backend is an independent DLL interacting with `ggml-base.dll` over the C ABI; mixing has been
verified:

```
[MSVC ggml-cpu-haswell.dll] + [Clang Lumina-Engine-cli.exe / ggml-base.dll]  →  pp128 = 15.41 ✅
```

---

## 4. Runtime selection logic

The `ggml` DL loader (`ggml_backend_load_all_from_path`):

1. Scans `ggml-cpu-*.dll`.
2. Reads each one's self-described CPU feature requirements.
3. Checks whether the current CPU satisfies them using `cpuid`.
4. Picks the **highest-featured** one that is satisfied.

The same `Lumina-Engine-Bin/` directory therefore runs on different-CPU Windows machines without a
rebuild.

---

## 5. What this machine actually loads

```
load_backend: loaded CPU backend from ...\ggml-cpu-haswell.dll
```

Comet Lake has no AVX-512 / VNNI, so `haswell` (AVX2) is selected. To confirm which variant a machine
loaded, check the `Lumina-Engine-cli` startup log or the `backend` column of `Lumina-Engine-bench`.

---

## 6. Windows requirements

- Windows 10 1607+ / Windows 11 (x64).
- The Universal CRT is required (`api-ms-win-crt-*.dll`, bundled with Win10).
- The VC++ and OpenMP runtime libraries ship with `Lumina-Engine-Bin/`.

---

## 7. Distribution manifest

```
Lumina-Engine-Bin/
├─ 14 × ggml-cpu-*.dll
├─ ggml-base.dll  ggml.dll  ggml-rpc.dll  ggml-vulkan.dll
├─ Lumina-Engine.dll  Lumina-Engine-common.dll  mtmd.dll
├─ 20 × Lumina-Engine-*.exe  +  ggml-rpc-server.exe
├─ 8 × *-impl.dll
├─ libomp140.x86_64.dll   (Clang backends/tools)
├─ vcomp140.dll           (MSVC backends)
└─ msvcp140.dll  vcruntime140.dll  vcruntime140_1.dll
```

**55** files in total (21 exe + 8 impl + 14 CPU variants + 4 ggml DLLs + 3 engine DLLs + 5 runtimes).
Copy the whole folder to any compatible Windows x86-64 machine and run.

---

## 8. Binary lookup table (Q1LUT)

- Requires AVX2 (`haswell` and above); the `x64` / `sse42` / `sandybridge` variants fall back to the
  generic kernel.
- Available for all AVX2 variants (MSVC and Clang), **enabled by default**; `GGML_Q1_LUT=0` disables
  the whole path, `GGML_Q1_LUT_PP=0` disables only the pp LUT.
- Keep `-t` at or below the physical core count (this machine has 6 physical cores: `-t 6` helps,
  `-t 12` hurts).
- This machine uses **single-channel memory**, so tg is capped by bandwidth (theoretical ~15.7 tok/s);
  on dual-channel / higher-bandwidth platforms the tg gain approaches the kernel-level value.
- Needs ~0.95 GiB extra memory for the permuted weights and a one-time fill of ~0.7 s on first use.
- When disabled, behavior is identical to before the changes.

---

## 9. Model formats and quantization types

| Item | Support |
|------|---------|
| Quantization | `Q1_0` (Bonsai 1-bit, dedicated kernels), `Q2_0` / `TQ2_0` (ternary), `Q4_K` / `Q6_K`, `Q4_0` / `Q5_0` / `Q8_0`, `MXFP4` / `NVFP4`, `IQ*`, ... (per-backend support tables) |
| Architectures | All upstream architectures including `qwen35` / `qwen35moe`; `qwen35` also supports M-RoPE and mmproj multimodal input |
| PrismML rotated weights | **Supported since Build 33**: when the GGUF carries `prism.hadamard.*` metadata the engine reads the block-512 normalized Sylvester-Hadamard plus explicit +/-1 signs and rotates the activations accordingly (`Ternary-Bonsai-2-2B-Q2_0` verified) |
| PrismML mmproj | `qwen3vl_merger` projectors (`Ternary-Bonsai-2-2B-mmproj-Q8_0`) load with `Lumina-Engine-mtmd-cli`; the vision vectors use the raw embedding branch and are **not** rotated |

- Models without `prism.hadamard.*` metadata behave exactly as before (all of that code is gated).
- Only the documented values are accepted (`transform = normalized-sylvester-walsh-hadamard`,
  `axis = input-last-dimension`, `sign_mode = explicit`); anything else only warns and **skips** the
  rotation (never aborts).
- The rotation reuses the FWHT fast path (CPU / Vulkan / CUDA / Metal / SYCL); backends without
  `GGML_HINT_SRC0_IS_HADAMARD` fall back to a plain matrix product - same result, slower.
- Debug switches: `LUMINA_PRISM_DISABLE=1`, `LUMINA_PRISM_ORDER=hd|dh`, `LUMINA_PRISM_EMBD=inv|fwd|none`,
  `LUMINA_PRISM_OUT=fwd|inv|none`, `LUMINA_PRISM_NO_FWHT=1` (plain matmul instead of FWHT),
  `LUMINA_PRISM_DEBUG=1` (rotation bookkeeping, needs `-v`), `LUMINA_NO_FUSED_GDN=1`.
- Acceptance note: **batched perplexity being fine does not prove generation is fine** - always compare
  the generated text for `-ub 512` vs `-ub 1` and check that `perplexity -b 1` does not return `nan`.
