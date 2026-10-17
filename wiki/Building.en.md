# Building Guide

> 🌐 [中文](Building.md) | **English**

## 1. Requirements

| Component | Version | Note |
|------|------|------|
| Windows | 10 / 11 x64 | |
| CMake | ≥ 3.14 | this project uses 4.x |
| Ninja | any | Visual Studio generator also works |
| MSVC | VS 2022 / 2026 | builds the 9 common CPU variants |
| Clang | ≥ 20 (optional) | builds the 5 clang-only variants and OpenMP (libomp) |
| LLVM | 20.1.8 | only needed for the full 14-variant build |

> MSVC can only build 9 variants (`x64/sse42/sandybridge/haswell/skylakex/
> cannonlake/cascadelake/icelake/alderlake`). `ivybridge/piledriver/cooperlake/
> zen4/sapphirerapids` require Clang because MSVC does not define `__F16C__/__FMA__` and does not
> support AVX512-BF16 / AMX intrinsics. Also, VS2026's MSVC STL requires Clang ≥ 20.

---

## 2. Simple single-variant build (fastest, host arch only)

```bat
cmake -S llama.cpp-b10991 -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLAMA_BUILD_SERVER=ON ^
  -DLLAMA_BUILD_TESTS=ON
cmake --build build -j
```

Artifacts land in `build/bin/`, with a single `ggml-cpu.dll` (compiled for `GGML_NATIVE` or manually
selected features).

If you only run on this machine, this is the simplest path; **distributing to other CPUs needs the
multi-variant build below**.

---

## 3. Multi-variant build (MSVC, 9 variants)

```bat
call "D:\VS2026\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\CMake\bin;...\Ninja;%PATH%

cmake -S llama.cpp-b10991 -B build-release -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=ON ^
  -DGGML_BACKEND_DL=ON ^
  -DGGML_CPU_ALL_VARIANTS=ON ^
  -DGGML_NATIVE=OFF ^
  -DGGML_RPC=ON ^
  -DGGML_OPENMP=ON ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_TOOLS=ON ^
  -DLLAMA_BUILD_SERVER=ON ^
  -DLLAMA_BUILD_APP=OFF ^
  -DLLAMA_CURL=OFF

cmake --build build-release -j 12
```

Key options:

| Option | Effect |
|------|------|
| `GGML_BACKEND_DL=ON` | compile backends as separate DLLs (loaded dynamically at runtime) |
| `GGML_CPU_ALL_VARIANTS=ON` | one CPU backend per x86 microarchitecture |
| `GGML_NATIVE=OFF` | no host-specific tuning, so variants can ship |
| `BUILD_SHARED_LIBS=ON` | prerequisite for `GGML_BACKEND_DL` |
| `GGML_RPC=ON` | produce `ggml-rpc.dll` / `ggml-rpc-server.exe` |

The configure step prints each variant and its flags:

```
-- Adding CPU backend variant ggml-cpu-haswell: /arch:AVX2 GGML_AVX2;GGML_FMA;...
-- Adding CPU backend variant ggml-cpu-cascadelake: /arch:AVX512 GGML_AVX512;__AVX512VNNI__;...
-- Adding CPU backend variant ggml-cpu-alderlake: /arch:AVX2 GGML_AVX2;...;__AVXVNNI__;...
```

---

## 4. Multi-variant build (Clang 20, 14 variants + libomp)

Install LLVM 20 first, then:

```bat
call "D:\VS2026\VC\Auxiliary\Build\vcvars64.bat"
set PATH=D:\llvm\bin;C:\Program Files\CMake\bin;...\Ninja;%PATH%

cmake -S llama.cpp-b10991 -B build-clang -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE=cmake/x64-windows-llvm.cmake ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=ON ^
  -DGGML_BACKEND_DL=ON ^
  -DGGML_CPU_ALL_VARIANTS=ON ^
  -DGGML_NATIVE=OFF ^
  -DGGML_RPC=ON ^
  -DGGML_OPENMP=ON ^
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON ^
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_APP=OFF -DLLAMA_CURL=OFF

cmake --build build-clang -j 12
```

Clang additionally builds `ivybridge/piledriver/cooperlake/zen4/sapphirerapids` and links
`libomp140.x86_64.dll`.

---

## 5. Mixed assembly (recommended delivery method)

MSVC generates slightly better code for the tiled kernel (+11% vs Clang), and only Clang can build
all 14 variants. The delivery therefore merges both:

The repo root provides a one-command script; it assembles into `C:\Lumina-Engine\Lumina-Engine-Bin` by default:

```bat
C:\Lumina-Engine\build-bin.bat
```

It is equivalent to:

```bat
REM 1. clang output as the base (14 variants + all tools + libomp)
xcopy /E /I build-clang\bin\* ..\Lumina-Engine-Bin\

REM 2. overwrite the 9 common variants with MSVC builds (faster tiled kernel)
for %v in (x64 sse42 sandybridge haswell skylakex cannonlake cascadelake icelake alderlake) do ^
  copy /Y build-release\bin\ggml-cpu-%v.dll ..\Lumina-Engine-Bin\

REM 3. add the runtime libraries
copy C:\Windows\System32\libomp140.x86_64.dll ..\Lumina-Engine-Bin\
copy C:\Windows\System32\vcomp140.dll          ..\Lumina-Engine-Bin\
copy C:\Windows\System32\msvcp140.dll          ..\Lumina-Engine-Bin\
copy C:\Windows\System32\vcruntime140.dll      ..\Lumina-Engine-Bin\
copy C:\Windows\System32\vcruntime140_1.dll    ..\Lumina-Engine-Bin\
```

> The block above is illustrative; the **actual script is `bench\assemble_bin.bat`**: it first
> deletes and recreates `Lumina-Engine-Bin`, lays down the full clang output, overwrites the 9 MSVC
> variants, and copies the 5 runtime libraries from `%SystemRoot%\System32` (no fallback to any old
> delivery directory). See [Delivery](Delivery.en.md).
>
> CPU backend DLLs are independent modules using the C ABI; MSVC- and Clang-built CPU backends have
> been verified to mix with a `ggml-base.dll` / tools built by the other compiler.

---

## 6. Artifact layout

```
build-*/bin/
├─ ggml-base.dll          # tensors / memory / graph
├─ ggml.dll               # scheduler
├─ ggml-rpc.dll           # RPC backend
├─ ggml-cpu-<arch>.dll    # one CPU backend per microarchitecture
├─ Lumina-Engine.dll              # model loading / decoding
├─ Lumina-Engine-common.dll       # common utilities
├─ mtmd.dll               # multimodal
├─ Lumina-Engine-*.exe            # 20 tools (cli/bench/server/perplexity/...)
└─ *-impl.dll             # companion libraries for the tools
```

`Lumina-Engine-Bin/` (at the repo root, next to the source directory) is the final 53-file delivery
set (14 CPU variants + runtime libraries), and it is the **only** delivery directory; the
`build-*/bin/` outputs above are just build intermediates. The old `C:\Lumina-Engine\llama` entry has been
removed. You can also call `bench\assemble_bin.bat` directly to assemble only. See [Delivery](Delivery.en.md).

---

## 7. Runtime dependencies

| DLL | Source | Needed when |
|-----|------|----------|
| `libomp140.x86_64.dll` | LLVM / System32 | Clang-built binaries |
| `vcomp140.dll` | MSVC / System32 | MSVC-built CPU backends |
| `msvcp140/vcruntime140(_1).dll` | VC++ runtime | MSVC builds |

`Lumina-Engine-Bin/` already contains all of them.

---

## 8. Incremental single-variant build (fastest during development)

When only `q1t.cpp` or `quants.c` changed, no full rebuild is needed:

```bat
cmake --build build-release --target ggml-cpu-haswell -j
```

A new `ggml-cpu-haswell.dll` (the variant used on this machine) is ready in about a minute.

---

## 9. Verifying the build

```bat
REM Confirm VNNI is emitted
dumpbin /disasm build-release\bin\ggml-cpu-cascadelake.dll | findstr vpdpbusd

REM Run the backend op tests (requires a tree configured with -DLLAMA_BUILD_TESTS=ON)
build-release\bin\test-backend-ops.exe test -b CPU -o MUL_MAT -p "type_a=q1_0"

REM Run the model (delivery directory)
Lumina-Engine-Bin\Lumina-Engine-cli.exe -m Bonsai-8B.gguf -p "hi" -n 8 -t 6
```
