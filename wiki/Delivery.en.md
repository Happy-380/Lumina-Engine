# Delivery Artifacts & Build Configuration

> 🌐 [中文](Delivery.md) | **English**

This page describes **where the final delivery artifacts live**, **how to produce them in one
command**, and the **build configuration / directory conventions**. All releases and
distributions follow the directory described here.

---

## 1. Delivery directory

The final delivery directory is fixed to:

```
C:\Lumina-Engine\Lumina-Engine-Bin          (equivalently: <repo root>\Lumina-Engine-Bin)
```

- It sits next to the source directory `llama.cpp-b10991\`, under the repo root `C:\Lumina-Engine\`.
- **55 files** in total: 14 CPU backend DLLs + all tools + runtime libraries.
- Assembly layout: **Clang build as the base (14 variants + tools + libomp), then the 9 MSVC
  variants overwrite their clang counterparts.**
- Copy the whole folder to any compatible Windows x86-64 machine — no rebuild required.
- This is the **only** delivery directory. `llama.cpp-b10991\build-release` / `build-clang` are
  merely **build intermediates**, not deliverables.

```bat
cd C:\Lumina-Engine\Lumina-Engine-Bin
Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 512 -n 128 -t 6
Lumina-Engine-cli.exe   -m ..\Bonsai-8B.gguf -p "The capital of France is" -n 8 -t 6
```

---

## 2. One-command build

The repo root provides a single entry-point script:

```bat
C:\Lumina-Engine\build-bin.bat
```

It runs three steps and aborts on the first failure:

| Step | Script | Purpose |
|:--:|------|------|
| 1 | `bench\build_all_msvc.bat` | Build `llama.cpp-b10991\build-release` (**9 MSVC variants** + tools) |
| 2 | `bench\build_all_clang.bat` | Build `llama.cpp-b10991\build-clang` (**14 variants** + tools + libomp) |
| 3 | `bench\assemble_bin.bat` | Assemble into `C:\Lumina-Engine\Lumina-Engine-Bin` |

> If you only changed sources and want to re-assemble, run `bench\assemble_bin.bat` alone
> (it deletes and recreates the delivery directory first).

---

## 3. Assembly configuration (`bench\assemble_bin.bat`)

Key variables inside the script:

```bat
set "ROOT=%~dp0.."                                  REM repo root = C:\Lumina-Engine
set "SRC_CLANG=%ROOT%\llama.cpp-b10991\build-clang\bin"
set "SRC_MSVC =%ROOT%\llama.cpp-b10991\build-release\bin"
set "DST=%ROOT%\Lumina-Engine-Bin"                  REM delivery directory
```

Assembly steps:

1. **Wipe the destination**: if `DST` exists, `rd /s /q "%DST%"` then `mkdir` (no stale files).
2. **Lay down the full clang build**: `xcopy /y /q /i "%SRC_CLANG%\*" "%DST%\"` (14 variants + all tools).
3. **Overwrite 9 common variants with MSVC builds** (better codegen for the tiled kernel):
   `x64 sse42 sandybridge haswell skylakex cannonlake cascadelake icelake alderlake`.
4. **Copy the 5 runtime libraries** from `%SystemRoot%\System32` (warn if missing):
   `libomp140.x86_64.dll`, `msvcp140.dll`, `vcomp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`.
5. Print the number of files written (normally **55**).

> The runtime libraries are taken from the system directory only — there is **no fallback to any
> old delivery directory**.

---

## 4. Build configuration (CMake)

### 4.1 MSVC: 9 variants (`configure-release.bat`)

```bat
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

### 4.2 Clang 20: 14 variants (`configure-clang.bat`)

```bat
cmake -S llama.cpp-b10991 -B build-clang -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE=cmake/x64-windows-llvm.cmake ^
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
cmake --build build-clang -j 12
```

Key options:

| Option | Effect |
|------|------|
| `GGML_BACKEND_DL=ON` | Compile backends as separate DLLs (loaded at runtime by CPUID) |
| `GGML_CPU_ALL_VARIANTS=ON` | One CPU backend per x86 micro-architecture |
| `GGML_NATIVE=OFF` | No host-specific optimization, so all variants can ship |
| `BUILD_SHARED_LIBS=ON` | Prerequisite for `GGML_BACKEND_DL` |
| `GGML_RPC=ON` | Produce `ggml-rpc.dll` / `ggml-rpc-server.exe` |

---

## 5. Directory / script overview

```
C:\Lumina-Engine\
├─ build-bin.bat                     # ★ one-click: MSVC + Clang build + assemble into Lumina-Engine-Bin
├─ configure-release.bat             # configure MSVC tree build-release (9 variants)
├─ build-release.bat                 # incremental build of build-release
├─ configure-clang.bat               # configure Clang tree build-clang (14 variants)
├─ build-clang.bat                   # incremental build of build-clang
├─ configure.bat / build.bat         # legacy single-variant dev build (build\, not delivered)
├─ Lumina-Engine-Bin\                # ★ the only delivery directory (55 files)
├─ llama.cpp-b10991\
│   ├─ build-release\bin\            # intermediate (MSVC, 9 variants)
│   └─ build-clang\bin\              # intermediate (Clang, 14 variants)
└─ bench\
    ├─ assemble_bin.bat              # ★ assembler (clang base + 9 MSVC overrides + runtimes)
    ├─ build_all_msvc.bat            # build build-release
    ├─ build_all_clang.bat           # build build-clang
    └─ build_haswell.bat / build_x64.bat   # single-variant incremental builds (dev)
```

---

## 6. Conventions (important)

1. **All future build output goes to `.\Lumina-Engine-Bin` only.** Do not add a second delivery directory.
2. The old `C:\Lumina-Engine\llama` entry **has been deleted**; it is superseded by `Lumina-Engine-Bin`.
3. `build-release` / `build-clang` are **build intermediates** consumed by `assemble_bin.bat`; they are not deliverables.
4. The legacy single-variant `build\` tree was cleaned up. `configure.bat` + `build.bat` remain for quick
   local development only and **do not participate in delivery**.
5. The delivery directory is rebuilt wholesale (delete-then-create), so do not keep hand-made files inside
   `Lumina-Engine-Bin`.

---

## 7. Post-build verification

```bat
cd C:\Lumina-Engine\Lumina-Engine-Bin

REM 1) Banner: should show Build 24 (Based on llama.cpp b-10991)
Lumina-Engine-cli.exe --version
Lumina-Engine-server.exe --help
Lumina-Engine-train.exe --version
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf -p "The capital of France is" -n 8 -t 6 --temp 0

REM 2) Confirm the file count (should be 54)
dir /b | find /c /v ""

REM 3) Confirm Q1LUT is active (AVX2 machines)
set GGML_Q1_LUT_DEBUG=1
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf -p "hi" -n 1 -t 6
REM expect q1lut alloc lines; clear the variable afterwards
set GGML_Q1_LUT_DEBUG=
```

Expected:

- Banner shows `Build 24 (Based on llama.cpp b-10991)`; every console (`cli` / `server` / `train`
  / ...) prints the same ASCII logo + build number + project links at startup;
- `-p "The capital of France is"` prints `Paris.`;
- 54 files in the delivery directory;
- Q1LUT is enabled by default on AVX2 machines (`q1lut alloc` logs; disable with `GGML_Q1_LUT=0`).

---

## 8. Related pages

- [Building](Building.en.md) — build details (single / multi-variant / dual toolchain)
- [Compatibility](Compatibility.en.md) — 14 CPU variants and runtime distribution
- [Troubleshooting](Troubleshooting.en.md) — FAQ and diagnostics
