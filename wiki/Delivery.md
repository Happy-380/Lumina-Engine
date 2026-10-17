# 交付产物与生成配置

> 🌐 **中文** | [English](Delivery.en.md)

本页说明**最终交付产物的位置**、**如何一键生成**，以及**构建配置与目录约定**。
所有发布/分发都以本页的目录为准。

---

## 1. 交付目录

最终交付目录固定为：

```
C:\Lumina-Engine\Lumina-Engine-Bin          （等价写法： 仓库根目录\Lumina-Engine-Bin）
```

- 与源码目录 `llama.cpp-b10991\` 同级，位于仓库根 `C:\Lumina-Engine\` 下。
- 共 **55 个文件**：14 个 CPU 后端 DLL + `ggml-vulkan.dll` + 全部工具 + 运行库。
- 组装方式：有 clang 构建时以 clang 产物为基底（14 变体），再用 9 个 MSVC 变体覆盖；
  无 clang 时（本机现状）以 MSVC `build-release` 为基底，并从 `bench\clang_variants_backup\`
  还原 5 个 clang 专属变体（ivybridge / piledriver / cooperlake / zen4 / sapphirerapids）。
- 直接整体拷贝即可在兼容的 Windows x86-64 机器上运行，无需重新编译。
- 这是**唯一**的交付目录。`llama.cpp-b10991\build-release` / `build-clang` 只是**构建中间产物**，不是交付物。

```bat
cd C:\Lumina-Engine\Lumina-Engine-Bin
Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 512 -n 128 -t 6
Lumina-Engine-cli.exe   -m ..\Bonsai-8B.gguf -p "The capital of France is" -n 8 -t 6
```

---

## 2. 一键生成

仓库根目录提供一键脚本：

```bat
C:\Lumina-Engine\build-bin.bat
```

它依次执行三步，任一步失败即中止：

| 步骤 | 脚本 | 作用 |
|:--:|------|------|
| 1 | `bench\build_all_msvc.bat` | 编译 `llama.cpp-b10991\build-release`（**9 个 MSVC 变体** + 工具） |
| 2 | `bench\build_all_clang.bat` | 编译 `llama.cpp-b10991\build-clang`（**14 个变体** + 工具 + libomp） |
| 3 | `bench\assemble_bin.bat` | 组装到 `C:\Lumina-Engine\Lumina-Engine-Bin` |

> 只改了某一份源码、想重新组装时，可**单独**运行 `bench\assemble_bin.bat`（它会先删除并重建交付目录）。

---

## 3. 组装配置（`bench\assemble_bin.bat`）

脚本内的关键变量与行为：

```bat
set "ROOT=%~dp0.."                                  REM 仓库根 = C:\Lumina-Engine
set "SRC_CLANG=%ROOT%\llama.cpp-b10991\build-clang\bin"
set "SRC_MSVC =%ROOT%\llama.cpp-b10991\build-release\bin"
set "DST=%ROOT%\Lumina-Engine-Bin"                  REM 交付目录
```

组装步骤：

1. **清空交付目录**：若 `DST` 存在，先 `rd /s /q "%DST%"`，再 `mkdir`（确保不残留旧文件）。
2. **clang 全量铺底**：`xcopy /y /q /i "%SRC_CLANG%\*" "%DST%\"`（14 个变体 + 全部工具）。
3. **MSVC 覆盖 9 个常用变体**（这些变体的分块内核代码生成更好）：
   `x64 sse42 sandybridge haswell skylakex cannonlake cascadelake icelake alderlake`。
4. **补齐 5 个运行库**（从 `%SystemRoot%\System32` 拷贝，找不到则告警）：
   `libomp140.x86_64.dll`、`msvcp140.dll`、`vcomp140.dll`、`vcruntime140.dll`、`vcruntime140_1.dll`。
5. 打印写入的文件数量（正常为 **55**）。

> 注意：运行库只从系统目录取，**不再回退到任何旧交付目录**。

---

## 4. 生成配置（CMake）

### 4.1 MSVC：9 变体（`configure-release.bat`）

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

### 4.2 Clang 20：14 变体（`configure-clang.bat`）

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

关键选项说明：

| 选项 | 作用 |
|------|------|
| `GGML_BACKEND_DL=ON` | 后端编译为独立 DLL（运行时按 CPUID 动态加载） |
| `GGML_CPU_ALL_VARIANTS=ON` | 为每个 x86 微架构各编译一份 CPU 后端 |
| `GGML_NATIVE=OFF` | 不针对本机优化，保证多份变体可分发 |
| `BUILD_SHARED_LIBS=ON` | `GGML_BACKEND_DL` 的前提 |
| `GGML_RPC=ON` | 生成 `ggml-rpc.dll` / `ggml-rpc-server.exe` |

---

## 5. 目录与脚本一览

```
C:\Lumina-Engine\
├─ build-bin.bat                     # ★ 一键：MSVC + Clang 编译 + 组装到 Lumina-Engine-Bin
├─ configure-release.bat             # 配置 MSVC 构建树 build-release（9 变体）
├─ build-release.bat                 # 增量编译 build-release
├─ configure-clang.bat               # 配置 Clang 构建树 build-clang（14 变体）
├─ build-clang.bat                   # 增量编译 build-clang
├─ configure.bat / build.bat         # 旧单变体本机开发构建（build\，不参与交付）
├─ Lumina-Engine-Bin\                # ★ 唯一交付目录（55 文件）
├─ llama.cpp-b10991\
│   ├─ build-release\bin\            # 中间产物（MSVC，9 变体）
│   └─ build-clang\bin\              # 中间产物（Clang，14 变体）
└─ bench\
    ├─ assemble_bin.bat              # ★ 组装脚本（clang 基底 + 9 个 MSVC 覆盖 + 运行库）
    ├─ build_all_msvc.bat            # 编译 build-release
    ├─ build_all_clang.bat           # 编译 build-clang
    └─ build_haswell.bat / build_x64.bat   # 单变体增量编译（开发用）
```

---

## 6. 约定（重要）

1. **以后所有编译产物只放 `.\Lumina-Engine-Bin`。** 不要新增第二份交付目录。
2. 旧的 `C:\Lumina-Engine\llama` 入口**已删除**，其内容由 `Lumina-Engine-Bin` 取代。
3. `build-release` / `build-clang` 是**构建中间产物**，供 `assemble_bin.bat` 取件；不是交付物。
4. 旧的单变体 `build\` 目录已清理，仅保留 `configure.bat` + `build.bat` 供本机快速开发重建，
   **不参与交付**。
5. 交付目录由脚本整体重建（先删后建），因此不要在 `Lumina-Engine-Bin` 内放置需要保留的手工文件。

---

## 7. 生成后验证

```bat
cd C:\Lumina-Engine\Lumina-Engine-Bin

REM 1) 版本横幅：应显示 Build 24 (Based on llama.cpp b-10991)
Lumina-Engine-cli.exe --version
Lumina-Engine-server.exe --help
Lumina-Engine-train.exe --version
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf -p "The capital of France is" -n 8 -t 6 --temp 0

REM 2) 确认交付文件数（应为 54）
dir /b | find /c /v ""

REM 3) 确认 Q1LUT 已启用（AVX2 机器）
set GGML_Q1_LUT_DEBUG=1
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf -p "hi" -n 1 -t 6
REM 应打印 q1lut alloc 行；若不想看日志，运行后清除该变量
set GGML_Q1_LUT_DEBUG=
```

预期：

- 横幅显示 `Build 24 (Based on llama.cpp b-10991)`；`cli` / `server` / `train` 等所有控制台
  启动时均打印同一套 ASCII logo + Build 号 + 项目链接；
- `-p "The capital of France is"` 输出 `Paris.`；
- 交付目录 54 个文件；
- AVX2 机器上默认启用 Q1LUT（`q1lut alloc` 日志 / 关闭用 `GGML_Q1_LUT=0`）。

---

## 8. 相关页面

- [Building](Building.md) — 编译细节（单变体 / 多变体 / 双工具链）
- [Compatibility](Compatibility.md) — 14 个 CPU 变体与运行时分发
- [Troubleshooting](Troubleshooting.md) — 常见问题与排障
