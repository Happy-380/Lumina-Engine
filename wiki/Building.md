# 编译指南

> 🌐 **中文** | [English](Building.en.md)

## 1. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Windows | 10 / 11 x64 | |
| CMake | ≥ 3.14 | 本项目使用 4.x |
| Ninja | 任意 | 也可用 Visual Studio generator |
| MSVC | VS 2022 / 2026 | 编译 9 个常用 CPU 变体 |
| Clang | ≥ 20（可选） | 编译 5 个 clang 专属变体、OpenMP(libomp) |
| LLVM | 20.1.8 | 仅在做完整 14 变体时需要 |

> 说明：MSVC 只能构建 9 个变体（`x64/sse42/sandybridge/haswell/skylakex/
> cannonlake/cascadelake/icelake/alderlake`）。`ivybridge/piledriver/cooperlake/
> zen4/sapphirerapids` 需要 Clang，因为 MSVC 不定义 `__F16C__/__FMA__`，也不支持
> AVX512-BF16 / AMX intrinsic。另外 VS2026 的 MSVC STL 要求 Clang ≥ 20。

---

## 2. 简单单变体编译（最快，仅本机架构）

```bat
cmake -S llama.cpp-b10991 -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLAMA_BUILD_SERVER=ON ^
  -DLLAMA_BUILD_TESTS=ON
cmake --build build -j
```

产物位于 `build/bin/`，只有一个 `ggml-cpu.dll`（按 `GGML_NATIVE`/手动特性编译）。

如果你只要在本机跑，这样最简单；但**分发到其它 CPU 需要下面的多变体构建**。

---

## 3. 多变体构建（MSVC，9 个变体）

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

关键选项：

| 选项 | 作用 |
|------|------|
| `GGML_BACKEND_DL=ON` | 后端编译为独立 DLL（运行时动态加载） |
| `GGML_CPU_ALL_VARIANTS=ON` | 为每个 x86 微架构编译一份 CPU 后端 |
| `GGML_NATIVE=OFF` | 不针对本机优化，保证多变体 |
| `BUILD_SHARED_LIBS=ON` | `GGML_BACKEND_DL` 的前提 |
| `GGML_RPC=ON` | 生成 `ggml-rpc.dll` / `ggml-rpc-server.exe` |
| `GGML_VULKAN=ON` | 生成 `ggml-vulkan.dll`；`configure-release.bat` 检测到 `D:\VulkanSDK\sdk\Bin\glslc.exe` 时自动启用，否则回退 CPU-only |

> Build 12 起默认交付带 Vulkan 后端。本机 Vulkan SDK 由 LunarG 安装包的 7z 载荷解包到
> `D:\VulkanSDK\sdk`（无需管理员权限）。运行时仅需系统自带的 `vulkan-1.dll`。
> 默认设备策略与实测性能见 [OPTIMIZATION_LOG.md](../OPTIMIZATION_LOG.md)（`-ngl 0` = 层留 CPU + 算子 offload）。

配置阶段会打印每个变体及其编译标志：

```
-- Adding CPU backend variant ggml-cpu-haswell: /arch:AVX2 GGML_AVX2;GGML_FMA;...
-- Adding CPU backend variant ggml-cpu-cascadelake: /arch:AVX512 GGML_AVX512;__AVX512VNNI__;...
-- Adding CPU backend variant ggml-cpu-alderlake: /arch:AVX2 GGML_AVX2;...;__AVXVNNI__;...
```

---

## 4. 多变体构建（Clang 20，14 个变体 + libomp）

先安装 LLVM 20，然后：

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

Clang 会额外构建 `ivybridge/piledriver/cooperlake/zen4/sapphirerapids`，
并链接 `libomp140.x86_64.dll`。

---

## 5. 混合组装（推荐交付方式）

MSVC 对分块内核的代码生成略优（+11% vs Clang），Clang 才能构建全部 14 个变体。
因此交付时把两者合并：

仓库根目录提供了一键脚本，默认组装到 `C:\Lumina-Engine\Lumina-Engine-Bin`：

```bat
C:\Lumina-Engine\build-bin.bat
```

它等价于：

```bat
REM 1. 以 clang 产物为基底（14 变体 + 全部工具 + libomp）
xcopy /E /I build-clang\bin\* ..\Lumina-Engine-Bin\

REM 2. 用 MSVC 版本覆盖 9 个常用变体（分块内核更快）
for %v in (x64 sse42 sandybridge haswell skylakex cannonlake cascadelake icelake alderlake) do ^
  copy /Y build-release\bin\ggml-cpu-%v.dll ..\Lumina-Engine-Bin\

REM 3. 补齐运行库
copy C:\Windows\System32\libomp140.x86_64.dll ..\Lumina-Engine-Bin\
copy C:\Windows\System32\vcomp140.dll          ..\Lumina-Engine-Bin\
copy C:\Windows\System32\msvcp140.dll          ..\Lumina-Engine-Bin\
copy C:\Windows\System32\vcruntime140.dll      ..\Lumina-Engine-Bin\
copy C:\Windows\System32\vcruntime140_1.dll    ..\Lumina-Engine-Bin\
```

> 上表为等价示意；**实际脚本为 `bench\assemble_bin.bat`**：它会先删除并重建 `Lumina-Engine-Bin`，
> 用 clang 全量铺底、9 个 MSVC 变体覆盖，并从 `%SystemRoot%\System32` 拷贝 5 个运行库
> （不再回退到任何旧交付目录）。详见 [Delivery](Delivery.md)。
>
> CPU 后端 DLL 是独立模块、走 C ABI，已验证 MSVC 与 Clang 构建的 CPU 后端可以
> 与另一编译器构建的 `ggml-base.dll` / 工具混用。

---

## 6. 产物布局

```
build-*/bin/
├─ ggml-base.dll          # 张量/内存/图
├─ ggml.dll               # 调度
├─ ggml-rpc.dll           # RPC 后端
├─ ggml-cpu-<arch>.dll    # 每个微架构一个 CPU 后端
├─ Lumina-Engine.dll              # 模型加载/解码
├─ Lumina-Engine-common.dll       # 公共工具
├─ mtmd.dll               # 多模态
├─ Lumina-Engine-*.exe            # 20 个工具（cli/bench/server/perplexity/...）
└─ *-impl.dll             # 各工具的伴随库
```

`Lumina-Engine-Bin/`（即仓库根目录下、与源码目录同级）为最终的 53 文件交付集（含 14 个 CPU 变体 + 运行库），
也是**唯一的交付目录**；上面的 `build-*/bin/` 只是构建中间产物。旧入口 `C:\Lumina-Engine\llama` 已删除。
也可直接调用 `bench\assemble_bin.bat` 只做组装。详见 [Delivery](Delivery.md)。

---

## 7. 运行时依赖

| DLL | 来源 | 何时需要 |
|-----|------|----------|
| `libomp140.x86_64.dll` | LLVM / System32 | Clang 构建的二进制 |
| `vcomp140.dll` | MSVC / System32 | MSVC 构建的 CPU 后端 |
| `msvcp140/vcruntime140(_1).dll` | VC++ 运行库 | MSVC 构建 |

`Lumina-Engine-Bin/` 已包含以上全部。

---

## 8. 增量编译单个变体（开发时最快）

只改了 `q1t.cpp` 或 `quants.c` 时，无需重编全部：

```bat
cmake --build build-release --target ggml-cpu-haswell -j
```

约 1 分钟即可得到新的 `ggml-cpu-haswell.dll`（本机使用的变体）。

---

## 9. 验证构建

```bat
REM 确认 VNNI 是否生成
dumpbin /disasm build-release\bin\ggml-cpu-cascadelake.dll | findstr vpdpbusd

REM 运行后端算子测试（需使用开启了 -DLLAMA_BUILD_TESTS=ON 的构建树）
build-release\bin\test-backend-ops.exe test -b CPU -o MUL_MAT -p "type_a=q1_0"

REM 运行模型（交付目录）
Lumina-Engine-Bin\Lumina-Engine-cli.exe -m Bonsai-8B.gguf -p "hi" -n 8 -t 6
```
