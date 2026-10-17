# Lumina-Engine

> 🌐 **中文** | [English](README.en.md)

> 基于 llama.cpp b-10991 的 Windows x86-64 推理 / 压缩 / 训练引擎：为 **Bonsai Q1_0** 提供专用 CPU 内核，同时提供面向更多模型的 **Vulkan 加速、自动设备策略、无损压缩与 QLoRA 训练**。

![Build](https://img.shields.io/badge/build-24-blue)
![llama.cpp](https://img.shields.io/badge/llama.cpp-b10991-green)
![Platform](https://img.shields.io/badge/platform-Windows%20x86--64-lightgrey)
![License](https://img.shields.io/badge/license-AGPL--3.0-orange)
[![Support on Afdian](https://img.shields.io/badge/Support-Afdian-ff69b4)](https://afdian.com/a/lumi_380)

[快速开始](#快速开始) · [优化范围](#优化范围) · [性能实测](#性能实测) · [文档](wiki/Home.md) · [English](README.en.md) · [贡献者](CONTRIBUTORS.md) · [❤ 支持](#支持这个项目)

**快速了解：**

- **这是什么**：llama.cpp b-10991 的增强分支。最核心的是针对 `GGML_TYPE_Q1_0`（1.125 bit/权重）的纯 CPU 快速路径；同时也把通用推理加速、GGUF 无损压缩和量化模型 QLoRA 训练整合进同一套交付。
- **给谁用**：想在普通 Windows 机器上低成本跑 1-bit 模型的人；用 Q4_K / Q8_0 模型做推理、压缩或微调的人；以及需要在 8 GB 内存上做长上下文训练的人。
- **核心差异**：不止"能跑"——Q1_0 预填充最高 **+43%**、token 生成 **+30%**；Q4_K 模型无损压到 **94.28%**；量化模型可在纯 CPU 上 QLoRA，8 GB 内存可跑 32K 上下文。

**最快跑起来（模型已就位）：**

```bat
cd /d C:\Lumina-Engine\Lumina-Engine-Bin
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf -p "你好，请介绍一下自己。" -n 128 -t 6
```

> 还没有模型？先看 [快速开始](#快速开始) 的前置条件。

---

## 目录

- [为什么需要它](#为什么需要它)
- [特性](#特性)
- [优化范围](#优化范围)
- [快速开始](#快速开始)
- [常用场景](#常用场景)
- [性能实测](#性能实测)
- [自动设备策略（CPU / GPU）](#自动设备策略cpu--gpu)
- [兼容性](#兼容性)
- [从源码构建](#从源码构建)
- [环境变量](#环境变量)
- [架构与目录](#架构与目录)
- [FAQ](#faq)
- [更新日志](#更新日志)
- [文档](#文档)
- [支持这个项目](#支持这个项目)
- [贡献](#贡献)
- [许可证与致谢](#许可证与致谢)

---

## 为什么需要它

Bonsai-8B 的全部权重都是 1-bit（`Q1_0`，128 个权重共享一个 fp16 尺度），模型只有 **1.07 GiB**，看起来很轻。但在 CPU 上：

- **Token 生成（tg）**：每生成 1 个 token 都要把 1.07 GiB 权重完整读一遍 → **内存带宽瓶颈**。单通道 DDR4 上，理论上限只有 ~15.7 tok/s。
- **Prompt 处理（pp）**：权重可以复用 → **计算瓶颈**，时间几乎全花在 `Q1_0 × Q8_0` 点积内核上。

上游通用 Q1_0 内核每 32 个权重就做一次浮点转换，且没有利用二值权重的特殊结构。Lumina-Engine 重新设计了这条路径：

| 优化 | 效果（本机 i5-10400，单通道 8 GB） |
|---|---|
| per-128 激活量化 + 寄存器分块 GEMM | pp **+28.5% ~ +42.5%** |
| 单列 GEMV（token 生成专用） | tg **+12% ~ +14%**（12 线程） |
| 二值查表 Q1LUT（默认启用） | tg **+30%**；pp512 **+26%**、pp2048 **+57%** |
| 14 个 CPU 微架构变体 | 同一份预编译包，几乎所有 x86-64 CPU 都能吃到最优内核 |

> 所有数字都是**同场 A/B 的比值**。测试机是单通道内存，tg 绝对值被带宽压低；双通道 / 高带宽平台上收益更接近内核层面。
>
> 以上重点在 Q1_0 路径；Vulkan、无损压缩、QLoRA 训练等通用优化见[优化范围](#优化范围)。

---

## 特性

- **1-bit 专用内核**：per-128 激活（Q8_128）、2×4 寄存器分块、单列 GEMV、VNNI（`vpdpbusd`）、`vpsignb`；全部**严格隔离**在 Q1_0 + AVX2 路径，其它模型零影响。
- **优化不止 Q1_0**：14 个 CPU 变体、Vulkan 后端与自动设备策略、GGUF 无损压缩、量化模型 QLoRA、长上下文训练内存优化，覆盖更广的模型与量化类型。
- **真正的二值查表（Q1LUT）**：利用 `T[c ^ 0xf] = -T[c]` 的对称性只建半张表，tg 默认走查表，pp 在 `ne11 >= 8` 时自动切换；`GGML_Q1_LUT=0` 可一键关闭。
- **数值无损**：Q1_0 位运算完全精确；47/47 后端测试通过，困惑度与通用路径同量级（差异 < 0.2%），贪婪解码输出一致。
- **一套包，14 种 CPU**：SSE4.2 / Sandy Bridge / Haswell / Skylake-X / Ice Lake / Alder Lake / Zen 4 / Sapphire Rapids……运行时按 CPUID 自动选择。
- **CPU / GPU 自动决策**：无 GPU 走纯 CPU；集显做算子 offload；独显按显存整层 offload；MoE 模型自动避免"越 offload 越慢"。
- **模型无损压缩**：`.gguf.lumina` 容器把 Q4_K / Q6_K 的 `d/dmin/scales` 与权重流分开排列后再 DEFLATE（F32 张量单独压缩）。以 `SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf` 为例，2.63 GB → 2.48 GB（**94.28%**），解压后 SHA-256 与原始文件逐字节一致。
- **内置训练引擎**：`Lumina-Engine-train` 直接对 Q4_K / Q8_0 / Q1_0 量化模型做 **QLoRA**，支持 JSONL 数据集、检查点、断点续训、MoE 专家适配器，以及 8 GB 机器上的 32K 长上下文训练（注意力反向现场重算 + 数据集 mmap，内存稳定约 3.5 GiB）。
- **一条命令交付**：`build-bin.bat` 依次编译 MSVC + Clang 双树并组装运行目录，不需要手工拷贝 DLL。

---

## 优化范围

并非所有改动都只服务于 1-bit：Q1_0 内核严格隔离，其余优化对更多模型和量化类型生效。

| 优化 | 生效范围 |
|---|---|
| per-128 激活、分块 GEMM、单列 GEMV、Q1LUT | **仅 Q1_0**（Bonsai 系列）；非 Q1_0 路径零改动 |
| Vulkan Q1_0 DP4A GEMV | **仅 Q1_0 + Vulkan 独显**（集显默认关闭） |
| 14 个 CPU 变体、自动设备策略、Vulkan 算子 offload | **所有模型**（dense / MoE，按模型类型自动选择策略） |
| QLoRA 训练、检查点 / 续训、长上下文内存优化 | **Q4_K / Q8_0 / Q1_0**，dense + MoE |
| `.gguf.lumina` 无损压缩 | **含 Q4_K / Q6_K / F32 张量的 GGUF**（当前打包器支持的类型） |
| 控制台工具、一键构建、训练 ETA / 线程优化 | 所有工具与训练任务 |

### 其他模型上的实测收益

| 场景 | 结果 |
|---|---|
| Qwen3-0.6B-Q8_0 + Intel 集显算子 offload（`-ngl 0`） | pp512 **约 2.4×**（420 vs 207 t/s），tg 持平 |
| phi-3-mini Q4_K_M + 集显算子 offload | pp256 **约 1.6×**（76 vs 48 t/s） |
| Qwen3-0.6B 自动设备策略 | PPL 23.7397 vs 纯 CPU 23.7639，**约 1.9×** 加速 |
| 量化模型 QLoRA | SmallThinker MoE loss **1.93 → 1.44**；Qwen2.5-0.5B **2.55 → 1.99** |
| 32K 长上下文训练（8 GB 机器） | 私有内存稳定 **~3.5 GiB**，无磁盘交换，loss 逐位一致 |

> 通用优化数据来自 i5-12500H / Intel Iris Xe（Vulkan）测试机，Q1_0 数据来自 i5-10400；以上能力不依赖 Q1_0，Q1_0 专用内核仍然只在 Q1_0 + AVX2 路径生效。

---

## 快速开始

### 前置条件

| 项目 | 要求 |
|---|---|
| 系统 | Windows 10 / 11 x86-64 |
| CPU | 支持 SSE4.2；**推荐 AVX2**（Haswell 2013+ / Zen+）。支持 VNNI / AVX-512 更快 |
| 内存 | 8B 模型建议 ≥ 8 GB；4 GB 可跑 1.7B |
| 模型 | **Bonsai-8B GGUF（Q1_0）**，约 1.07 GiB，需自行从 [PrismML](https://prismml.com) 获取，仓库不附带；其他 Q4_K / Q8_0 GGUF 也可直接加载 |
| 运行时 | 推理**无需 Python / Node / CUDA**，预编译包已带全部 DLL；压缩脚本需要 Python 3 |

### 方式一：直接使用预编译产物（推荐）

`Lumina-Engine-Bin/` 已包含 14 个 CPU 变体、全部工具和运行库（54 个文件）。把模型放到仓库根目录，然后：

```bat
cd /d C:\Lumina-Engine\Lumina-Engine-Bin

REM 单轮推理
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf -p "The capital of France is" -n 32 -st -t 6
```

预期输出：

```text
The capital of France is Paris.
...
llama_perf_context_print:        eval time = ...
```

交互式对话（不带 `-p` 即进入多轮模式）：

```bat
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf -t 6
```

启动 OpenAI 兼容的 HTTP 服务：

```bat
Lumina-Engine-server.exe -m ..\Bonsai-8B.gguf -c 4096 -t 6
```

```bat
curl.exe http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"用一句话解释量子计算\"}]}"
```

> **线程数默认 = 物理核心数（不启用超线程）**。1-bit GEMV 在超线程下共享 L1 与内存带宽，通常更慢；只有显式传 `-t <逻辑核数>` 才会启用 SMT。

### 方式二：从源码构建

需要 CMake ≥ 3.14、Ninja、VS 2022 / 2026（MSVC）；Clang 20 可选（额外 5 个变体）。详见 [wiki/Building.md](wiki/Building.md)。

```bat
cd /d C:\Lumina-Engine

REM 1) 配置多变体构建树（首次需要）
configure-release.bat

REM 2) 编译 MSVC 树
build-release.bat

REM 3) 组装运行目录
bench\assemble_bin.bat
```

> 根目录的 `.bat` 按作者本机环境写死了路径（`D:\VS2026`、`D:\llvm`、`C:\Lumina-Engine`、可选 `D:\VulkanSDK\sdk`）。换机器或换目录时，请修改脚本开头的路径，或直接执行下方的 CMake 命令。

`build-bin.bat` 用于在 **已配置好** `build-release`（以及可选的 `build-clang`）后，一次完成编译与组装；若 `clang` 不在 `PATH` 中会自动跳过 Clang 构建，并从 `bench\clang_variants_backup\` 还原 5 个 Clang 专属变体。

只为自己这台机器编译（最快，单变体）：

```bat
cmake -S llama.cpp-b10991 -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
cmake --build build -j
```

产物在 `build\bin\`。

---

## 常用场景

> 以下命令默认在 `Lumina-Engine-Bin\` 目录中执行；第 1 步会先切换目录。

### 1. 基准测试

```bat
cd /d C:\Lumina-Engine\Lumina-Engine-Bin
Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 512 -n 128 -t 6 -r 3
```

### 2. 困惑度（验证数值无损）

```bat
Lumina-Engine-perplexity.exe -m ..\Bonsai-8B.gguf -f ..\wiki\Background.md -c 512 -t 6 --chunks 4
```

> `-f` 接受任意 UTF-8 文本文件，换成你自己的语料即可。

### 3. 和上游通用内核做 A/B

```bat
set GGML_NO_Q1_TILE=1
Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 512 -n 128 -t 6
set GGML_NO_Q1_TILE=
```

想单独对比查表路径：

```bat
set GGML_Q1_LUT=0   && REM 关闭 Q1LUT（省约 0.95 GiB 内存）
set GGML_Q1_LUT_PP=0 && REM 只关闭 pp 查表，保留 tg 查表
```

### 4. 无损压缩模型（可选，当前支持 Q4_K / Q6_K / F32 张量）

```bat
REM 回到仓库根目录（脚本在 tools\ 下）
cd /d C:\Lumina-Engine

REM 以 Q4_K_M 模型为例（该模型实测 94.28%）
python tools\lumina_gguf_pack.py SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf
REM -> SmallThinker-...gguf.lumina（原文件保留）

python tools\lumina_gguf_unpack.py verify SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf.lumina SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf

Lumina-Engine-Bin\Lumina-Engine-cli.exe -m SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf.lumina -p "Hello" -n 16 -st -t 6
```

引擎按魔数 `LGGC` 自动识别，`.gguf` 与 `.gguf.lumina` 都能直接加载。首次加载会解压到 `%TEMP%` 并复用。当前打包器支持 Q4_K / Q6_K / F32 张量；其他量化类型会明确报错，不会生成损坏的容器。详见 [wiki/ModelCompression.md](wiki/ModelCompression.md)。

### 5. QLoRA 微调量化模型

`Lumina-Engine-train` 无需反量化基座，直接训练低秩适配器，输出标准 LoRA GGUF（把 `data\...` 换成你的数据集路径）：

```bat
cd /d C:\Lumina-Engine\Lumina-Engine-Bin

Lumina-Engine-train.exe -m ..\Bonsai-8B.gguf ^
  --dataset data\train.jsonl --dataset-val data\val.jsonl ^
  -c 8192 -b 2048 -ub 128 --cpu-cores 0 ^
  -lr 5e-5 --epochs 2 -lora-rank 64 -lora-alpha 128 ^
  --save-steps 500 --keep-last 3 ^
  -o out\stage2.lora.gguf
```

```bat
REM 推理时挂载适配器（无需合并）
Lumina-Engine-cli.exe -m ..\Bonsai-8B.gguf --lora out\stage2.lora.gguf -p "你好" -n 128 -t 6
```

- 数据集为 JSONL，逐行自动识别 `{"text": ...}`、`{"messages": [...]}`、`{"conversations": [...]}`；默认只对 assistant 回复计算 loss。
- 支持**稠密与 MoE** 模型（含 3D 专家适配器、router）。
- 在已有适配器上继续训练用 `--init-lora`；断点续训用 `--resume`；检查点由 `--save-steps` / `--save-every-epochs` / `--keep-last` 管理。
- `--cpu-cores 0`（默认）使用**全部逻辑核**，比只用物理核约快 **1.3×**，loss 逐位一致。
- 长上下文训练在 8 GB 机器上也能跑 `-c 32768`（注意力反向现场重算 + 数据集 mmap）。

三阶段完整流程见 **[TRAINING_PIPELINE.md](TRAINING_PIPELINE.md)**，参数与实现见 **[wiki/Training.md](wiki/Training.md)**。

---

## 性能实测

> 以下为 **Q1_0 专用路径**的数据；Vulkan、多 CPU 变体、训练与压缩等通用优化的收益见[优化范围](#优化范围)。

**测试机**：Intel Core i5-10400（6 核 12 线程）· 8 GB 单通道 DDR4-2666（流式读 ~14–17 GB/s）· Windows 10 19045
**模型**：Bonsai-8B.gguf（Qwen3-8B 架构，全权重 Q1_0，1.07 GiB）· `-r 3`

### Prompt 处理（pp，6 线程）

| 测试 | 通用路径 | 优化后 | 提升 |
|------|---------:|-------:|-----:|
| pp32  | 13.18 | **18.78** | **+42.5%** |
| pp64  | 13.14 | **18.46** | **+40.5%** |
| pp128 | 13.11 | **17.94** | **+36.8%** |
| pp256 | 12.98 | **17.31** | **+33.4%** |
| pp512 | 12.83 | **16.49** | **+28.5%** |

> 通用路径 = `GGML_NO_Q1_TILE=1`（仍包含 `vpsignb` 与 VNNI 优化）。

### Token 生成（tg，12 线程，同场交替 A/B）

| 测试 | 旧版 | 新版（单列 GEMV） | 提升 |
|------|-----:|------------------:|-----:|
| tg32  | 11.27 | **12.67** | **+12.4%** |
| tg64  | 11.22 | **12.83** | **+14.3%** |
| tg128 | 11.17 | **12.69** | **+13.6%** |

### 二值查表 Q1LUT（本机 AVX2，无 VNNI，受控 A/B）

| 场景 | 打包内核 | Q1LUT | 提升 |
|------|---------:|------:|-----:|
| tg（`-p 64 -n 128 -t 6`） | 9.70 / 9.99 | **13.04 / 13.01** | **~+30%**（已到单通道带宽上限） |
| pp512 | ~15.8 | **~19.9** | **+26%** |
| pp1024 | 19.68 | **24.82** | **+26%** |
| pp2048 | 14.58 | **22.86** | **+57%** |

默认策略：**tg 走查表；pp 在 `ne11 >= 8` 走查表，`2..7` 走分块 GEMM**。代价是首次约 0.7 s 建表与约 0.95 GiB 额外内存。

### 正确性

| 检查 | 结果 |
|------|------|
| `test-backend-ops` Q1_0 MUL_MAT | **47 / 47 通过** |
| 困惑度（Bonsai 白皮书语料，4 chunks，c=512） | Q1LUT **5.7049** / 分块路径 **5.7173**（通用 5.7160） |
| 贪婪解码 | 与通用路径基本一致（浮点求和顺序不同，1e-6 量级） |
| 采样参数 | `--temp / --top-k / --top-p / --seed` 正常 |

推荐采样参数（与模型发布方一致）：`temperature 0.5–0.7`、`top-k 20–40`、`top-p 0.85–0.95`、`repetition penalty 1.0`。

完整数据、方法与复现命令见 [wiki/Benchmarks.md](wiki/Benchmarks.md)、[wiki/Correctness.md](wiki/Correctness.md)。

---

## 自动设备策略（CPU / GPU）

一套预编译版本，程序自动决定用 CPU 还是 GPU；显式 `-ngl` 始终优先。

| 机器 / 模型 | 自动策略 |
|---|---|
| 无 GPU | 纯 CPU |
| 集成显卡 + dense 模型 | 层留 CPU + Vulkan 算子 offload（pp 约 2.4×） |
| 集成显卡 + **MoE 模型** | 纯 CPU（MoE 走算子 offload 反而慢约 30%） |
| 独立显卡 | 按显存整层 offload |
| **Q1_0 + 支持 DP4A 的独显** | 自动 offload，并启用 Vulkan Q1_0 GEMV |

Vulkan 的 Q1_0 DP4A GEMV 相对原生 Vulkan GEMV 实测 tg **3.7×（Bonsai-1.7B）/ 7.1×（Bonsai-8B）**，作者在独显（Turing）上验证输出正确；但该内核在 Intel UMA 集显上端到端输出不正确，因此**集显默认关闭**。手动开关：`LUMINA_Q1_VK_GEMV=0/1`。

训练默认强制纯 CPU（Vulkan 后端没有反向算子）；`LUMINA_TRAIN_ALLOW_GPU=1` 仅为实验开关。

---

## 兼容性

`Lumina-Engine-Bin/` 内含 14 个 CPU 后端 DLL，运行时按 CPUID 自动选择：

```text
x64  sse42  sandybridge  ivybridge  piledriver  haswell  skylakex
cannonlake  cascadelake  icelake  cooperlake  alderlake  zen4  sapphirerapids
```

- AVX2 CPU 使用寄存器分块 GEMM / 单列 GEMV / Q1LUT。
- 支持 VNNI 的 CPU 额外使用 `vpdpbusd`。
- 更老的 CPU 自动回退到通用 Q1_0 内核（仍然正确）。
- 9 个常用变体由 MSVC 编译，5 个 Clang 专属变体由 Clang 20 编译，已验证可混用。

细节见 [wiki/Compatibility.md](wiki/Compatibility.md)。

---

## 从源码构建

### 多变体（分发用）

```bat
cmake -S llama.cpp-b10991 -B build-release -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ^
  -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON ^
  -DGGML_NATIVE=OFF -DGGML_RPC=ON ^
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build build-release -j
```

编译完成后用 `bench\assemble_bin.bat` 组装到 `Lumina-Engine-Bin\`（需要 `build-clang\bin` 作为基底；没有 Clang 构建时使用 MSVC 基底 + 备份变体）。

> 构建时若检测到 `D:\VulkanSDK\sdk` 会自动启用 Vulkan，未安装则自动回退 CPU-only。

构建树、产物布局与故障排查见 [wiki/Building.md](wiki/Building.md)、[wiki/Delivery.md](wiki/Delivery.md)。

---

## 环境变量

| 变量 | 默认 | 作用 |
|------|------|------|
| `GGML_NO_Q1_TILE=1` | 关 | 关闭分块 / GEMV / 查表路径，回退通用 Q1_0 内核（A/B 或排障） |
| `GGML_NO_Q1_GEMV=1` | 关 | 只关闭 tg 的单列 GEMV |
| `GGML_Q1_GEMV_ROWS=1\|2\|4` | `1` | 单列 GEMV 行分组大小 |
| `GGML_Q1_LUT=0` | 开 | 关闭整个二值查表路径，可省约 0.95 GiB 内存 |
| `GGML_Q1_LUT_PP=0` | 开 | 只关闭 pp 查表（pp 回退到 2×4 分块 GEMM） |
| `GGML_Q1_LUT_KB=N` | `0` | 查表 K 分块（0 = 整表，实测最快；>0 为实验性） |
| `GGML_Q1_LUT_DEBUG=1` | 关 | 打印查表缓存的分配 / 填充信息 |
| `LUMINA_Q1_VK_GEMV=0\|1` | auto | Vulkan Q1_0 DP4A GEMV：强制关 / 开（默认独显开、集显关） |
| `LUMINA_Q1_VK_CACHE_MB=N` | `2048` | Vulkan Q1_0 权重重排缓存上限（MB，`0` = 不限） |
| `LUMINA_Q1_VK_INT8=1` | 关 | 实验性 int8 prefill 路径 |
| `LUMINA_TRAIN_ALLOW_GPU=1` | 关 | 允许训练使用 GPU（实验性，不推荐） |
| `LUMINA_OP_PROF=1` | 关 | 打印训练算子耗时剖析 |

---

## 架构与目录

```text
Bonsai-8B.gguf / .gguf.lumina
        │
        ▼
┌───────────────────────────┐
│ 模型加载（透明解压 LGGC）  │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ 后端选择（CPUID / Vulkan） │
└───────────────────────────┘
        │
        ├── Q1_0 + AVX2 ──► Q1LUT 查表 / 2×4 分块 GEMM / 单列 GEMV
        └── 其它模型/量化 ─► 上游 llama.cpp 通用内核
        │
        ▼
 cli / server / bench / perplexity / train / quantize ...
```

```text
Lumina-Engine/
├─ llama.cpp-b10991/          # 修改后的源码
│  ├─ ggml/src/ggml-cpu/q1t.cpp      # Q1_0 分块 GEMM + 单列 GEMV + Q1LUT
│  ├─ src/lumina-gguf.cpp            # .gguf.lumina 无损容器解码器
│  └─ tools/lumina-train/            # C++ QLoRA 训练引擎
├─ Lumina-Engine-Bin/         # ★ 预编译交付产物（54 文件 / 14 变体）
├─ bench/                     # 内核微基准 + 构建 / 组装脚本
├─ tools/                     # GGUF 压缩、解包、查看脚本
├─ wiki/                      # 中英文详细文档
├─ TRAINING_PIPELINE.md       # 三段式训练流程
├─ OPTIMIZATION_LOG.md        # 逐版本优化记录与 A/B 数据
├─ HANDOFF.md                 # 开发交接文档
├─ README.en.md               # 英文 README
└─ CONTRIBUTORS.md            # 贡献者名单与上游致谢
```

---

## FAQ

**Q：一定要有 GPU 吗？**
不需要。默认就是纯 CPU 路径；Vulkan 只是可选加速，没有 Vulkan 设备会自动回退。

**Q：支持 NVIDIA CUDA 吗？**
当前预编译产物为 **CPU + Vulkan**。需要 CUDA 请基于上游 llama.cpp 自行构建；Q1_0 专用 CPU 内核是 x86 路径，另有 Vulkan Q1_0 路径。

**Q：优化会改变模型输出吗？**
不会。权重逐字节不变，Q1_0 位运算精确；只有浮点累加顺序不同，差异在 1e-6 量级，困惑度同量级。

**Q：为什么我的 tg 只有十几 token/s？**
8B 模型每生成一个 token 都要读约 1.07 GiB 权重，速度由内存带宽决定。单通道 DDR4 的理论上限约 15.7 tok/s；换双通道或更高频内存提升最明显。

**Q：老 CPU 能跑吗？**
能。最低 SSE4.2，自动回退到通用内核；但 AVX2 才能用到分块 GEMM / Q1LUT 的完整收益。

**Q：模型在哪里下载？**
仓库不附带模型。请从 [PrismML](https://prismml.com) 获取 Bonsai 系列 GGUF（Q1_0），放到仓库根目录即可。

**Q：支持 Linux 吗？**
源码本身跨平台，但本仓库的预编译交付与性能测试集中在 Windows x86-64；Linux 需自行用 CMake 构建。

**Q：能微调 1-bit 模型吗？**
可以。基座权重保持量化并冻结，只训练 LoRA 适配器（QLoRA），支持 Q1_0 / Q4_K / Q8_0 和 MoE。

**Q：这些优化只对 1-bit 模型有效吗？**
不是。Q1_0 专用内核只影响 1-bit 模型；14 个 CPU 变体、Vulkan、自动设备策略、无损压缩和 QLoRA 训练都适用于其他模型 / 量化类型。详见[优化范围](#优化范围)。

---

## 更新日志

| Build | 主要内容 |
|------:|----------|
| **24** | 所有控制台工具统一启动画面（ASCII logo + Build 号 + 项目链接） |
| 23 | 训练默认用满全部逻辑核，约 **1.3×** 提速，loss 逐位一致 |
| 22 | 修复 `--resume` 后 ETA 持续上涨；新增 `--dataset-stride`，epoch 时间可再减半 |
| 21 | 修复 `--resume` 从第 1 步重训；从检查点所在数据点 / 微批继续 |
| 20 | 长上下文训练 RAM 优化：`-c 32768` 在 8 GB 机器上稳定运行、无磁盘交换 |
| 19 | Vulkan Q1_0 DP4A GEMV 移植 + 自动后端 / 模型选择 |
| 12–16 | Vulkan 后端、机器感知设备策略、训练 acc 修复、统一横幅 |
| 6–8 | `.gguf.lumina` 无损压缩容器（v1 → v2，94.28%） |

更完整的逐版本说明见 [HANDOFF.md](HANDOFF.md) 与 [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)。

---

## 文档

| 文档 | 内容 |
|------|------|
| [wiki/Home.md](wiki/Home.md) | 文档总览与一页速览 |
| [wiki/Background.md](wiki/Background.md) | Bonsai / Q1_0 / GGUF 背景与瓶颈分析 |
| [wiki/Optimizations.md](wiki/Optimizations.md) | 各优化的原理与实现（含 Q1LUT） |
| [wiki/Building.md](wiki/Building.md) | 单变体 / 多变体 / 双工具链编译 |
| [wiki/Delivery.md](wiki/Delivery.md) | 交付目录、组装流程与约定 |
| [wiki/Benchmarks.md](wiki/Benchmarks.md) | 完整测试数据与方法 |
| [wiki/Correctness.md](wiki/Correctness.md) | 数值等价性与后端测试 |
| [wiki/Compatibility.md](wiki/Compatibility.md) | 14 个 CPU 变体与分发 |
| [wiki/ModelCompression.md](wiki/ModelCompression.md) | 压缩模型 / 使用压缩模型（用户指南） |
| [wiki/LosslessCompression.md](wiki/LosslessCompression.md) | 无损容器格式与原理 |
| [wiki/Training.md](wiki/Training.md) | QLoRA 训练引擎参数与实现 |
| [wiki/Changes.md](wiki/Changes.md) | 逐文件代码更改 |
| [wiki/Troubleshooting.md](wiki/Troubleshooting.md) | 常见问题与排障 |

每个页面均提供英文版（`wiki/*.en.md`）。完整的英文项目说明见 [README.en.md](README.en.md)。

---

## 支持这个项目

Lumina-Engine 的每一条性能数据，都来自真实硬件上反复交替的 A/B 测量；每一次“默认启用”的背后，都是正确性、困惑度和兼容性回归。项目由维护者持续投入时间开发：没有广告，核心功能全部开源。一次完整的 14 变体构建与回归要花数小时，一个新的 CPU / GPU 平台适配往往需要几十组 A/B。

如果它帮你省下了时间，欢迎支持它继续变快：

### [❤️ 在爱发电支持 Lumina-Engine →](https://afdian.com/a/lumi_380)

你的支持会被优先用于：

- **更多硬件覆盖**：AVX-512 / VNNI / AMD / NVIDIA 等平台的测试与调优
- **更长的测试**：不只跑一次，而是跑够轮次、跑完 32K 长上下文与整模型回归
- **更快的上游跟进**：llama.cpp 更新、模型格式变化、issue 复现
- **更好的文档**：基准数据、构建脚本与排障记录

不花钱也可以帮忙：

- ⭐ Star 仓库，让更多人发现它
- 📊 分享你的硬件 benchmark 或复现结果
- 🐛 提交高质量 issue / PR
- 🔔 关注 [Bilibili](https://space.bilibili.com/3546376302496335) 获取更新

> 赞助完全自愿。无论是否支持，Lumina-Engine 都保持 AGPL-3.0 开源。

---

## 贡献

欢迎 Issue 和 PR。这个项目的核心原则是：**先正确，再快**；任何优化都必须能通过 A/B 与困惑度验证，并且不能影响非 Q1_0 路径。贡献者名单与致谢见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。

```bat
git clone https://github.com/Happy-380/Lumina-Engine.git
cd Lumina-Engine

REM 配置 + 构建 + 组装（脚本中的工具链与仓库路径按你的环境修改）
configure-release.bat
build-release.bat
bench\assemble_bin.bat

REM 基本验证
Lumina-Engine-Bin\Lumina-Engine-cli.exe -m Bonsai-8B.gguf -p "The capital of France is" -n 8 -st
Lumina-Engine-Bin\Lumina-Engine-perplexity.exe -m Bonsai-8B.gguf -f wiki\Background.md -c 512 -t 6 --chunks 4
```

`test-backend-ops` 需要构建时加 `-DLLAMA_BUILD_TESTS=ON`（默认交付脚本关闭测试）。

提交前请尽量附上：

- 修改前后的**同场 A/B** 数据（`GGML_NO_Q1_TILE=1` 作为基线），并注明 CPU、内存、线程数、模型。
- 正确性证据：`test-backend-ops`（Q1_0 MUL_MAT）、困惑度、固定 seed 的贪婪输出对比。
- 若改动涉及编译配置，请说明使用的 MSVC / Clang / CMake 版本。

内核微基准在 `bench/` 下（如 `bench_lut2.cpp`、`bench_gemv128.cpp`）；构建与组装脚本也在该目录。

---

## 许可证与致谢

- 本仓库的修改与新增代码以 **GNU AGPL-3.0** 发布，见 [LICENSE](LICENSE)。
- 上游 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 采用 **MIT License**；上游文件保留其原始版权与许可声明。
- Q1_0 量化格式与 Bonsai 模型来自 [PrismML](https://prismml.com)。
- 内置 DEFLATE 解码使用 [miniz](https://github.com/richgel999/miniz)（MIT / 公有领域）。
- 完整贡献者名单与上游致谢见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。

> 如果你的服务通过网络向用户提供本项目的修改版本，AGPL-3.0 要求你向用户提供对应源码。
