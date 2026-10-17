# Wiki 总览

> 🌐 **中文** | [English](Home.en.md)

本 wiki 详细介绍本仓库对 **Bonsai 1-bit（Q1_0）模型在 CPU 上的推理优化**：
原理、实现、编译、交付、测试与排障。

> **当前构建：Build 24（Based on llama.cpp b-10991）。** 所有控制台工具（`cli` /
> `server` / `train` / `bench` / `perplexity` / `quantize` / `imatrix` / `tokenize` /
> `gguf-split` / `rpc-server`）启动时都会打印统一的 Lumina-Engine 启动画面
> （ASCII logo + Build 号 + GitHub / 爱发电 / Bilibili 链接），Build 号由
> `LLAMA_BUILD_NUMBER` 自动生成，与 `--version` 一致。

## 页面

| 页面 | 内容 |
|------|------|
| [Home](Home.md) | 本页，总览与快速导航 |
| [Background](Background.md) | Bonsai 模型、Q1_0 量化格式、GGUF、1-bit 推理的瓶颈 |
| [Optimizations](Optimizations.md) | 各核心优化的原理与实现细节（含二值查表 Q1LUT） |
| [Building](Building.md) | 单变体 / 多变体 / 双工具链编译，产物布局 |
| [Delivery](Delivery.md) | **交付目录 `Lumina-Engine-Bin` 与生成配置（一键脚本、组装流程、约定）** |
| [Benchmarks](Benchmarks.md) | 测试方法、原始数据、A/B 对比、微基准 |
| [Correctness](Correctness.md) | 数值等价性、困惑度、采样、后端测试 |
| [Compatibility](Compatibility.md) | 14 个 CPU 变体、选择逻辑、分发说明 |
| [Changes](Changes.md) | 逐文件的主要代码更改 |
| [ModelCompression](ModelCompression.md) | **用户指南：如何压缩模型、如何加载压缩模型** |
| [LosslessCompression](LosslessCompression.md) | **模型无损压缩容器（Build 8）：字段拆分 + 上下文排序 + DEFLATE，`.gguf.lumina`** |
| [Training](Training.md) | **训练引擎：用 C++ 对量化模型 / `.gguf.lumina` 做 QLoRA 微调** |
| [Troubleshooting](Troubleshooting.md) | 常见问题与排障 |

## 一页速览

### 问题
Bonsai-8B 的权重全部是 1-bit（`GGML_TYPE_Q1_0`，128 权重共享一个 fp16 尺度，1.125 bit/权重）。
CPU 上：
- **Token 生成（tg）** 每 token 要读完 ~1.06 GB 权重 → 内存带宽瓶颈。
- **Prompt 处理（pp）** 权重可复用 → 计算瓶颈，落在 `Q1_0 × Q8_0` 点积内核上。

### 方案
1. **激活 per-128 量化（Q8_128）**：每 128 权重只保留 1 个激活尺度，使一个 Q1_0 块内 4 个子块的 int32 点积可先累加，浮点转换次数 4 → 1。
2. **寄存器分块（2 行 × 4 列）**：符号展开一次，4 列复用，无额外内存流量。
3. **VNNI**：`vpdpbusd` 合并 `vpmaddubsw + vpmaddwd`。
4. **`vpsignb`**：单条指令实现“+1 保留 / −1 取反”。
5. **单列 GEMV（tg）**：复用 per-128 激活，每 128 权重一次浮点转换，激活量化向量化并跨线程分块。
6. **二值查表（Q1LUT，默认启用）**：为二值权重实现激活侧查表 GEMV；tg 与 `ne11 >= 8` 的 pp 走查表。
7. **多微架构构建**：同一份源码编译出 14 个 CPU 后端，运行时按 CPUID 自动选择。

### 结果（本机 i5-10400，6 线程）

> ⚠️ **测试机为单通道内存**（8 GB DDR4-2666，实测流式读带宽约 14–17 GB/s）：
> tg（token 生成）受带宽上限压制（理论上限 ~15.7 tok/s），因此 tg 绝对值偏低；
> 双通道/更高带宽平台上 tg 提升会更接近内核层面的 ~30%。请以同场 A/B 的**比值**而非绝对值为准。

- **pp +29% ~ +43%**（相对通用路径；Q1LUT 在 `ne11 >= 8` 时再快约 +26% ~ +57%）
- **tg 约 +12% ~ +14%**（单列 GEMV，12 线程）；**Q1LUT 默认启用后再约 +30%**（~9.8 → ~13.0 t/s）
- 困惑度：优化后 5.7173；Q1LUT 5.7049（均与通用 5.7160 同量级，无精度损失）
- 47/47 后端测试通过

### 快速命令
```bat
REM 交付产物在仓库根目录的 Lumina-Engine-Bin\（唯一的交付目录，见 Delivery.md）
cd C:\Lumina-Engine\Lumina-Engine-Bin
Lumina-Engine-bench.exe -m ..\Bonsai-8B.gguf -p 512 -n 128 -t 6

set GGML_NO_Q1_TILE=1   & REM 回退到通用内核做 A/B (pp+tg)
set GGML_NO_Q1_GEMV=1   & REM 只回退 tg 的 GEMV
set GGML_Q1_GEMV_ROWS=2 & REM 调整 tg 行分组（1/2/4）
set GGML_Q1_LUT=0       & REM 关闭二值查表（默认启用）
set GGML_Q1_LUT_PP=0    & REM 仅关闭 pp 查表
set GGML_Q1_LUT_DEBUG=1 & REM 打印查表缓存分配
```
