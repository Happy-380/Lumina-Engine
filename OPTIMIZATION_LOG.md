# Lumina-Engine 性能优化与 A/B 测试记录

> 环境：Windows 11 x64 / i5-12500H（4P+8E，12 物理核 / 16 线程）/ 16 GB DDR4 / Intel Iris Xe (Vulkan 1.4)
> 构建：MSVC 19.41（Build 12 起启用 Vulkan 1.4.363 后端）
> 测试模型：`D:\Qwen3-0.6B-Q8_0.gguf`、`D:\phi-3-mini-128k-instruct.q4_k_m.gguf`、`D:\SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf`
> 说明：本机 CPU 状态在长会话中有明显漂移（同一配置可差 20%+），因此下表以「同批次相邻测量」为准，绝对值仅供参考。

---

## Build 24：所有控制台统一启动画面

**目标**：所有控制台工具（`cli` / `server` / `train` / `bench` / `perplexity` /
`quantize` / `imatrix` / `tokenize` / `gguf-split` / `rpc-server` 等）启动时显示同一套
Lumina-Engine 启动画面（ASCII logo + Build 号 + 项目链接），并把 CLI 的 Build 号从 23 升到 24。

**改动**：
- `CMakeLists.txt`：`LLAMA_BUILD_NUMBER = 24`（`llama_build_info()` 由此生成，
  启动画面与 `--version` 永远一致）。
- `common/build-info.cpp.in` / `common/build-info.h`：新增 `lumina_ascii_logo()`，
  `llama_print_lumina_banner()` 输出完整启动画面。
- `common/arg.cpp`：`common_params_parse()` 成功后统一打印启动画面（`cli` 除外，
  由 CLI 自身打印带模型信息与命令列表的横幅），所有走公共参数解析的工具自动生效。
- 删除 `tools/server/server.cpp`、`tools/cvector-generator/cvector-generator.cpp` 中
  重复的横幅打印；`tools/cli/cli-context.cpp` 改用共享 `lumina_ascii_logo()`。
- `tools/gguf-split/gguf-split.cpp`、`tools/rpc/rpc-server.cpp` 显式打印同一画面。

**输出**：

```
   __                 _
  / / _   _ _ __ ___ (_)_ __   __ _
 / / | | | | '_ ` _ \| | '_ \ / _` |
/ /__| |_| | | | | | | | | | | (_| |
\____/\__,_|_| |_| |_|_|_| |_|\__,_|  Engine

build      : Build 24 (Based on llama.cpp b-10991)
GitHub     : https://github.com/Happy-380/Lumina-Engine
Donate     : https://afdian.com/a/lumi_380  (tip the developer / support development)
Bilibili   : https://space.bilibili.com/3546376302496335
```

**交付**：MSVC + Clang 双树已重建，`bench\assemble_bin.bat` 组装 54 文件到
`Lumina-Engine-Bin`；`--version` 与所有控制台启动画面均显示 `Build 24`。

---

## Build 23：训练默认用满逻辑核（`--cpu-cores 0`，约 1.3×，逐位无损）

**目标**：固定命令
`-c 32768 -b 2048 -ub 128 --cpu-cores 0 ...`（6 核 / 12 线程 i5-10400，8 GB）
在**不改变命令、不增加 RAM、不改训练结果**的前提下提速。

**剖析**（内置 `LUMINA_OP_PROF=1`，`ggml/src/ggml-cpu/ggml-cpu.c` 的
`ggml_prof_dump`，只在设置环境变量时生效）：

| 上下文 | 主导算子（占累计线程时间） |
|---|---|
| `-c 4096`（1 数据点，32 微批） | `MUL_MAT_ID_BACK_B` 31% · `OUT_PROD` 30% · `MUL_MAT` 21% |
| `-c 8192`（1 数据点，64 微批） | `OUT_PROD` 29% · `MUL_MAT` 28% · `MUL_MAT_ID_BACK_B` 24% |
| `-c 16384`（数据点尾段，`n_kv≈15k`） | `MUL_MAT` 42% · `OUT_PROD` 29% · `SOFT_MAX(+BACK)` 11% · `MUL_MAT_ID_BACK_B` 9% |

长上下文下注意力主导；所有主导算子都是**内存带宽**受限的 f32 向量乘加 / 反量化。

**根因**：`--cpu-cores 0` 时训练线程数取引擎默认的**物理核数**（`common_cpu_get_num_math()`
= 6），6 个 SMT 逻辑线程闲置。验证：`-c 4096` 同一数据点，6 线程 258 s，12 线程 202 s。

**改动**（`tools/lumina-train/lumina-train.cpp`）：`--cpu-cores 0`（默认）使用
`std::thread::hardware_concurrency()`（全部逻辑核）；`--cpu-cores N>0` 不变。
命令行无需改动。**逐位无损**：每个内核仍把整行 / 整列一次性分配给单一线程，
逐元素累加顺序与线程数无关；实测两者前 5 个微批 loss 完全一致
（2.58786 / 2.36802 / 2.46651 / 2.57018 / 2.62832）。
`CMakeLists.txt`：`LLAMA_BUILD_NUMBER = 23`。

**未启用的更大提速（改变训练语义）**：默认 stride = `n_ctx/2`（50% 重叠，每 token 训练
两次）。命令加 `--dataset-stride 32768` 可去掉重叠，epoch 时间约减半；因会改变每个 epoch
每个 token 的训练次数，未设为默认。

**交付**：MSVC 全量重建并组装到 `Lumina-Engine-Bin`，`--version` = `build 23`。

---

## Build 22：训练进度 ETA 误报修复 + 数据集窗口步长可选（约 2× 提速）

**现象**：长上下文训练（`-c 32768 -b 2048 -ub 128 --resume`）时，刚启动显示 ETA ~400h，
随后一路涨到 ~10000h 且“越来越慢”。用户提供的 21b 日志（`D:\Dataset\output\train-monitor21b.log`）
中 ETA 从 1479h 单调涨到 13010h。

**根因**：实际吞吐是**恒定**的，是 ETA 估计器本身有偏。
逐微批统计日志可证明：相邻两个数据点周期（各 256 微批）耗时 7635s 与 7624s，
即 29.8 s/微批，全程不变。ETA 增长来自两个叠加因素：

1. **`--resume` 后累计均值起点被拉低。** 进度条用
   `ETA = t_ibatch * (ibatch_max - ibatch) / ibatch`，其中 `ibatch` 是**全局**微批计数
   （从检查点续接，日志中从 651 起），而 `t_ibatch` 只统计**本次进程**的耗时（从 0 起）。
   于是 `t_ibatch/ibatch` 从远低于真实值的 2 s/微批开始，随时间缓慢收敛到真实的 ~30 s/微批，
   ETA 也就“一直增加”。
2. **一个数据点内部注意力开销随上下文位置增长（O(context)）。** 同一个 32k token 文档内，
   微批耗时从开头 ~7s 线性升到末尾 ~70s；累计均值在第一个数据点结束前严重偏低。

**改动**：
- `ggml/src/ggml-opt.cpp`：`ggml_opt_epoch_callback_progress_bar()` 的 ETA 改为
  **最近一个完整数据点的滑动窗口吞吐**（窗口长度 = `n_ctx / n_ubatch`，下限 16、上限 4096），
  首窗未填满时回退到累计均值。窗口跨数据点边界时正好是一个完整循环，估值稳定且与
  `--resume` 起点无关。（顺带修正：`ggml_opt_inputs()` 是一维 token 张量，`ne[1]==1`，
  微批大小应取 labels 张量的 `ne[1]`。）
- `common/common.h` / `common/arg.cpp`：新增 `--dataset-stride N`。默认 `0` 保持
  现有 `stride = n_ctx/2`（窗口 50% 重叠）语义；设为 `n_ctx` 时每个 token 只训练一次，
  epoch 时间约减半。大于 `n_ctx` 截断为 `n_ctx`（不跳过 token）。
- `common/build-info.cpp.in`：banner 的 Build 号改为由 `LLAMA_BUILD_NUMBER` 动态生成，
  避免与 `--version` 漂移。
- `CMakeLists.txt`：`LLAMA_BUILD_NUMBER = 22`。

**验证**（`-c 512 -b 256 -ub 64` 小上下文冒烟，SmallThinker Q4_K_M）：
- 普通训练：ETA 平滑从 29:10 降到 26:19，不再随累计均值漂移。
- `--resume`（从 `ckpt-32` 恢复，`global step 32`）：第一帧回退值 01:06，
  第 2 帧即跳到正确的 ~32:54，随后稳定在 ~33:30 并缓慢下降；
  旧实现此阶段会显示极低的 ETA 并持续上涨。
- 两种构建均通过：MSVC（`build-release`）与 clang（`build-clang`），
  已组装 54 文件 / 14 个 CPU 变体到 `Lumina-Engine-Bin`，`--version` 显示 `build 22`。

**交付**：Build 22，全部二进制已重建并组装到 `Lumina-Engine-Bin`。

---

## Build 21：修复 `--resume` 从第 1 步重新训练的问题

**问题**：从检查点（如 `stage1.lora.ckpt-600.gguf`）用 `--resume` 继续训练时，进度显示 `data=1`，
整一个 epoch 的数据从头重训（检查点编号虽然继续，但数据没跳过）。

**根因**：`llama_opt_epoch()` 的训练循环固定从 `idata = 0` 开始；`--resume` 只恢复了
`cumulative_steps` 与 `start_epoch`，没有任何“从当前 epoch 的第几个数据点继续”的信息。
另外 `--save-steps` 的帮助/文档写作“优化步”，实际实现数的是**训练微批（ubatch step）**。

**改动**：
- `include/llama.h` / `src/llama-context.{h,cpp}`：`llama_opt_epoch()` 新增 `idata_start` 与
  `ubatch_start`；训练循环从该位置开始，进度 `data=` 也随之续接。
- 恢复位置的那个数据点，其**已训练前缀**仍做一次**前向**以重建 KV cache，但
  `backward=false`、`result=nullptr`，并用 `ggml_opt_reset_opt_i()` 保证不打乱梯度累积相位。
  → 前缀不产生任何梯度、不更新任何权重（真正无损续训；不是靠“置掩码”，因为交叉熵反向对
  全零标签行并不为零，实测会继续更新权重）。
- `tools/lumina-train/lumina-train.cpp`：把检查点的微批步数换算回
  `(数据点, 微批)`：`idata_start = steps/ubatch_per_ctx`、`ubatch_start = steps%ubatch_per_ctx`，
  并保持 `cumulative_steps` 与实际位置一致（多次续训也正确）。
- `ggml/include/ggml-opt.h` / `ggml/src/ggml-opt.cpp`：新增 `ggml_opt_reset_opt_i()`。
- `common/arg.cpp` / `TRAINING_PIPELINE.md`：`--save-steps` 文案由“优化步”更正为“训练微批
  （一个优化步 = n_batch/n_ubatch 个微批）”，与实际计数一致。

**验证**（`-c 2048 -b 2048 -ub 128`，从 `ckpt-600` 恢复）：
- 日志：`resuming at datapoint 37/1945 of epoch 1, micro-batch 8/16 (global step 600)`；
- 进度从 `data=601` 开始（原来为 `data=1`），并在第 616 个微批正常完成首次优化步；无错误。

**交付**：Build 21，MSVC + Clang 双树重建并组装到 `Lumina-Engine-Bin`（55 文件 / 14 个 CPU 变体）。

---

## Build 20：长上下文训练 RAM 优化（消除磁盘交换）

**问题**：在 8 GB 机器上用

```
-c 32768 -b 2048 -ub 128 -lora-rank 64 -lora-alpha 128 --epochs 1 --resume
```

训练 `SmallThinker-4BA0.6B-Instruct.Q4_K_M`（MoE，32 层，n_head=12，n_head_kv=2，head_dim=128）时，
训练开始后内存持续增长并进入磁盘交换，训练极慢甚至无法继续。

**三个根因**（按影响排序）：

1. **注意力反向的激活内存是 O(n_layer × n_ctx)**。非 FlashAttention 路径把每层的
   `softmax(QKᵀ)` 激活 `[n_ctx, n_ubatch, n_head]` 保存下来供反向使用；线性图在反向时必须
   同时保活全部 32 层的该激活。`n_ctx=32768` 时这一项就是
   `32 × 32768 × 128 × 12 × 4 ≈ 6.0 GiB`（实测 compute buffer 每微批增长 ~28 MiB）。
2. **`ggml_opt_result` 累积逐 token 预测**。`ggml_opt_eval()` 每次评估都把 `[n_ubatch]` 个 argmax
   追加进 `result->pred`；一个 epoch（stage1 约 10163 个数据点 × 256 微批）会累积约
   **1.3 GiB**，且只在 epoch 结束时才清空。
3. **数据集 token 流是匿名内存**。stage1 的 ~166.5 M token 以 `std::vector<llama_token>` 常驻
   （~635 MiB，容量峰值更大），内存紧张时只能被换出到 pagefile。

**改动**：

1. **注意力反向梯度重算（内存高效注意力）**。
   - `src/llama-graph.cpp::build_attn_mha()`：训练时在注意力输出 matmul 上打标记
     `kqv->src[2]=k, src[3]=q, src[4]=mask`（简单情形：无 kq_b / softcap / Grok / sinks）。
   - `ggml/src/ggml.c::ggml_compute_backward()`：识别该标记后**现场重算** `kq = k·qᵀ`、
     `softmax`，再按与标准路径完全相同的公式求 `grad_v/grad_k/grad_q`。前向的
     `kq`/`softmax` 不再被反向引用，缓冲区按层复用。激活内存从
     O(n_layer × n_ctx) 降为 O(n_ctx)（仅一层）。数值**逐位一致**（同样的算子、同样的输入）。
   - 关闭 `llama_context::opt_init()` 中「关闭注意力标记」的旧限制；训练仍禁用 FlashAttention。
2. **`collect_pred` 开关**（`ggml/include/ggml-opt.h`、`ggml/src/ggml-opt.cpp`）。
   新增 `ggml_opt_params::collect_pred`；`llama_context::opt_init()` 训练时置 `false`，
   预测向量不再累积。loss / accuracy 不受影响（accuracy 用 `n_count_equal` 标量）。
3. **数据集 token 流内存映射**（`tools/lumina-train/lumina-train.cpp`）。加载完成后把
   token（及可选掩码）写入系统临时文件并 `mmap` 只读映射，再释放 `std::vector`。
   页变为 file-backed 干净页，内存压力下可被系统直接回收（回读 page cache），不再逼迫换页。
4. 训练开始打印真实内存占用（model / KV cache / compute buffers），便于复核。

**验证**（本机 8 GB，`-c 32768 -b 2048 -ub 128`）：

| 指标 | Build 19（旧） | Build 20 |
|---|---:|---:|
| private working set（stage1 全量，稳定值） | 持续增长 → 5.3 GiB+ | **稳定 ~3.5 GiB** |
| compute buffer（n_ctx=32768 峰值，外推） | ~7.9 GiB | **~1.7 GiB** |
| 剩余物理内存 | 466 MB（持续下降） | **>1.6 GB（稳定）** |
| 训练 loss 序列 | 参考 | **逐位一致**（18/19/20 = 2.72211 / 2.70184 / 2.69565） |
| 磁盘交换 | 发生 | **无** |

**交付**：Build 20，MSVC 全量重建并组装到 `Lumina-Engine-Bin`（55 文件 / 14 个 CPU 变体）。
Vulkan 后端 DLL 沿用 Build 19 产物（本机 Vulkan SDK 已卸载；后端 ABI 未变，训练为纯 CPU）。

---

## Build 19：Vulkan Q1_0（1-bit）DP4A GEMV 移植（Bonsai 模型）

将用户提供的 `1.patch`（CPU q1t 的 1-bit 逻辑移植到 Vulkan）rebase 到 Build 18 源码树，
修复其集成缺陷，并接入「自动后端/模型选择」。实测（Bonsai Q1_0，`-ngl 99`，`-t 6`）：

| 模型 | 原生 Vulkan GEMV tg64 | DP4A GEMV tg64 | 提升 |
|---|---:|---:|---:|
| Bonsai-1.7B | 40.8 t/s | **152.4 t/s** | **3.7×** |
| Bonsai-8B | 11.7 t/s | **83.7 t/s** | **7.1×** |

prefill（pp256）不受影响。**但本机 Intel Iris Xe（UMA）上 DP4A GEMV 端到端输出错误**
（重复 token），隔离的 `test-backend-ops` 用例通过、`GGML_VULKAN_CHECK_RESULTS` 每算子误差约 1%，
说明是 UMA/驱动特有的集成问题（作者在独显 Turing 上验证正常）。因此：

- Q1_0 DP4A GEMV 默认 **auto**：**独显启用、集显(UMA)禁用**（`LUMINA_Q1_VK_GEMV=0/1` 可强制）；
- `common` 设备策略据此选择：Q1_0 + 具备 DP4A 内核的 GPU → offload；否则集显保持层在 CPU；
- 本机自动策略下 Bonsai Q1_0 走 CPU（tg ≈ 90 t/s，正确；比原生 Vulkan 的 41 t/s 更快）。

正确性（贪心，Bonsai-1.7B）：CPU / Vulkan 原生 / 无 DP4A 回退 均为 `The capital of France is **Paris**...`。

---

## Build 11（基线，CPU-only）

`-t 8`：

| 模型 | pp | tg |
|---|---:|---:|
| Qwen3-0.6B-Q8_0 | 207.18 t/s（pp512） | 61.60 t/s（tg128） |
| phi-3-mini Q4_K_M | 48.31 t/s（pp256） | 17.71 t/s（tg64） |
| SmallThinker MoE Q4_K_M | 177.09 t/s（pp128） | 62.77 t/s（tg64） |

默认设备选择：无 GPU 后端，等价 `-ngl 0`（纯 CPU）。

---

## Build 12：启用 Vulkan 后端 + 默认改为「层留 CPU + 算子 offload」

### 改动

1. `configure-release.bat`：检测 `D:\VulkanSDK\sdk` 并启用 `-DGGML_VULKAN=ON`（无 SDK 时自动回退为 CPU-only）。
2. `common/common.h`：`n_gpu_layers` 默认值 `-1`（auto，全量 offload）→ `0`（不在 GPU 存放整层，但允许算子 offload）。
3. `common/arg.cpp`：移除旧的 `GGML_ASSERT(n_gpu_layers < 0)` 并修正 `-ngl` 帮助文案。
4. 交付新增 `ggml-vulkan.dll`。

### 动机（实测）

同一 Qwen3-0.6B，`completion` 默认路径：

| `-ngl` | 说明 | prompt eval | eval(tg) |
|---|---|---:|---:|
| `auto`（旧默认） | 整层 offload 到 iGPU | 41.4 t/s | **40.3 t/s** |
| `0`（新默认） | 层留 CPU + 算子 offload | 50.0 t/s | **53.5 t/s** |
| `99` | 全量 offload | 46.7 t/s | 41.2 t/s |

→ 旧默认 `auto` 在集成显卡上会把生成速度拉低 ~25%；新默认 `0` 同时拿到更快的 pp 与最快的 tg。

### A/B（同批次，`-t 8`）

| 模型 | 配置 | pp | tg | 对比 Build 11 |
|---|---|---:|---:|---|
| Qwen3-0.6B-Q8_0 | `-ngl 0`（op-offload） | 420.92（pp512） | 54.29（tg128） | pp **≈2.4×**，tg 持平 |
| Qwen3-0.6B-Q8_0 | `-ngl 99`（全量 Vulkan） | 671.82（pp128） | 41.48（tg32） | pp 更快，tg 变慢 |
| phi-3-mini Q4_K_M | `-ngl 0` | 76.02（pp256） | 16.40（tg64） | pp **≈1.6×**，tg 略降 |
| phi-3-mini Q4_K_M | `-ngl 99` | 150.65（pp256） | 13.54（tg64） | pp 更快，tg 变慢 |
| SmallThinker MoE | `-ngl 0` | 135.91（pp128） | 53.04（tg64） | 噪声范围内（MoE 收益不明显） |
| SmallThinker MoE | `-ngl 99` | 172.77（pp128） | 40.16（tg64） | pp 持平，tg 变慢 |

### 正确性

- Qwen3-0.6B：贪心输出 `The capital of France is...` CPU / op-offload / 全量 Vulkan **逐字节一致**。
- 困惑度（Pride and Prejudice，4×512）：
  - Build 11 纯 CPU：**PPL = 40.5381**
  - Build 12 `-ngl 0`：**PPL = 40.5837**（差 **+0.11%**，数值等效）
  - Build 12 纯 CPU（`--no-op-offload`）：**PPL = 40.5864**
- FlashAttention 对比（纯 CPU）：`-fa 0` PPL 40.5864 / `-fa 1` PPL 40.5381，差异 0.12%（且 `AUTO` 默认已等价 `-fa 1`）。
- server / cli / completion / perplexity / train 均随新默认正常启动、无断言。

### 结论与推荐

- 本机（集成显卡）最佳默认：**`-ngl 0`**（Build 12 默认）——dense 模型 pp 提升 1.6~2.4×，tg 不降。
- 长提示、批量场景（RAG / 摘要）：可显式 `-ngl auto` 或 `-ngl 99` 换取更高 pp，代价是 tg 变慢。
- 需要与 CPU 逐位一致的确定性推理：加 `--no-op-offload` 关闭算子 offload。
- MoE 模型（SmallThinker）收益不明显，保持默认即可。

---

## Build 13：训练强制纯 CPU（修复 Vulkan 对训练指标的污染）

**问题**：交付带 Vulkan 后，训练时若启用算子 offload（默认开），图调度器会把部分前向算子放到 GPU，
导致训练准确率指标异常（`acc ≈ 8.25e8%`），并可能改变梯度。Build 11（无 Vulkan）时 acc 正常（~10%）。

**修复**：`tools/lumina-train/lumina-train.cpp` 在训练前强制 `n_gpu_layers = 0` 且 `no_op_offload = true`，
并在日志中提示。推理工具不受影响（仍默认 Vulkan 算子 offload）。

**验证**（stage3 persona 5 条，n_ctx=256，rank 4）：

| 版本/配置 | acc |
|---|---|
| Build 12 默认（offload 开） | 825753607.03（异常） |
| Build 12 `--no-op-offload` | 7.03 / 10.16 / 10.42（正常） |
| Build 13 默认 | 7.03 / 10.16 / 10.42（正常，日志提示自动禁用） |

---

## Build 14：机器感知 + 模型感知的自动设备策略（一套预编译、程序自决）

需求：一套预编译版本，程序根据机器/模型自动决定用 CPU 还是 GPU，用户无需选择。

实现：
- `common_model_params_to_llama()`：用户未显式 `-ngl` 时，按设备类型自动决定：
  - 无 GPU → `n_gpu_layers = 0`（纯 CPU）；
  - 只有集成显卡（IGPU）→ `n_gpu_layers = 0`（层留 CPU，仍允许算子 offload）；
  - 存在独立显卡（GPU）→ `n_gpu_layers = -1`（auto，按显存 fit 层）。
- `common_init_result()`：模型加载后扫描 GGUF 元数据里的 `*.expert_count` 检测 MoE；
  若是 MoE 且用户未显式设置，自动 **关闭算子 offload**（`cparams.op_offload = false`）。
- 显式 `-ngl` / `--op-offload` / `--no-op-offload` 始终优先。

验证（本机 IGPU）：

| 模型 | 自动决策 |
|---|---|
| Qwen3-0.6B（dense） | 层留 CPU + 算子 offload（`integrated GPU, keeping layers on CPU`） |
| SmallThinker（MoE） | 层留 CPU + **关闭算子 offload**（`MoE model detected: disabling GPU op-offload`） |
| 显式 `-ngl 99` | 尊重用户（无策略日志） |

正确性：Qwen3 默认 op-offload PPL **23.7397** vs 纯 CPU **23.7639**（+0.10%），且快 ~1.9×。
MoE pp 修复：算子 offload 使 SmallThinker pp 慢 ~30%（143 vs 207 t/s），关闭后恢复。

---

## Build 15：GPU 训练可行性实验（结论：不可行，默认关闭）

背景：用户希望 GPU 训练以加速。实验：训练工具加入 `LUMINA_TRAIN_ALLOW_GPU=1` 逃生口，
允许 `-ngl 99` 走 Vulkan。

结果（SmallThinker，persona 4 条）：
- 能跑完（exit 0），但 **无加速**：~28s/step，与纯 CPU 相同（反向仍走 CPU + 跨设备拷贝）。
- `acc` 指标被破坏（8.25e8%），loss 与纯 CPU 略有差异（不可信）。

根因：**Vulkan 后端没有实现任何反向/梯度算子**，只有 `ggml-cpu` 实现了反向（含本分支为 MoE
补的 `MUL_MAT_ID_BACK_*` 等）。因此 GPU 训练要么无收益，要么结果不可信。

交付：保留 `LUMINA_TRAIN_ALLOW_GPU`（**默认关闭**、带显式警告）仅供实验；默认训练强制纯 CPU。
真正的 GPU 训练需 CUDA/ROCm（NVIDIA/AMD），本机只有 Intel 集显，不可用。

---

## Build 16：修复 GPU 训练下 acc 指标损坏 + 控制台横幅

**acc 修复**：根因是 Vulkan 的 `count_equal` 内核固定写 32 位，而 `ggml_count_equal` 输出是 **I64**（训练用）——
类型不匹配导致读数错误。修法：`ggml-backend-vk` 的 `supports_op` 仅当目标为 I32 时才用 Vulkan，
I64 自动回退 CPU。`ggml/src/ggml-vulkan/ggml-vulkan.cpp` 一处修改。

验证（训练 acc）：

| 模型 | 修复前 | 修复后 |
|---|---|---|
| Qwen3-0.6B（dense, `-ngl 99`） | 8.59 → 4.1e8 / 5.5e8 | 8.59 / 9.38 / 8.85（正常） |
| SmallThinker（MoE, `-ngl 99`） | 8.25e8 | 7.03 / 10.16 / 10.42（正常） |

loss 同时与 CPU 一致（未受影响）。

**控制台横幅**：`common/build-info.*` 新增项目信息打印，现已在 `--version`、CLI 交互横幅、server 启动时显示：

```
Build 16 (Based on llama.cpp b-10991)
  GitHub   : https://github.com/Happy-380/Lumina-Engine
  Donate   : https://afdian.com/a/lumi_380  (tip the developer / support development)
  Bilibili : https://space.bilibili.com/3546376302496335
```

---

## 回退记录

- 未采用「修改 `auto` 语义以区分集显/独显」的方案：会深入模型加载逻辑，风险高，按稳妥原则放弃。
- 未采用「OpenMP 开关」等需要重配重建且收益不确定的方案。
