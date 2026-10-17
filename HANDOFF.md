# 交接文档 — 自动设备策略与性能优化 / Build 12–24

> **当前交付：Build 24**
> - 交付目录：`C:\Lumina-Engine\Lumina-Engine-Bin`（14 个 CPU 变体 + 全部工具 + 运行库）
> - 训练工具：`Lumina-Engine-Bin\Lumina-Engine-train.exe`（`--version` = `build 24`）
> - 构建树：MSVC `build-release` + Clang `build-clang` 双树均全量重建，14 个 CPU 变体
>   由 `bench\assemble_bin.bat` 组装进 `Lumina-Engine-Bin`（54 文件）
> - 详细优化记录：`OPTIMIZATION_LOG.md`；训练流程：`TRAINING_PIPELINE.md`
> - 最近四版：Build 24 = 所有控制台统一启动画面；Build 23 = 训练默认用满逻辑核（约 1.3×）；
>   Build 22 = 训练 ETA 误报修复 + `--dataset-stride`；Build 21 = `--resume` 续训修复

---

## Build 24（本次）：所有控制台统一启动画面

**目标**：让 `cli` / `server` / `train` / `bench` / `perplexity` / `quantize` / `imatrix` /
`tokenize` / `gguf-split` / `rpc-server` 等所有控制台工具在启动时显示同一套
Lumina-Engine 启动画面（ASCII logo + Build 号 + 项目链接），并顺手把 CLI 的 Build
号从 23 升到 24。

**改动**：
- `CMakeLists.txt`：`LLAMA_BUILD_NUMBER = 24`。`llama_build_info()` 继续由该值生成，
  因此 `--version` 与启动画面永远一致：`Build 24 (Based on llama.cpp b-10991)`。
- `common/build-info.cpp.in` / `common/build-info.h`：新增 `lumina_ascii_logo()`，
  `llama_print_lumina_banner()` 改为输出完整启动画面（ASCII logo + build + GitHub /
  爱发电 / Bilibili）。
- `common/arg.cpp`：`common_params_parse()` 成功后统一调用 `llama_print_lumina_banner()`，
  于是所有走公共参数解析的工具（含全部 `examples/`）自动获得该画面；`cli` 例外
  （跳过统一画面，由 `tools/cli/cli-context.cpp` 打印带模型信息与命令列表的完整横幅）。
- 移除重复打印：`tools/server/server.cpp`、`tools/cvector-generator/cvector-generator.cpp`
  不再各自单独打印横幅。
- `tools/gguf-split/gguf-split.cpp`、`tools/rpc/rpc-server.cpp`：不使用公共参数解析的
  工具显式打印同一画面（RPC server 为保持不链接 `llama-common`，使用编译期
  `LUMINA_RPC_BUILD_NUMBER` + 同一段 ASCII art）。
- `tools/cli/cli-context.cpp`：删除本地 logo 副本，改用共享的 `lumina_ascii_logo()`。

**验证**：启动画面渲染为（Build 号随 `LLAMA_BUILD_NUMBER` 自动更新）：

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
`Lumina-Engine-Bin`；`Lumina-Engine-cli.exe --version`、`server` / `train` / `bench`
启动画面均显示 `Build 24 (Based on llama.cpp b-10991)`。

---

## Build 23（历史）：训练默认用满逻辑核（`--cpu-cores 0`，约 1.3×）

**背景**：用户以固定命令在 6 核 / 12 线程的 i5-10400、8 GB 机器上做 `-c 32768` 全量
训练，实际吞吐约 31 s / 微批（2,601,728 微批 ≈ 2.2 万小时）。要求「不改变命令、不增加
RAM、无损」提速。

**定位**：用内置算子剖析（`LUMINA_OP_PROF=1`）测得：短上下文（`-c 4096/8192`）下
`MUL_MAT_ID_BACK_B`（MoE 专家反向）与 `OUT_PROD`（LoRA / 注意力反向）占 ~55–60%；
长上下文（`-c 16384` 尾段）下注意力 `MUL_MAT` + `SOFT_MAX`/`SOFT_MAX_BACK` + 部分
`OUT_PROD` 合计 ~65%。这些算子的共同瓶颈是**内存带宽**。

**根因**：`--cpu-cores 0` 时训练线程数取引擎默认的「物理核数」（6），把 6 个 SMT 逻辑
线程全部闲置；而上述反向算子主要是带外存访问的 f32 向量乘加，逻辑线程能继续吃满带宽。

**改动**：
- `tools/lumina-train/lumina-train.cpp`：`--cpu-cores 0`（默认）改为使用**全部逻辑核**
  （`std::thread::hardware_concurrency()`）；显式 `--cpu-cores N` 行为不变。命令行无需改动。
  该改动不影响算术：每个内核仍按整行 / 整列划分工作，逐元素累加顺序与线程数无关，
  实测前 5 个微批 loss 与 6 线程**逐位一致**。
- `CMakeLists.txt`：`LLAMA_BUILD_NUMBER = 23`。

**验证**（`-c 4096 -b 2048 -ub 128`，SmallThinker Q4_K_M，1 个数据点）：
- 6 线程 258 s → 12 线程 **202 s（1.28×）**；loss 序列逐位一致。
- `-c 16384` 长上下文剖析确认为注意力主导，符合长上下文训练的实际瓶颈。

**说明（还可以更快，但会改变训练语义，默认未启用）**：默认 `--dataset-stride 0` 等价
`n_ctx/2`（相邻窗口 50% 重叠，每个 token 被训练两次）。命令里加入 `--dataset-stride 32768`
可去掉重叠、epoch 时间约**再减半**（Build 22 起支持）。这会让每个 token 在一个 epoch 中
只被训练一次，属于训练语义变化，因此未设为默认。

---

## Build 22（历史）：训练 ETA 误报修复 + `--dataset-stride`（约 2× 提速）

**问题**：`-c 32768 -b 2048 -ub 128 --resume` 训练时 ETA 从 ~400h 持续涨到 ~10000h，
看起来“越跑越慢”。用户日志 `D:\Dataset\output\train-monitor21b.log` 中 ETA 从 1479h 单调涨到 13010h。

**根因**：实际吞吐恒定（相邻数据点周期 7635s / 7624s，即 29.8 s/微批），是 ETA 估计器有偏：
`--resume` 时 `ibatch`（全局微批计数）从检查点续接，而 `t_ibatch`（本进程耗时）从 0 开始，
累计均值 `t/ibatch` 从远低于真实值开始缓慢收敛，于是 ETA 长时间单调上升；
再叠加单个 32k 数据点内部注意力随上下文位置 O(context) 增长（7s → 70s 的锯齿）。

**改动**：
- `ggml/src/ggml-opt.cpp`：`ggml_opt_epoch_callback_progress_bar()` 的 ETA 改用
  “最近一个完整数据点”的滑动窗口吞吐（窗口 = `n_ctx/n_ubatch`，钳制到 [16,4096]），
  窗口未满时回退累计均值；顺带修正微批大小（`ggml_opt_inputs()` 是 1D，应取 labels 的 `ne[1]`）。
- 新增 `--dataset-stride N`（`common/common.h`、`common/arg.cpp`、`tools/lumina-train/lumina-train.cpp`）：
  默认 `0` 保持 `n_ctx/2`（50% 重叠）；设为 `n_ctx` 去掉重叠，epoch 时间约减半。
- `common/build-info.cpp.in`：banner 版本号由 `LLAMA_BUILD_NUMBER` 动态生成。
- `CMakeLists.txt`：`LLAMA_BUILD_NUMBER = 22`。

**验证**：`-c 512 -b 256 -ub 64` 冒烟：普通训练 ETA 稳定下降；从 `ckpt-32` 恢复时
第 2 帧即显示正确的 ~32:54 并稳定（旧实现会持续上涨）。MSVC + clang 双树重建，
组装 54 文件 / 14 个 CPU 变体，`--version` = `build 22`。

**交付**：Build 22，全部二进制已重建并组装到 `Lumina-Engine-Bin`。

---

## Build 21（历史）：修复 `--resume` 从第 1 步重训

**问题**：`--resume` 从检查点继续时，进度显示 `data=1`，整个 epoch 的数据从头重训。

**根因**：`llama_opt_epoch()` 固定从 `idata=0` 开始；检查点只记录了 `step`/`epoch`，
没有“从当前 epoch 的第几个数据点继续”的信息。

**改动**：
- `llama_opt_epoch()` 新增 `idata_start` / `ubatch_start`（`include/llama.h`、`src/llama-context.*`）。
- 恢复位置所在数据点的已训练前缀只做**前向**（重建 KV cache，无反向、无优化步，
  `ggml_opt_reset_opt_i()` 保持梯度累积相位不变），因此不更新任何权重（真正无损续训）。
- `lumina-train.cpp` 把检查点微批步换算回 `(数据点, 微批)`，进度与检查点编号均正确续接。
- 新增 `ggml_opt_reset_opt_i()`；`--save-steps` 文档由“优化步”更正为“训练微批”。

**验证**：从 `ckpt-600` 恢复，日志显示 `resuming at datapoint 37/1945 ... (global step 600)`，
进度从 `data=601` 开始，第 616 微批正常完成首次优化步。

**交付**：Build 21，全部二进制已重建并组装到 `Lumina-Engine-Bin`（55 文件 / 14 个 CPU 变体）。

---

## Build 20（历史）：长上下文训练 RAM 优化（消除磁盘交换）

**目标**：在 8 GB 机器上，用
`-c 32768 -b 2048 -ub 128 -lora-rank 64 -lora-alpha 128 --epochs 1 --resume`
训练 `SmallThinker-4BA0.6B-Instruct.Q4_K_M`（MoE）时不再进入磁盘交换，且**训练无损、命令不变**。

**根因**（详见 `OPTIMIZATION_LOG.md` Build 20）：
1. 非 FlashAttention 训练反向需要同时保活全部 32 层的 `[n_ctx, n_ubatch, n_head]` softmax 激活
   （`n_ctx=32768` 约 6.0 GiB，实测 compute 每微批 +28 MiB）。
2. `ggml_opt_result::pred` 逐 token 累积，一个 epoch 约 1.3 GiB。
3. 数据集 token 流为匿名内存（~635 MiB），压力下被迫换出。

**改动**：
- `src/llama-graph.cpp`：训练时给注意力输出 matmul 打标记（`src[2]=k, src[3]=q, src[4]=mask`）。
- `ggml/src/ggml.c::ggml_compute_backward()`：对有标记的注意力 matmul **反向现场重算**
  kq/softmax 并求 `grad_v/grad_k/grad_q`，激活内存 O(n_layer·n_ctx) → O(n_ctx)，逐位无损。
- `ggml/include/ggml-opt.h` / `ggml/src/ggml-opt.cpp`：新增 `ggml_opt_params::collect_pred`；
  训练置 `false`，停止累积预测。
- `src/llama-context.cpp`：`opt_init` 设置 `cparams.training`、`collect_pred=false`，并打印真实训练内存。
- `src/llama-cparams.h`：新增 `training` 标志。
- `tools/lumina-train/lumina-train.cpp`：token 流写入临时文件并 `mmap`，`std::vector` 随即释放。
- **`--resume`**：见上方 Build 21（Build 20 时尚未修复）。

**验证**（本机 8 GB）：stage1 全量训练 private working set **稳定 ~3.5 GiB**（旧版持续增长至 5.3 GiB+），
compute 峰值 ~1.7 GiB（旧版外推 ~7.9 GiB），剩余物理内存 >1.6 GB，**无交换**；
loss 序列与优化前**逐位一致**。

**交付**：Build 20，全部二进制已重建并组装到 `Lumina-Engine-Bin`（55 文件 / 14 个 CPU 变体）。

---

## Build 19（历史）：Vulkan Q1_0（1-bit）DP4A GEMV 移植 + 自动后端/模型选择

**来源**：用户提供的 `1.patch`（把 CPU q1t 的 1-bit 优化移植到 Vulkan），原本基于 Build 16；
本次已 rebase 到 Build 18 源码树，并修复了集成缺陷。

**改动**：
- 新增 `ggml/src/ggml-vulkan/vulkan-shaders/lumina_q1_{quantize,repack,gemv,gemm,gemm_cm2}.comp`
  与 `vulkan-shaders-gen.cpp` 注册；`ggml-vulkan.cpp` 增加 Q1_0 int8/DP4A 旁路与权重 repack 缓存。
- **相对原始 patch 的修复**：
  - `lumina_q1_repack` / `lumina_q1_gemv` 的 `wg_denoms` 由 `{32,1,1}` 改为 `{1,1,1}`：
    `ggml_vk_dispatch_pipeline()` 会用 `wg_denoms` 去除元素数，原写法只派发了 1/32 的工作组。
  - UMA（集显）张量缓冲解析与 `ggml_vk_tensor_subbuffer()` 一致，走 `ggml_vk_host_get()`。
  - repack 缓存键冲突防护（张量名/形状不同即重建）与 `prealloc_y` 复用缓存失效。
  - 能力守卫：仅当设备支持 `integer_dot_product` 且路径启用时才创建管线，否则自动回退。
- **自动后端/模型选择**：
  - `ggml_backend_dev_caps` 新增 `q1_dp4a`（追加字段，ABI 安全）；Vulkan 设备按能力上报。
  - `common/common.cpp` 设备策略：**Q1_0 模型 + 具备 DP4A Q1 内核的 GPU** → offload（`-ngl -1`）；
    否则沿用「无 GPU→纯 CPU / 集显→层留 CPU / 独显→offload」。
  - Q1_0 DP4A GEMV 默认 **auto**：**独显启用、集显(UMA)禁用**（原因见下）。

**验证（本机 Intel Iris Xe，Vulkan 1.3，`int dot: 1`）**：
- 正确性（Bonsai-1.7B Q1_0，贪心）：CPU / Vulkan 原生 / 无 DP4A 回退 全部输出 `Paris`；
  `LUMINA_Q1_VK_GEMV=1` 强制启用 DP4A 时输出退化为重复 token —— **该内核在 Intel UMA 上端到端不正确**
  （隔离的 `test-backend-ops` MUL_MAT 用例通过，`GGML_VULKAN_CHECK_RESULTS` 每算子误差 ~1%，
  但整模型 tg 输出错误；作者在独显 Turing 上验证正常）。因此默认在 UMA 上关闭。
- 性能（`Lumina-Engine-bench`）：Bonsai-1.7B `-ngl 99` 下 DP4A GEMV 相对原生 GEMV
tg64 **40.8 → 152 t/s（3.7×）**；Bonsai-8B **11.7 → 83.7 t/s（7.1×）**。prefill 不受影响。
  本机自动策略（集显）保持 Q1_0 在 CPU，tg 约 90 t/s（比原生 Vulkan 快、且正确）。

**开关**：`LUMINA_Q1_VK_GEMV=0/1`（强制关/开）、`LUMINA_Q1_VK_CACHE_MB`（repack 缓存预算）、
`LUMINA_Q1_VK_INT8=1`（实验性 prefill）。

**交付**：Build 19，全部二进制已重建并组装到 `Lumina-Engine-Bin`。

---

## Build 16（历史）：GPU 训练 acc 修复 + 控制台横幅

**acc 修复**：Vulkan `count_equal` 内核固定写 32 位，而 `ggml_count_equal` 输出 I64（训练）——类型不匹配。
`ggml-vulkan.cpp` 的 `supports_op` 改为仅当目标为 I32 才用 Vulkan，I64 自动回退 CPU。
验证：Qwen3（dense）acc 8.59/9.38/8.85；SmallThinker（MoE）acc 7.03/10.16/10.42（之前 8.25e8）。

**横幅/链接**：`common/build-info.*` 新增 `llama_print_lumina_banner()`，在 `--version`、CLI 交互横幅、
server 启动时输出 Build 号 + 项目链接（GitHub / 爱发电 / Bilibili）。

**D:\Dataset 回归**：四种格式（`text` / `messages` / `conversations` / `messages+val`）均可正常加载训练。

---

## Build 14–15（历史）：自动设备策略 + GPU 训练实验

**Build 14（机器+模型感知的自动设备策略）**：一套预编译、程序自决 CPU/GPU。
- `common_model_params_to_llama()`：未显式 `-ngl` 时，无 GPU/IGPU → `n_gpu_layers = 0`；dGPU → `-1`（auto）。
- `common_init_result()`：模型加载后扫描 `*.expert_count` 检测 MoE，自动关闭算子 offload（MoE 走
  算子 offload 反而慢 ~30%）。显式参数始终优先。
- 验证：Qwen3 自动 op-offload（PPL 23.7397 vs 纯 CPU 23.7639，+0.10%，快 ~1.9×）；
  SmallThinker 自动关闭 op-offload（pp 143→207 t/s）。

**Build 15（GPU 训练实验）**：结论**不可行**。Vulkan 后端无任何反向算子（只有 CPU 实现）。
`LUMINA_TRAIN_ALLOW_GPU=1 -ngl 99` 能跑完但无加速（~28s/step，反向仍走 CPU）且 `acc` 指标损坏。
保留该开关（默认关闭、带警告）仅供实验；默认训练强制纯 CPU。

---

## Build 13（历史）：训练强制纯 CPU（修复 Vulkan 污染训练指标）

**问题**：Build 12 交付带 Vulkan 后，训练时默认启用算子 offload，图调度器把部分前向算子放到 GPU，
导致训练准确率指标异常（`acc ≈ 8.25e8%`），可能影响梯度。Build 11（无 Vulkan）时 acc 正常（~10%）。

**改动**：`tools/lumina-train/lumina-train.cpp` 训练前强制 `n_gpu_layers = 0` 与 `no_op_offload = true`。
推理工具不受影响（仍默认 Vulkan 算子 offload）。

**验证**：Build 12 默认 acc = 825753607（异常）；Build 12 `--no-op-offload` 与 Build 13 默认 acc = 7.03 / 10.16 / 10.42（正常）。

---

## Build 12（历史）：Vulkan 后端 + 默认设备策略优化

**目标**：优化框架运行模型的效率（读取 / 生成），在 CPU 与 Vulkan 上均有优化，保证推理正确；版本号升至 **Build 12**。

**问题/背景**：本机为 Intel Iris Xe 集成显卡（Vulkan 1.4.363）。只启用 Vulkan 后端还不够：`-ngl auto`（旧默认）会整层 offload 到 iGPU，使生成速度下降约 25%。

**改动**：
- `configure-release.bat`：检测 `D:\VulkanSDK\sdk`（本机用 LunarG 安装包 7z 载荷手动解包，无需管理员），存在则 `-DGGML_VULKAN=ON`，否则自动回退 CPU-only。
- `common/common.h`：`n_gpu_layers` 默认 `-1`（auto）→ **`0`**（层留 CPU + 允许算子 offload）。
- `common/arg.cpp`：移除旧的 `GGML_ASSERT(params.n_gpu_layers < 0)`（该断言仅为旧的 `string_format` 默认值写法而设）并修正 `-ngl` 帮助文案。
- 交付新增 `ggml-vulkan.dll`（共 55 文件）。

**验证**（详见 `OPTIMIZATION_LOG.md`）：
- Qwen3-0.6B-Q8_0：`-ngl 0` pp512 **≈2.4×**（420 vs 207），tg 持平；`-ngl 99` pp 671（tg 变慢）。
- phi-3-mini Q4_K_M：`-ngl 0` pp256 **≈1.6×**（76 vs 48），tg 略降。
- SmallThinker MoE：收益在噪声范围内，tg 不降。
- 正确性：贪心输出 CPU / op-offload / 全量 Vulkan 逐字节一致；PPL 40.5381 → 40.5837（+0.11%）。
- FlashAttention：确认 `AUTO` 默认已等价 CPU `-fa 1`（pp 175 vs 关闭 146）。

**注意**：本机 CPU 状态在长会话中明显漂移（同配置可差 20%+），A/B 以同批次相邻测量为准。

---

# 交接文档 — 数据集训练 / 检查点 / Build 11

---

## Build 11（历史）：数据集格式 / 验证集 / 检查点 / CPU 核心数

**目标**：让 `Lumina-Engine-train`（C++ QLoRA 训练引擎）支持 `D:\Dataset\data` 的数据集格式，
新增训练 CPU 核心数、检查点保存/保留/续训参数，支持训练集与验证集分离；版本号升至 **Build 11**。

**数据集格式**：`--dataset` / `--dataset-val` 指向 `.jsonl`，逐行自动识别：
- `{"text": "..."}` —— 继续预训练（stage1_pt）；
- `{"messages": [{"role","content"}, ...]}` —— OpenAI 风格 SFT（stage2_sft / stage3_persona）；
- `{"conversations": [{"from","value"}, ...]}` —— ShareGPT（stage2_sft_sharegpt）。

对话记录用模型自带 chat template **逐条渲染**（`common_chat_format_single`），因此可准确标记
assistant 片段；默认 `--mask-history` 只对 assistant 回合计算 loss，其余标签写 `-1` 哨兵
（`src/llama-context.cpp::opt_epoch_iter` 跳过负标签，loss 与梯度严格为 0）。验证集可用
`--dataset-val` 单独指定，或用 `--val-split` 从训练流切分。

**改动**：
- `tools/lumina-train/lumina-train.cpp`：重写，新增数据集加载、train/val 打包、检查点保存/清理/续训、
  `--cpu-cores`。
- `common/common.h` / `common/arg.cpp`：新增 `--dataset` / `--dataset-val` / `--dataset-limit` /
  `--dataset-max-tokens` / `--dataset-chat-template` / `--mask-history` / `--cpu-cores` /
  `--save-steps`(=`--save-every-steps`) / `--save-every-epochs` / `--keep-last` /
  `--checkpoint-dir` / `--resume` / `--no-resume`。
- `include/llama.h` / `src/llama-adapter.cpp`：新增并实现 `llama_adapter_lora_copy_weights()`
  （续训时把检查点 LoRA 权重拷入新适配器）；同时修复上一会话中断遗留的
  `llama_adapter_meta_val_str()` 缺行编译错误。
- `src/llama-context.cpp`：`opt_epoch_iter()` 支持负标签掩码。
- `common/build-info.cpp.in`、`CMakeLists.txt`：**Build 11**，`LLAMA_BUILD_NUMBER = 11`。
- `bench/assemble_bin.bat` / `build-bin.bat`：clang 缺失时回退到 MSVC 基座，并从
  `bench\clang_variants_backup\` 还原 5 个 clang 专属 CPU 变体；所有 `.bat` 的 VS 路径
  `D:\VS2026` → `D:\VS2022`（本机环境已变化）。

**验证**（`SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf`，`-t 8` = 4C8T）：
- `stage1_pt`（text）训练正常，loss 约 `2.66 → 2.83`（样本少、步数短，仅验证流程）；
- `stage2_sft`（messages）/ `stage2_sft_sharegpt`（conversations）加载与 token 统计正确；
- `stage3_persona`：`--dataset-val` 生效（5 train / 5 val datapoints），`--mask-history` 下
  `231 / 968` token 可训练；
- 检查点每 2 步 + 每 epoch 保存，`--keep-last 2` 自动删除旧检查点；`--resume` 从最新检查点恢复
  并继续训练；最终适配器 `--lora` 可加载推理（`你好，你是谁？` → 连贯回复）。
- `Lumina-Engine-Bin`（54 文件，14 个 CPU 变体）已组装；`--version` → `build 11`，
  `llama_build_info()` → `Build 11 (Based on llama.cpp b-10991)`。

> 环境说明：本机 **没有可用的 clang**（`D:\llvm` 不存在，VS 自带 LLVM 只有 clang-format/tidy）。
> 本次全量重建用 MSVC 14.41（`D:\VS2022`），5 个 clang 专属 CPU 变体沿用旧构建产物（ggml 未改动，ABI 兼容）。

---

# 交接文档 — MoE 微调支持（QLoRA on MoE）与 Build 10

---

## Build 10（历史）：MoE 模型可训练 + 全量产物

**目标**：让 `Lumina-Engine-train`（C++ QLoRA 训练引擎）支持 **MoE 混合专家模型**
（如 `D:\SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf.lumina`）的微调；版本号升至 **Build 10**。

**问题**：MoE 前向使用 `GGML_OP_MUL_MAT_ID`，而 ggml 自动微分 `ggml_compute_backward` 缺少其反向，
训练直接中止；此外 MoE 专家权重是 3D `[n_in, n_out, n_expert]`，需要 3D 适配器。

**改动**：
- `ggml/include/ggml.h` / `ggml-rpc.h`：新增 `GGML_OP_MUL_MAT_ID_BACK_AS` / `_BACK_B`；
  `GGML_OP_COUNT 101 → 103`，`RPC_PROTO_PATCH_VERSION 0 → 1`。
- `ggml/src/ggml.c`：在 `ggml_compute_backward` 接入 `GGML_OP_MUL_MAT_ID`（A/B 梯度）；
  补齐 `REGLU`（MoE 激活）、`SIGMOID`/`TANH`（gating）、`CLAMP`、broadcast `DIV` 的反向；
  `ggml_get_rows_back` 支持批量索引（`ids=[n_expert_used, n_tokens]`）。
- `ggml/src/ggml-cpu/ggml-cpu.c` / `ops.cpp`：两个新反向算子的 CPU 内核（`_BACK_B` 对量化
  `as` 逐行反量化），批量 `get_rows_back` 的 scatter-add。
- `src/llama-adapter.cpp`：`llama_adapter_lora_init_trainable` 为 3D 专家权重创建 3D 适配器
  `[n_in, rank, n_expert]` / `[rank, n_out, n_expert]`。
- `src/llama-context.cpp` / `.h`：训练期图节点预算翻倍（`opt_reserve`）。
- `common/build-info.cpp.in`、`CMakeLists.txt`：**Build 10**，`LLAMA_BUILD_NUMBER = 10`。

**验证**：
- `SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf.lumina`：1 epoch，loss 约 `1.93 → 1.44`；适配器
  `384` 张量（`192` 对，其中 `192` 个 3D 专家适配器，如 `[1536,4,32]`）；`--lora` 正常加载推理。
- `Qwen2.5-0.5B-Q4_K_M`（稠密）回归：loss 约 `2.55 → 1.99`，适配器可加载。
- `build-bin.bat` 全量重建（MSVC + Clang）并组装 `Lumina-Engine-Bin`（54 文件）；
  `Lumina-Engine-cli.exe --version` → `build 10`；`llama_build_info()` → `Build 10 (Based on llama.cpp b-10991)`。

---

# 交接文档 — 二值查表（Q1LUT）与全量产物更新

---

## Build 8（本次）：压缩容器扩展名约定 `.gguf.lumina`

**目标**：让压缩产物与原始模型一眼可区分（`A.gguf` → `A.gguf.lumina`），并保证引擎对
`.gguf` 与 `.gguf.lumina` 都能直接加载/使用；版本号升至 **Build 8**。

**做法 / 改动**：
- `tools/lumina_gguf_pack.py`：默认输出改为 `<input>.lumina`（**原文件保留**）；`main()` 支持
  `pack.py <input>`、`pack.py <input> <output>`、`pack.py <input> <output> <level>`，并打印
  `output` 与「原文件保留」提示。
- `tools/lumina_gguf_unpack.py`：usage 文案同步为 `.gguf.lumina`。
- `common/preset.cpp`：新增 `is_gguf_model_name()` / `model_name_from_file()`，服务端模型目录
  列表同时识别 `.gguf.lumina`（显示名去掉后缀）。
- `common/arg.cpp`：`-m` 的 bash 补全由 `!*.gguf` 改为 `!*.gguf*`（含 `.gguf.lumina`）。
- `src/lumina-gguf.h`：注释说明扩展名约定；解码本身仍只认魔数，扩展名无关。
- `common/build-info.cpp.in`：`llama_build_info()` → **Build 8**。
- 文档：`README.md`、`HANDOFF.md`、`wiki/ModelCompression.*`、`wiki/LosslessCompression.*`、
  `wiki/Changes.*`、`wiki/Home.*`、`wiki/Delivery.*` 全部同步。

**结果 / 约定**：仓库根目录同时保留两份：
- `D:\SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf` —— **原始未压缩**模型（2,630,213,408 B）
- `D:\SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf.lumina` —— **压缩容器 v2**（2,479,779,960 B，94.28%）

**验证**：
- Python：`lumina_gguf_unpack.py verify <...gguf.lumina> <...gguf>` → OK，SHA-256
  `fb584d0b...ab61` 与原文件一致；`.lumina` md5 `b90162213a8b0f9912bd048179a3ab0b`。
- `Lumina-Engine-Bin`（53 文件）已由 `build-bin.bat` 全量重建；横幅
  `Build 8 (Based on llama.cpp b-10991)`。
- `-m ...gguf.lumina` 与 `-m ...gguf` 两种路径均正常输出 `Paris`；解压临时文件 SHA-256 一致。
- 打包器默认命名已验证：`python tools\lumina_gguf_pack.py <...gguf>` → 生成 `<...gguf>.lumina`。

---

## Build 7（历史）：模型无损压缩优化（容器 v2）

**目标**：在**完全无损**（不量化、不剪枝）的前提下，进一步压小
`SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf`。

**做法**：在 Build 6 的 k-quant 字段拆分基础上，加入两项可逆变换：

1. **fp16 字节平面**：`d` / `dmin` 不再按块交错，而是低字节流、高字节流分开
   （Q4_K 的 `d`、`dmin`，Q6_K 的 `d`）。
2. **上下文排序**：Q4_K 的 4-bit 量化值按子块分组后，按上下文键 `(scale << 6 | min)` 稳定排序；
   每个子分布各自集中，DEFLATE 建模效率显著提升（该流从 96.69% → 95.53%）。
   另将 6-bit `scale`/`min` 解包为字节流，Q6_K 的 `scales` 做奇偶拆分。

容器版本号升为 **v2**；`src/lumina-gguf.cpp` 解码器**同时兼容 v1 与 v2**（旧容器仍可加载）。

**改动**：
- `tools/lumina_gguf_pack.py`：v2 打包器（Q4_K 上下文排序 + fp16 字节平面 + scale/min 解包）。
- `tools/lumina_gguf_unpack.py`：v2 参考解包器，兼容 v1；`verify` 逐字节校验。
- `src/lumina-gguf.cpp` / `.h`：新增 v2 解码（计数排序复原、字节平面/奇偶还原），保留 v1。
- `common/build-info.cpp.in`：`llama_build_info()` → **Build 7**。
- `wiki/LosslessCompression.md` / `.en.md`、`wiki/ModelCompression.md` / `.en.md`、`README.md`、
  `wiki/Changes.md` / `.en.md`、`wiki/Home.md` / `.en.md`、`wiki/Delivery.md` / `.en.md`：同步更新。

**结果**：`2,630,213,408 B → 2,479,779,960 B`（**94.28%**，节省 **143.5 MiB**），
相比 Build 6 容器 v1（`2,506,568,815 B`，95.30%，117.9 MiB）再省 **25.6 MiB**。
解压结果与原文件 SHA-256 完全一致：

```
fb584d0b0f79f9e9944f91ac061e24726a8d014b6702213bfd5585729db7ab61
```

**验证**：
- Python：`lumina_gguf_unpack.py verify` 对 v1、v2 容器均为 OK。
- C++：MSVC 与 Clang 重建后的引擎均能加载 v2 容器；解码写出的临时文件
  `%TEMP%\lumina-<hash>.gguf` SHA-256 与原始文件一致。
- 冒烟：`Lumina-Engine-Bin\Lumina-Engine-cli.exe -m <v2容器> -p "The capital of France is" -n 8 -st`
  输出 `The capital of France is **Paris**`，横幅显示 `Build 7 (Based on llama.cpp b-10991)`。
- 交付目录 `Lumina-Engine-Bin`（53 文件）已由 `build-bin.bat` 全量重建并组装。
- 根目录模型文件已替换为 v2 压缩容器（同名 `.gguf`，md5 `b90162213a8b0f9912bd048179a3ab0b`）。

**还剩多少空间**：模型 99.7% 是 Q4_K/Q6_K 量化值。Q4_K 量化值零阶熵 3.87/4 bit（~96.7% 上限），
条件熵 3.82 bit（~95.6%），v2 已压到 95.5%，基本触及无损上限；Q6_K 量化值接近满熵。
因此 94.3% 已接近“不量化不剪枝”的理论极限。

---

## Build 6（历史）：模型无损压缩容器

**目标**：在不量化、不剪枝的前提下减小 `SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf`。

**做法**：新增一套“Lumina 无损压缩 GGUF”容器。打包器按 k-quant 字段拆分每个 256 权重块
（`d/dmin/scales` 与 4/6-bit 权重流分离）后 DEFLATE；引擎侧识别容器魔数 `LGGC`，加载时流式还原
为原模型并走原有 GGUF 路径。权重逐字节不变，未量化、未剪枝。

**改动**：
- `src/lumina-gguf.cpp` / `.h`（新增）：容器解码器，解压到 `%TEMP%\lumina-<hash>.gguf` 并缓存复用。
- `src/miniz/miniz.{c,h}`（新增）：内置单文件 DEFLATE（MIT/公有领域），无外部依赖。
- `src/llama.cpp`（修改）：`llama_model_load_from_file()` 接入 `lumina_gguf_resolve_path()`。
- `src/CMakeLists.txt`（修改）：编译新源文件。
- `tools/lumina_gguf_pack.py` / `lumina_gguf_unpack.py`（新增）：打包器与逐字节校验。
- `common/build-info.cpp.in`（修改）：`llama_build_info()` → **Build 6**。
- `wiki/ModelCompression.md` / `.en.md`（新增）：**中英双语用户指南**——如何压缩模型、如何加载压缩模型；
  `wiki/LosslessCompression.md` / `.en.md` 为容器原理与格式说明。

**结果**：`2,630,213,408 B → 2,506,568,815 B`（**95.30%**，节省 117.9 MiB）。
原文件与解压结果 SHA-256 均为
`fb584d0b0f79f9e9944f91ac061e24726a8d014b6702213bfd5585729db7ab61`。
交付目录 `Lumina-Engine-Bin`（53 文件）已用 MSVC + Clang 全量重编并组装；冒烟输出 `Paris.`，
横幅显示 `Build 6 (Based on llama.cpp b-10991)`。根目录模型文件已替换为压缩容器（同名 `.gguf`）。

> 构建与交付已全部改用 `C:\Lumina-Engine`（源码 `C:\Lumina-Engine\llama.cpp-b10991`，构建树
> `build-release` / `build-clang`，交付目录 `C:\Lumina-Engine\Lumina-Engine-Bin`）；不再使用 `D:\llama`。

---

> 日期：2025-09-23（第二次更新，见 §0）
> 状态：**Q1LUT 已转为默认启用；14 个 CPU 变体 + 工具已重新编译；交付目录改为 `C:\Lumina-Engine\Lumina-Engine-Bin`。**
> 旧入口 `D:\llama\llama` 已废弃（现统一为 `C:\Lumina-Engine`）。

---

## 0. 2025-09-23：Q1LUT 转正 + 交付目录变更

### 0.1 复核结论（推翻 9-22 的“查表对 tg/pp 都无收益”）

9-22 的 §4 结论是机器降频/后台负载下测得的假象。9-23 用**受控交替 A/B**（off/lut/off，控制预热与顺序）复核：

- **tg（`-p 64 -n 128 -t 6`）**：打包 GEMV **9.70 / 9.99 t/s** → Q1LUT **13.04 / 13.01 t/s**，**约 +30%**，已到单通道 DDR4 带宽上限。
- **pp**：从 `ne11 >= 8` 起查表全面快于 2×4 分块 GEMM，batch 越大优势越大：

| 测试 | 分块 GEMM | Q1LUT pp | 变化 |
|---|---:|---:|---:|
| pp8 | 16.9~18.3 | 20.8~23.7 | ~ +27% |
| pp64 | ~17.0 | ~21.4 | +26% |
| pp512 | ~15.8 | ~19.9 | +26% |
| pp1024 | 19.68 | 24.82 | +26% |
| pp2048 | 14.58 | 22.86 | +57% |

### 0.2 新的默认策略（按实际 token 数）

| 阶段 | 条件 | 内核 |
|---|---|---|
| tg | `ne11 == 1` | Q1LUT（默认启用） |
| pp | `ne11 >= 8` | Q1LUT（默认启用） |
| pp | `ne11 = 2..7` | 2×4 分块 GEMM |
| 任意 | `GGML_Q1_LUT=0` | 全部回退打包内核（省 ~0.95 GiB 内存） |
| pp | `GGML_Q1_LUT_PP=0` | pp 回退分块 GEMM |

### 0.3 代码改动

- `ggml/src/ggml-cpu/q1t.cpp`
  - 新增 `q1t_env_bool()`（布尔解析：`0/false/off/no` 为假）。
  - `q1t_lut_enabled()` / `q1t_lut_pp_enabled()` 默认由关改为 **开**。
  - `GGML_Q1_LUT_DEBUG` 也改用布尔解析（`=0` 不再强制打印）。
- `common/build-info.cpp.in`：`llama_build_info()` 固定返回 `Build 5 (Based on llama.cpp b-10991)`。
- `tools/cli/cli-context.cpp`：`LLAMA_ASCII_LOGO` 换成 Lumina ASCII logo。

### 0.4 交付目录 / 编译

- 新交付目录：`C:\Lumina-Engine\Lumina-Engine-Bin`（53 个文件；clang 基底 + 9 个 MSVC 变体覆盖 + 5 个运行库）。
- 一键：`C:\Lumina-Engine\build-bin.bat`（MSVC + Clang 全量 → `bench\assemble_bin.bat` 组装到新目录）。
- 冒烟：新目录下 `Lumina-Engine-cli.exe ... -p "The capital of France is" -n 8` 输出 `Paris.`，横幅显示新 logo 与 `Build 5 (Based on llama.cpp b-10991)`。
- 注：本次未更新 `D:\Lumina-AI\Lumina-Engine`；如需请另行同步。

---

## 1. 9-22 目标与结论（历史，已被 §0 推翻）

- 目标：按 BitNet TL 的思路实现“真 2 值”查表生成，pp 和 tg 都要提升，并更新 `/Lumina-Engine` 下所有产物。
- 结论（实测，详见 §4）：
  - **查表内核单线程比符号 GEMV 快 ~78%**（46.2 vs 25.9 GMAC/s）。
  - **但整机 tg 不提升**：tg 是 DRAM 带宽瓶颈，查表不减少权重字节数，`-n 512` 同场 A/B 基本持平。
  - **pp 也不提升**：现有 2×4 分块 GEMM 的 pp 内核约 60 GMAC/s，快于查表的 46 GMAC/s；pp 查表 pp128/256 反而慢 5%~10%。
  - **pp 的提升来自已有分块 GEMM**：pp512 优化 20.0 t/s vs 通用 14.14 t/s = **+41%**。
  - 默认线程数 = 物理核（本机 6），不使用超线程；只有显式 `-t 12`/`-t 0` 才启用 HT。
- 因此：查表保留为**实验性、默认关闭**；本次真正交付的是**最新优化后的全量产物**。

---

## 2. 代码改动（`C:\Lumina-Engine\llama.cpp-b10991`）

### 2.1 `ggml/src/ggml-cpu/q1t.cpp`
- `q1lut_build_group`：利用真二值对称性 `T[c ^ 0xf] = -T[c]`，只累加 `c=0..7`，其余取负填入（生成量减半）。
- `struct q1lut_codes::d0` 由 `float*` 改为 `ggml_half*`，`q1lut_fill_group` 直接拷贝原 `block_q1_0.d`（fp16），
  使 tg 权重流量回到 1.125 bit/权重（与打包格式一致）。
- 抽出 `q1lut_load_d0`（F16C 批量转 float）与 `q1lut_acc128`（一个 128 权重块、32 行的查表累加）。
- `q1lut_gemv32`：整表直写（默认路径）。
- `q1lut_gemv32_add` + `q1t_lut_kb()`：实验性 K 分块累加，环境变量 `GGML_Q1_LUT_KB`（默认 0 = 整表）。
- 新增 `q1t_lut_pp_enabled()`：环境变量 `GGML_Q1_LUT_PP`。
- 新增 pp 查表路径：按 `Q1LUT_PP_COLS=32` 列分块建表，`q1lut_gemv32` 一次算一列；门控
  `ne11 >= Q1LUT_PP_MIN(8)` 且同时设置 `GGML_Q1_LUT=1` 与 `GGML_Q1_LUT_PP=1`。
- `ggml_cpu_q1t_scratch_size` 增加 pp 查表临时表大小（仅在该路径启用时）。

### 2.2 `bench/bench_lut2.cpp`
- `lut_build_group` 同步改为对称生成。
- 新增 `lut_gemv_blocked`（K 分块、全行累加）与 `bench_lut3.exe K M iters KB`。

### 2.3 新增脚本（`C:\Lumina-Engine\bench\`）
- `build_haswell.bat`、`build_x64.bat`、`build_all_msvc.bat`、`build_all_clang.bat`、`build_lut3.bat`。

---

## 3. 产物更新（已完成）

- MSVC `build-release`：9 变体
  `x64 sse42 sandybridge haswell skylakex cannonlake cascadelake icelake alderlake`
- Clang `build-clang`：14 变体（额外）
  `ivybridge piledriver cooperlake zen4 sapphirerapids`
- 组装方式（沿用 wiki 的混合组装）：**clang 产物为基底 + 9 个 MSVC 变体覆盖**。
- 已更新目录（各 53 个文件，md5 已逐一比对一致）：
  - `D:\Lumina-AI\Lumina-Engine\`  ← 用户指定交付目录
  - `C:\Lumina-Engine\llama\`             ← 开发/README 引用目录
- 备份：
  - `C:\Lumina-Engine\bench\dll_backup\ggml-cpu-haswell.orig.dll`（9-22 原始 haswell）
  - `C:\Lumina-Engine\bench\dll_backup\Lumina-Engine-old\`（旧交付 53 文件）

---

## 4. A/B 实测数据（历史，已被 §0 复核推翻）

### 4.1 内核微基准（`bench/bench_lut2.cpp`，K=4096，M=65536，max err ~1e-8）
| 内核 | GMAC/s |
|---|---:|
| 符号 GEMV（Q8_128 单列） | 25.9 |
| 二值查表 GEMV（32 行，整表） | **46.2 (+78%)** |

### 4.2 整机 tg（`-p 64 -n 512 -t 6`，同场交替）
| 配置 | 通用 GEMV (t/s) | Q1LUT 整表 (t/s) |
|---|---:|---:|
| 原始 DLL | 11.54 / 11.53 | 11.55 / 11.53 |
| 当前 DLL | 11.69 / 11.66 | 11.67 / 11.56 |

→ 基本持平。tg 每 token 需读完 ~1.06 GiB 权重，是 DRAM 带宽瓶颈。

### 4.3 K 分块（`GGML_Q1_LUT_KB`，`-p 64 -n 128 -t 6`）
KB=1024 ~11.0、KB=2048 ~11.1、KB=4096 ~11.9、整表(KB=0) ~12.0 —— 整表最优，分块无收益。

### 4.4 pp 查表（`GGML_Q1_LUT=1 GGML_Q1_LUT_PP=1`）
| 测试 | 分块 GEMM (t/s) | pp 查表 (t/s) |
|---|---:|---:|
| pp128 | 25.91 | 23.24 (-10%) |
| pp256 | 24.89 | 23.52 (-5.5%) |
| pp512 | 22.70~22.97 | 22.66~22.85（持平） |

### 4.5 现有 pp 提升（交付产物，`-p 512 -t 6`）
优化路径 pp512 **20.00 t/s** vs 通用 `GGML_NO_Q1_TILE=1` **14.14 t/s** = **+41%**。

### 4.6 正确性
- 困惑度（`whitepaper.txt`，4 chunks，c=512）：LUT off = **5.7173**，LUT on = **5.7173**（逐项一致）。
- pp 查表：c=64 PPL 与 off 相同；长提示（~400 token）贪婪输出逐字一致（`diff` 为空）。
- 默认线程数：`Lumina-Engine-bench` 报告 6（物理核）。

---

## 5. 环境变量速查

| 变量 | 作用 |
|---|---|
| `GGML_Q1_LUT=1` | 启用 tg 查表（实验性，默认关） |
| `GGML_Q1_LUT_KB=N` | 查表 K 分块（默认 0 = 整表，实测最快） |
| `GGML_Q1_LUT_PP=1` | 额外启用 pp 查表（需配合 `GGML_Q1_LUT=1`；实测打不过分块 GEMM） |
| `GGML_Q1_LUT_DEBUG=1` | 打印查表缓存分配 |
| `GGML_Q1_GEMV_ROWS=1\|2\|4` | tg 单列 GEMV 行分组（默认 1；实测 1/2/4 在噪声内） |
| `GGML_NO_Q1_TILE=1` | 回退到通用 Q1_0 内核做 A/B |
| `GGML_NO_Q1_GEMV=1` | 只关 tg 单列 GEMV |

---

## 6. 9-22 遗留验证（已完成，见 §0）

1. **重跑最终验证**（今晚最后一组命令未返回，疑似机器卡顿/超时，非代码问题）：
   ```bat
   cd /d D:\Lumina-AI\Lumina-Engine
   set GGML_Q1_LUT=
   Lumina-Engine-perplexity.exe -m \d\llama\Bonsai-8B.gguf -f \d\llama\docx\whitepaper.txt -c 512 -t 6 --chunks 4
   set GGML_Q1_LUT=1
   Lumina-Engine-perplexity.exe -m \d\llama\Bonsai-8B.gguf -f \d\llama\docx\whitepaper.txt -c 512 -t 6 --chunks 4
   ```
   期望两者都输出 `[4]5.7173`。
2. **产物冒烟**：从 `D:\Lumina-AI\Lumina-Engine\` 运行
   `Lumina-Engine-cli.exe ... -p "The capital of France is" -n 8 --temp 0`，确认输出 `Paris`。
3. **确认 14 个 DLL 都可加载**：可在非 haswell 的 CPU 上验证；本机只会选 haswell。
4. **（可选）回归测试**：`test-backend-ops` Q1_0 MUL_MAT（当前 build 未编 tests）。
5. **决策点**：查表对 pp/tg 都无收益，是否
   - (a) 保留为实验特性（现状），或
   - (b) 若坚持要 pp/tg 数值提升，需换方向（例如 AVX-512/VNNI 平台、真·K 分块 fused GEMM、
     或更强的内存带宽平台），二值查表在该模型/本机上已证明不是答案。

---

## 7. 复现/重建命令

> 交付目录现为 `C:\Lumina-Engine\Lumina-Engine-Bin`，由 `bench\assemble_bin.bat` 组装
> （clang 产物为基底，9 个 MSVC CPU 变体覆盖）。`bench\build-bin.bat` 会依次跑
> MSVC → clang → 组装，但它的 `where clang` 检测不到 `D:\llvm\bin`，会**跳过 clang 构建**，
> 所以建议分三步手动执行：

```bat
REM 1) 全量 MSVC（9 变体 + 工具）
cmd /c C:\Lumina-Engine\bench\build_all_msvc.bat

REM 2) 全量 clang（14 变体 + libomp）；脚本内部已把 D:\llvm\bin 加入 PATH
cmd /c C:\Lumina-Engine\bench\build_all_clang.bat

REM 3) 组装到 C:\Lumina-Engine\Lumina-Engine-Bin
cmd /c C:\Lumina-Engine\bench\assemble_bin.bat
```

单变体快速构建（测试用）：`cmd /c C:\Lumina-Engine\bench\build_haswell.bat`。
内核微基准：`bench_lut3.exe 4096 8192 5 1024`。
（旧的 `cp build-clang/bin/*` + 手工覆盖 MSVC 变体的组装方式已被 `assemble_bin.bat` 取代。）

---

## 8. 已知非问题
- 编译时 `UI: HF download failed / no assets` 是离线环境的正常告警，不影响产物。
- `q1t.cpp` 编译在 `sse42/x64` 等非 AVX2 变体上有 `src1` 未使用的告警，来自既有的
  `ggml_cpu_q1t_scratch_size`，非本次引入。
- **训练 ETA 在小文档内呈“锯齿”**：同一个数据点内 attention 是 O(context)，微批耗时
  从头到尾上升，滑动窗口跨数据点边界后均值恒定。这是真实算力分布，不是变慢。
- **`--resume` 后 `t=` 从 0 重新计时**：`t=` 只表示本进程耗时；ETA 已不受此影响
  （Build 22 起用滑动窗口吞吐）。
- **改 `--dataset-stride` / `-c` / `-b` 后旧检查点的步号会对不上**：这些参数会改变
  `steps_per_epoch`，续训请保持训练配置一致（`TRAINING_PIPELINE.md` 已注明）。
