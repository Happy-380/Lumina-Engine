# Lumina-Engine 性能优化与 A/B 测试记录

> 环境：Windows 11 x64 / i5-12500H（4P+8E，12 物理核 / 16 线程）/ 16 GB DDR4 / Intel Iris Xe (Vulkan 1.4)
> 构建：MSVC 19.41（Build 12 起启用 Vulkan 1.4.363 后端）
> 测试模型：`D:\Qwen3-0.6B-Q8_0.gguf`、`D:\phi-3-mini-128k-instruct.q4_k_m.gguf`、`D:\SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf`
> 说明：本机 CPU 状态在长会话中有明显漂移（同一配置可差 20%+），因此下表以「同批次相邻测量」为准，绝对值仅供参考。
> Build 31 起的测量环境：Windows 11 x64 / **i5-10400（6C12T，无 AVX512）/ 8 GB DDR4**；测量期间后台有
> `Lumina-Engine-train.exe`（8 线程）在跑，绝对带宽会整体漂 20% 以上，因此改用「同一进程内交错 A/B + 多轮中位数」。

---

## Build 31（本次）：Q4_K GEMV 软件预取 —— CPU 解码吞吐 +13%

**背景**：CPU 单 token 生成的时间几乎全部花在量化 GEMV 内核上（`ggml_gemv_q4_K_8x8_q8_K`，
即 repack 后的 Q4_K×8 内核）。本机（i5-10400 / 6C12T / DDR4 / 8 GB）实测该内核单线程只能跑到
**6.0–7.5 GB/s**，而同一线程纯流式读的上限是 **11.6–13.1 GB/s** —— 说明它不是因为带宽不够，
而是**访存延迟没被填满**：一个 block 的 8 条 32B 载入要等上一个 block 的算术，乱序窗口覆盖不到。

**改法**（`ggml/src/ggml-cpu/arch/x86/repack.cpp`，`ggml_gemv_q4_K_8x8_q8_K` 的 block 循环）：

```c
{
    const char * pf = (const char *) (b_ptr + b + 2);   // 提前 2 个 block（每个 1152 B = 18 行）
    for (size_t pf_off = 0; pf_off < sizeof(block_q4_Kx8); pf_off += 64) {
        _mm_prefetch(pf + pf_off, _MM_HINT_T0);        // 整块拉进 L1
    }
}
```

**关键坑：不要加 `if (b + 2 < nb)` 边界判断**。加了之后热循环里的那次分支比末尾几次无用预取
贵得多，实测 **-13%**（见下表 `prefetch f2+guard`）；而 `PREFETCHh` 对无效地址不会 fault，
越过张量末尾是无害的。

**测法**（可复现）：`bench\build_bench_cpu_kern.bat` 会另开一个 `build-bench` 树（不动交付树），
编译 `llama.cpp-b10991/tests/bench-cpu-kern.cpp`：用 `ggml_quantize_chunk` 造真实 Q4_K 数据 →
按 loader 的规则 interleave → 调**真实内核入口**（`repack.h` 声明）→ 与**独立标量参考**逐列比对 →
在**同一进程内交错**跑各变体（抵消 CPU 频率/后台负载漂移）。变体代码由 `bench/gen_q4k_body.py`
从 `repack.cpp` 原文抽取生成（`tests/q4k_gemv_body.inc`），除注入的预取外与出厂代码逐字相同。

**内核级，单线程，18 MiB 矩阵（DRAM 驻留），24 轮交错中位数**：

| 变体 | 中位数 | 最好 25% | 相对出厂 |
|---|---:|---:|---:|
| 出厂（未改） | 6.00–6.47 GB/s | 6.45–7.12 | — |
| 预取整块 +1 | 6.56–7.37 | 7.29–8.17 | **+7…+19%** |
| 预取整块 +2 | 6.34–7.72 | 7.03–8.45 | **+10…+19%** |
| 预取整块 +4 | 6.37–7.15 | 6.98–7.92 | +7…+14% |
| 预取整块 +2 **带边界判断** | 5.49–5.73 | 5.89–6.44 | **-4…-15%** |
| 只预取块首行 +2 | 4.78–5.63 | 5.35–6.16 | -14…-28% |
| 改后（`shipped`） | 6.78 | 7.73 | **+13.6%** |

> 每格是多轮测量的区间（后台有训练任务，绝对带宽整体会漂 ±20%，同一进程内交错比的相对值稳定得多）。
> 复现确认（删掉 `build-bench` 后用 `bench\build_bench_cpu_kern.bat` 重新生成，20 轮）：
> `copy(pdist=0)` 5.87 GB/s / `shipped` 6.63 GB/s = **+13.0%**，`prefetch full 2` 6.61（与 shipped 一致）。

**端到端（真实模型 `Qwen3.5-0.8B-Q4_K_M.gguf`，4 线程，`llama-bench -p 16 -n 32`）**：

| 组合 | 解码 tg32 | 说明 |
|---|---:|---|
| Clang 构建 · 改前 | 13.40 / 13.45 / 13.54 / 13.78 / 14.12 / 14.49（中位 **13.66**） | `build-clang` |
| Clang 构建 · 改后 | 13.72 / 15.28 / 15.49 / 15.53 / 16.31 / 16.74（中位 **15.53**） | clang（`build-bench` 树）+ 预取，**+13.7%** |
| MSVC 变体 · 改前 | 5.82 / 6.26 / 7.70 / 7.86 / 8.24 / 9.16 / 9.82（中位 **7.86**） | 交付包里的 `ggml-cpu-haswell.dll` |
| MSVC 变体 · 改后 | 4.85 / 6.60 / 6.72 / 8.64 / 9.28 / 10.68 / 13.65（中位 **8.64**） | 逐对中位比 **+13%** |

- 预填充未受影响：`pp128` 改前 79.15 / 86.05 / 87.59 / 89.04，改后 87.42 / 87.60 / 89.08 / 89.46
  （GEMM 路径没动，这也是预期）。
- **L3 驻留时无收益也无损失**：0.56 MiB 矩阵 14 轮，改后 10.30 / 改前 10.07 GB/s（噪声内）。

**正确性**：

| 检查 | 结果 |
|---|---|
| 7 个变体 vs 独立标量参考（每列 dequant 后点乘，double 累加） | `max_abs_err = 4.8e-07`（|ref|max ≈ 3.5，容差 2.9e-06）全部 OK |
| 改后内核输出 vs 改前内核输出 | **逐位相同**（`memcmp` 一致，预取不参与任何算术） |
| 真实引擎贪心解码（`--temp 0 --seed 1234 -n 48`，改前/改后各一次） | 输出文本 **md5 相同** |
| `Lumina-Engine-Bin` 冒烟（Qwen3.5-0.8B，2 轮思考输出） | 正常，`--version` = `build 31` |

**交付**（Build 31，两棵构建树全部重编）：

| 项 | 状态 |
|---|---|
| 版本号 | `CMakeLists.txt` 的 `LLAMA_BUILD_NUMBER` 与 `tools/rpc/rpc-server.cpp` 的 `LUMINA_RPC_BUILD_NUMBER` 都 30 → **31** |
| `build-clang`（Clang 20，14 变体 + 全部工具） | 删除后重新 configure + 全量重编，**513 目标 0 error**（原缓存指向 `D:\Lumina-Engine`，直接构建会报 CMakeCache 目录不一致） |
| `build-release`（MSVC，9 变体 + 全部工具） | 重新 configure + 增量重编，0 error |
| Vulkan | **已装回 Vulkan SDK 1.4.363**（官方安装包载荷解包到 `D:\VulkanSDK\sdk`，无需管理员）；两棵树以 `-DGGML_VULKAN=ON` 重新 configure 并编译，Bin 多出 `ggml-vulkan.dll`（55 文件） |
| `Lumina-Engine-Bin` | `bench\assemble_bin.bat` 重新组装：**54 文件 / 14 变体**，`--version` = `build 31`，`ggml-rpc-server` 也是 `Build 31` |
| Kaggle | `kaggle/src/Lumina-Engine-src.zip` 重新生成（10,635,903 B，sha256 `746d13e1…e855e87`）；notebook 重新生成；`build_cuda.sh` / `make_notebook.py` 的版本校验改为 `Build 31` |
| 构建产物布局 | clang 构建改为**直接输出到 `Lumina-Engine-Bin`**（`configure-clang.bat` + `llama.cpp-b10991/CMakeLists.txt` 尊重 `CMAKE_RUNTIME/LIBRARY_OUTPUT_DIRECTORY`），`build-clang\bin` 重复副本（约 78 MB）已删除；`bench\assemble_bin.bat` 改为只覆盖 9 个 MSVC 变体 + 拷贝运行库 |
| 文档 | `README(.en).md`、`HANDOFF.md`、`KAGGLE_TEST.md`、`TRAINING_PIPELINE.md`、`wiki/*.md` 均按 Build 31 更新 |

**顺带确认的结论**：
- 同一机器上 `ggml_gemv_q4_0_8x8_q8_0`（LUT 版）与 Q4_K 版吞吐基本一致（≈9 GB/s L3 驻留），
  所以「把 Q4_K 改成 q4_0 那种 LUT 查表」没有 2 倍空间，本次没有采用。
- X86 上没有 Q6_K 的 repack 内核（只有 ARM/Q2_K 有），Q6_K 走普通 AVX2 `vec_dot`；
  它是下一个可动的目标（本机 Qwen3.5-0.8B 里 Q6_K 约占解码字节数的 1/4）。

---

## Build 30（历史）：CLI 多轮对话修复（空 assistant 轮污染历史）+ stdin EOF 死循环

**症状**：MoE 模型对话时，第二轮及之后都只回 `None`，且无法自愈。

**根因**：CLI 每轮无条件把 `{"role":"assistant","content":...}` 追加进历史。只要某轮没有可见输出
（首 token 即 EOS / 流中断 / content 解析为空 / 请求报错），这条空 assistant 轮就永久留在
`impl->messages` 里；下一轮请求变成 `[user, assistant(""), user]`，严格 chat template 的 MoE 模型
把空 assistant 段当作缺失字段，从此每轮只回一个占位串。

**改法**（`tools/cli/cli-context.cpp` + `tools/cli/cli-ui.h`）：

| 位置 | 内容 |
|---|---|
| `cli_context::run()` | `generate_completion()` 返回值生效；`content`/`reasoning` 同时为空的轮次不再入历史，改打印 `[empty response: nothing was added to the chat history]`（`--output` 也不写空段） |
| 同上（输入循环） | 空读取 = EOF → `break`；空行仍 `continue`（保留了跳过空消息的行为） |
| `list_and_ask_models()` | 空选择返回 `false`（原为 `continue` 死循环） |
| `ui::assistant_turn` | 新增 `finish()`，每轮结束（含中断/报错）补 `[End thinking]` 并复位颜色/行尾状态 |

**实测（8 GB / 12 线程 / 纯 CPU，同一个 Build 29 二进制作对照）**

| 场景 | Build 29 | Build 30 |
|---|---|---|
| `printf '你好\n1+1=?\n' \| cli -m SmallThinker -c 1024 -t 6 -n 24` | 两轮正常 → 无限 `> `，300 s 写出 **1,483,183 行**，需 kill（exit 124） | 两轮正常 → `Exiting...`，**37 行**，exit 0 |
| Qwen3.5-0.8B（thinking）2 轮 `--jinja` | `[Start thinking]` ×2、`[End thinking]` ×0 | ×2 / ×2，每轮闭合 |
| SmallThinker + `--lora stage1.lora.ckpt-7100` 2 轮 | 正常 | 正常 |

**构建**：Clang 增量 0 error（`cli-ui.h` / `cli-context.cpp` / `build-info` 重新编译），
`bench\assemble_bin.bat` 重新组装 Bin（54 文件），`--version` = **build 30**，启动画面 `Build 30 (Based on llama.cpp b-10991)`。

---

## Build 29：`--mask-history` 训练路径损坏定位 + 归一化口径显式化

**A/B（同一文件、同 token 流、各跑两次逐位一致）**，`-c 4096`，窗口 2,432–3,060 的每批 loss：

| 运行 | loss |
|---|---|
| `--mask-history` | 11.916 11.884 12.026 11.884 11.863 |
| `--no-mask-history` | 1.468 1.440 1.419 1.401 1.376 |

标签经诊断核对无误（过渡批 21 行零 / 43 行 one-hot，argmax 与 `labels_sparse`、下一个 token 一致），
token 流完全相同，推理侧同文本 PPL 2.03 → 问题出在「稀疏标签 → 训练前向」这一层
（疑为 labels 与计算图缓冲别名）。**可用结论：训练改用 `--no-mask-history`。**

代码：新增 `ggml_opt_result_add_rows()`；loss/accuracy 的「按已标注行归一化」改用实际评估行数
`nrows`（原先用 `ndata`=数据集点数，单位不一致）；llama-context 每 ubatch 上报行数。
实测两者数值一致（训练路径上 `ndata` 本就按行累加）→ 不改变既有输出，只把口径写清。

构建：Clang 全量 0 error，Bin 重装，`--version` = build 29。
长上下文训练按 `-c 8192 --no-mask-history` 重启（1,445 条 / 4.0M tokens，24.3 tok/s，ETA 46 h）。

---

## Build 28（历史）：训练 loss 显示口径修正（长上下文训练经实验确认可用）

**更正**：Build 28 初稿曾判定「长上下文训练产出不可用适配器」，该结论来自误读进度条 loss，已被推翻。

**决定性对照**（同一段 ChatML 文本、`-c 4096`、同位置 token）：

| 位置 | chat 记录（prompt 掩码） | 纯 text（不掩码） |
|---|---|---|
| 2,432 | 10.00 | ~1.48 |
| 2,560 | 11.88 | 1.44 |
| 3,008 | 11.19 | 1.33 |

原因：CE 核按**全部行**求均值（掩码行贡献 0 但进分母），而 `ggml_opt_result_loss()` 的补偿项
`ndata/nlabeled` 用「数据集点数 / 已标注行数」（单位不一致），于是已标注行很少时读数被放大数倍。

**其它实测**：纯 text 在同一 4k 窗口内 loss 3.00 → 1.20（单调下降）；推理 PPL `-c1024` 4.57 /
`-c4096` 2.03 / `-c8192`(24k 文本) 3.05，且与 `-b/-ub` 无关 → 长上下文前向与推理均正常。
吞吐：`-c1024` 27.5、`-c8192` 23.9、`-c32768` 14.2 tok/s。

**代码改动**：删除 `-c > 2048` 的错误告警，改为 `--mask-history` 开启时的口径说明（用 `--dataset-val` 判断进度）。
被判不安全的提速方案（跳过掩码反向 / f16 KV / repack）均未采用，原因见 HANDOFF.md §4。

## Build 27：训练数据集磁盘流式落盘（内存不再随语料线性增长）

**背景**：训练加载器把整个语料累计到两个 vector（`int32` ids + `uint8` mask = 5 B/token），
加载完成才写临时文件并 mmap。1.18 B tokens 需要 ~5.9 GiB 匿名内存，8 GB 机器必然失败。

**改动**（`tools/lumina-train/lumina-train.cpp`、`ggml/src/ggml-opt.cpp`）：

| 文件 | 内容 |
|---|---|
| lumina-train.cpp | `lumina_token_stream` 增加磁盘 sink（`FILE*` + 计数 + `sink_dir`）；`lumina_append()` 直接 `fwrite` ids/mask；新增 `lumina_stream_begin/finish/discard` + `lumina_append_padding`；`lumina_stream_view` 拆出 `map_file()` 并新增 `map_streamed()` |
| lumina-train.cpp | 每个进程使用 `%TEMP%\lumina-train\<tag>-<pid>\` 独立临时目录（避免并发/重叠 resume 互相覆盖） |
| ggml-opt.cpp | 进度条追加实时 `tok/s`；`ggml_opt_result_loss()` 在 result 无 batch 时返回 `NAN`（显示 `loss=n/a`）。验证集首行的 `loss=0.00000` 另属「该 micro-batch 全为掩码 prompt」的既有口径 |
| *.bat (10 个) | 去掉硬编码 `C:\Lumina-Engine`，改用 `%~dp0` 相对路径；`build-clang.bat` 传参改为 `--target` |

**同数据 A/B（`engine_full/train.jsonl`，`--dataset-max-tokens 300000000`，i5-10400 / 8 GB）**：

| 版本 | 加载峰值 RSS | 数据集就绪 RSS | 数据集统计 |
|---|---|---|---|
| Build 26 | 2074 MB | 1163 MB | 29282 rec / 300,021,789 tok（283,997,789 trainable）/ 278,340 datapoints |
| Build 27 | **1371 MB** | 1371 MB | **完全相同** |

`.stream.bin` 尺寸一致（60 M tokens：两者均 286.2 MiB）→ 训练器读到的字节不变。
冒烟集（40 rec / 6929 tok / 3863 trainable / 12 datapoints）与短验证集掩码填充行为也与 Build 26 一致。

**吞吐**：进度条现在直接给出 `tok/s`（本机 Qwen3.5-0.8B Q4_K_M：`-c 512` ≈ 30 tok/s，`-c 1024` ≈ 26–27 tok/s），
优化本身不改变每 token 的算力需求，改变的是**可加载语料规模**（不再受 5 B/token 限制）。

**回归项**：checkpoint 命名/元数据、`--resume`、`--dataset-limit/--dataset-max-tokens`、`--init-lora`、
`--mask-history`、`--dataset-chat-template`、LoRA 目标挑选与原内存回退路径均未改动。

---

## Build 26：修复训练梯度累加器从不清零（loss 线性攀升）

**现象**：`llama-finetune` / `llama_opt_epoch()` 路径（`Lumina-Engine-train`）在
`n_batch / n_ubatch > 1` 时，优化器每步实际使用的是从训练开始到当前步的**所有历史梯度之和**，
更新步长无界增长，loss 持续、近似线性上升（Build 25 的 Qwen3.5 实验 2.2 → 6.5 即此问题）。

**根因**：
- `ggml_opt` 的梯度累加器 `grad_accs` 是 `ctx_static` 中的持久张量，每 `opt_period` 个
  micro-batch 清零一次。
- 非静态图路径（`llama_context::opt_epoch_iter()` 每个 micro-batch 都调用
  `ggml_opt_prepare_alloc()` + `ggml_opt_alloc()` 重新构图）下，`ggml_opt_eval()` 每次求值后
  把 `opt_ctx->gb_grad` 置为 `nullptr`；原清零代码 `ggml_graph_reset(opt_ctx->gb_grad)`
  于是变成静默 no-op，累加器从未清零。
- 另有相位问题：仅前向求值（验证 / `--resume` 重建 KV cache）也会推进 `opt_i`，
  导致跨 epoch 的首次周期同样跳过清零。

**修复**：
- `ggml/src/ggml-opt.cpp`：新增 `ggml_opt_zero_grad_accs()`，直接操作持久累加器
  （参数清零，loss 累加器置 1）并记录 `grad_acc_is_loss`；`ggml_opt_alloc()` 在
  `backward && opt_period > 1 && opt_i == 0` 时调用；`ggml_opt_eval()` 仅在梯度图上推进
  `opt_i`；`ggml_opt_reset(..., false)` 也改用该函数。
- `tests/test-opt.cpp`：新增 `test_gradient_accumulation_nonstatic` 回归测试。

**验证**（非静态图，`opt_period=2`，每 micro-batch 梯度 = 1，SGD α=1）：

| | step 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧（累加器不清零） | 0 | -2 | -2 | -6 | -6 | -12 | -12 | -20 |
| 新（每周期清零） | 0 | -2 | -2 | -4 | -4 | -6 | -6 | -8 |

`test-opt`：120/120（AdamW）+ 48/48（SGD）通过；真实模型冒烟 train loss 3.03 → 2.44、
val loss ≈ 2.80，持续下降。

**结论**：Build 26 之前产出的 LoRA 检查点是在污染梯度下训练的，建议废弃并重训。

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
Vulkan 后端 DLL 沿用 Build 19 产物（当时本机 Vulkan SDK 已卸载；后端 ABI 未变，训练为纯 CPU）。
> Build 31 已重新安装 Vulkan SDK 1.4.363 并重建 Vulkan 后端。

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
