# Lumina-Engine 三段式训练流程（Stage1 → Stage2 → Stage3）

> 适用：`SmallThinker-4BA0.6B-Instruct`（MoE）QLoRA 三阶段微调
> 工具：`C:\Lumina-Engine\Lumina-Engine-Bin\Lumina-Engine-train.exe`（Build 24）
>
> Build 20 起训练已支持长上下文低内存：注意力反向现场重算 + 数据集内存映射，
> `-c 32768` 在 8 GB 机器上可稳定运行而不进入磁盘交换（详见 `OPTIMIZATION_LOG.md`）。
>
> Build 21 修复了 `--resume`：恢复训练时从检查点所在**数据点 / 微批**继续（不再从第 1 步重来）；
> 已训练过的前缀只做前向以重建 KV cache，不产生梯度、不更新权重。
>
> Build 22 修复了进度条 ETA 的误报：ETA 改为按“最近一个数据点”的滑动窗口吞吐估计，
> 不再因为 `--resume` 后累计均值起点偏低而**持续增长**；新增 `--dataset-stride`，
> 设为 `n_ctx` 可去掉窗口 50% 重叠，全量 epoch 时间约减半。
>
> Build 23 让 `--cpu-cores 0`（默认）真正用满**全部逻辑核**（此前只用了物理核数，
> 闲置了 SMT 线程）。命令行无需改动，实测约 1.3× 提速，且 loss 逐位一致。
>
> Build 24 让 `train` 与所有其它控制台（`cli` / `server` / `bench` 等）启动时显示同一套
> Lumina-Engine 启动画面（ASCII logo + Build 号 + 项目链接）。
> **全程无需合并 LoRA**：每一阶段用 `--init-lora` 在上一阶段的适配器上继续训练，各自单独保存。

---

## 1. 总体流程

```
基座模型
  │  stage1 继续预训练（text）
  ▼
stage1.lora.gguf
  │  stage2 长上下文指令微调（messages，--init-lora stage1）
  ▼
stage2.lora.gguf
  │  stage3 人设微调（messages，--init-lora stage2，低学习率/少轮次）
  ▼
stage3.lora.gguf   ← 累计包含 stage1+stage2+stage3，单独加载即可
```

- 三个阶段的 `-lora-rank` / `-lora-alpha` / `-lora-targets` **必须完全一致**，否则 `--init-lora` 会因形状不符而报错退出。
- 本文统一使用 `-lora-rank 64 -lora-alpha 128`（可用 `-lora-targets` 覆盖默认目标层）。
- `stage3.lora.gguf` 是**累计结果**（含 stage2 能力），因此人设训练后仍能保留指令跟随与 Markdown 能力；推理时**只挂 stage3 一个适配器**即可，不需要加载 stage2。

---

## 2. 路径一览

| 用途 | 路径 |
|------|------|
| 基座模型 | `D:\SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf` |
| 训练工具 | `C:\Lumina-Engine\Lumina-Engine-Bin\Lumina-Engine-train.exe` |
| Stage1 训练 / 验证 | `D:\Dataset\data\stage1_pt\train.jsonl` / `D:\Dataset\data\stage1_pt\val.jsonl` |
| Stage2 训练 / 验证 | `D:\Dataset\data\stage2_sft\train.jsonl` / `D:\Dataset\data\stage2_sft\val.jsonl` |
| Stage3 训练 / 验证 | `D:\Dataset\data\stage3_persona\train.jsonl` / `D:\Dataset\data\stage3_persona\val.jsonl` |
| 输出目录 | `D:\Dataset\output` |

先建输出目录（只需一次）：

```bat
if not exist D:\Dataset\output mkdir D:\Dataset\output
```

> 命令均为**单行**，可直接粘贴到 `cmd`。所有参数都写在同一行。

---

## 3. 三条训练命令

> 以下三条命令均为**全量训练**：不含 `--dataset-limit` / `--dataset-max-tokens`（即加载全部数据）。
> 需要限制数据量时请自行加这两个参数（用法见 §7 试跑）。

### Stage1 —— 继续预训练（PT，`{"text": ...}`）

```bat
C:\Lumina-Engine\Lumina-Engine-Bin\Lumina-Engine-train.exe -m D:\SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf --dataset D:\Dataset\data\stage1_pt\train.jsonl --dataset-val D:\Dataset\data\stage1_pt\val.jsonl -c 8192 -b 2048 -ub 128 --cpu-cores 0 -lr 1e-4 --epochs 1 -lora-rank 64 -lora-alpha 128 --save-steps 500 --save-every-epochs 1 --keep-last 3 -o D:\Dataset\output\stage1.lora.gguf
```

### Stage2 —— 长上下文指令微调（SFT，`{"messages": ...}`，从 stage1 继续）

```bat
C:\Lumina-Engine\Lumina-Engine-Bin\Lumina-Engine-train.exe -m D:\SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf --dataset D:\Dataset\data\stage2_sft\train.jsonl --dataset-val D:\Dataset\data\stage2_sft\val.jsonl --init-lora D:\Dataset\output\stage1.lora.gguf -c 8192 -b 2048 -ub 128 --cpu-cores 0 -lr 5e-5 --epochs 2 -lora-rank 64 -lora-alpha 128 --save-steps 500 --save-every-epochs 1 --keep-last 3 -o D:\Dataset\output\stage2.lora.gguf
```

> 如需 stage2 从基座独立训练（不接 stage1），删掉 `--init-lora D:\Dataset\output\stage1.lora.gguf` 即可；本文默认按 stage1 → stage2 → stage3 链式训练。

### Stage3 —— 人设微调（SFT，从 stage2 继续，推荐参数）

> 推荐参数：`-lr 5e-5`、`--epochs 1`（人设数据窄且短，学习率/轮次过高会把 stage2 的指令与 Markdown 能力带偏）。
> `-c 2048` 与人设数据的长度匹配（中位回复 ~48 字符，官方配置也是 2048），**不会**削弱长上下文能力；长上下文主要由 stage1/stage2 的 `-c 8192` 保留。
> **不传 `--init-lora` 就是从基座独立训练**——不是本文推荐的流程。

```bat
C:\Lumina-Engine\Lumina-Engine-Bin\Lumina-Engine-train.exe -m D:\SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf --dataset D:\Dataset\data\stage3_persona\train.jsonl --dataset-val D:\Dataset\data\stage3_persona\val.jsonl --init-lora D:\Dataset\output\stage2.lora.gguf -c 2048 -b 2048 -ub 128 --cpu-cores 0 -lr 5e-5 --epochs 1 -lora-rank 64 -lora-alpha 128 --save-steps 500 --save-every-epochs 1 --keep-last 3 -o D:\Dataset\output\stage3.lora.gguf
```

---

## 4. 关键参数说明

| 参数 | 说明 |
|------|------|
| `--dataset` / `--dataset-val` | 训练集 / 独立验证集（JSONL，自动识别 `text`、`messages`、`conversations`） |
| `--init-lora PATH` | 在已有 LoRA 上继续训练（本流程的链式关键；rank/targets 必须一致） |
| `--mask-history` | 默认开启：对话数据只对 assistant 回复算 loss |
| `--cpu-cores N` | 训练线程数，`0` = 使用全部**逻辑核**（本文用 `0`，Build 23 起为 `hardware_concurrency()`）；想限核可写 `--cpu-cores 8` |
| `-c` / `-b` / `-ub` | 上下文 / 逻辑 batch / micro-batch（内存不足时先降 `-c`，并保持 `-b` ≥ `-ub`） |
| `--dataset-stride N` | 相邻训练窗口的 token 步长；默认 `0` = `n_ctx/2`（窗口重叠 50%，每个 token 被训练两次）。设为 `n_ctx`（如 `32768`）让每个 token 只训练一次，epoch 时间约减半；大于 `n_ctx` 会被截断为 `n_ctx` |
| `-lr` / `--epochs` | 学习率 / 轮数 |
| `-lora-rank` / `-lora-alpha` | 低秩维度 / 缩放（三阶段必须一致） |
| `-lora-targets` | 目标层子串，默认覆盖 attn + ffn（MoE 的 router / 专家 / 共享专家都会被命中） |
| `--save-steps N` | 每 N 个训练微批（ubatch step）保存检查点（一个优化步 = `n_batch/n_ubatch` 个微批；默认 500） |
| `--save-every-epochs N` | 每 N 个 epoch 保存检查点（默认 1） |
| `--keep-last N` | 只保留最近 N 个检查点，旧的自删除（默认 0 = 全留） |
| `--resume` | 从同名最新检查点恢复权重与 epoch，并**跳过当前 epoch 已经训练过的数据点**（从检查点所在数据点边界继续，不再从第 1 步重来） |

检查点文件：`D:\Dataset\output\stage2.lora.ckpt-500.gguf` + 同名 `.json`（记录 epoch/step/rank/alpha/targets/基座）。

---

## 5. 断点续训

训练中断后，原命令**加上** `--resume` 再跑一次即可（其余参数保持一致）：

```bat
C:\Lumina-Engine\Lumina-Engine-Bin\Lumina-Engine-train.exe -m D:\SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf --dataset D:\Dataset\data\stage2_sft\train.jsonl --dataset-val D:\Dataset\data\stage2_sft\val.jsonl --init-lora D:\Dataset\output\stage1.lora.gguf -c 8192 -b 2048 -ub 128 --cpu-cores 0 -lr 5e-5 --epochs 2 -lora-rank 64 -lora-alpha 128 --save-steps 500 --save-every-epochs 1 --keep-last 3 --resume -o D:\Dataset\output\stage2.lora.gguf
```

> `--resume` 找的是**同名输出的检查点**；`--init-lora` 是**阶段起点**，两者可以同时出现：有检查点就用检查点，没有就用 `--init-lora`。

---

## 6. 推理（无需合并 LoRA）

```bat
REM 人设推理：只挂 stage3（已累计 stage1+stage2+stage3）
C:\Lumina-Engine\Lumina-Engine-Bin\Lumina-Engine-cli.exe -m D:\SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf --lora D:\Dataset\output\stage3.lora.gguf -p "你好，你是谁？" -n 128 -t 8

REM 通用指令 / Markdown：挂 stage2
C:\Lumina-Engine\Lumina-Engine-Bin\Lumina-Engine-cli.exe -m D:\SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf --lora D:\Dataset\output\stage2.lora.gguf -p "用 Markdown 表格对比 HTTP 与 HTTPS" -n 256 -t 8

REM 想弱化人设强度：--lora-scaled D:\Dataset\output\stage3.lora.gguf:0.8
```

---

## 7. 先小样本试跑（强烈建议）

> 本节命令才限制数据量，用于验证流程；正式训练请用 §3 的全量命令。

本机为纯 CPU 训练，MoE 每步（ubatch=128）约 13～17 秒，**全量训练不可行**：

| 阶段 | 全量训练 token | 估算耗时（本机 4C8T） |
|------|------|------|
| Stage1 | ~166.5M | 数百天量级 |
| Stage2 | ~67.2M | 上百天量级 |
| Stage3 人设 | ~1.2M（估） | 2～4 天 / epoch |

（吞吐约 10 token/s；引擎默认按 `stride = n_ctx/2` 打包，实际处理约 2× token，
可加 `--dataset-stride <n_ctx>` 去掉重叠、耗时约减半。正式全量训练建议改用 GPU 训练栈，
如仓库里 `configs/*.yaml` 对应的 LLaMA-Factory。）

```bat
C:\Lumina-Engine\Lumina-Engine-Bin\Lumina-Engine-train.exe -m D:\SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf --dataset D:\Dataset\data\stage3_persona\train.jsonl --dataset-val D:\Dataset\data\stage3_persona\val.jsonl --dataset-limit 500 -c 1024 -b 1024 -ub 128 --cpu-cores 8 -lr 5e-5 --epochs 1 -lora-rank 64 -lora-alpha 128 --save-steps 20 --keep-last 3 -o D:\Dataset\output\stage3.trial.lora.gguf
```

也可用 `--dataset-max-tokens 2000000` 限制 token 数。试跑通过后再去掉 `--dataset-limit` / `--dataset-max-tokens` 跑全量。

---

## 8. 注意事项

1. **三阶段 rank/alpha/targets 必须一致**，否则 `--init-lora` 报 `shape mismatch`。
2. 人设推荐 `-lr 5e-5`、`--epochs 1`；不要用配置文件里的 `1e-4` / 3 epoch（容易过拟合窄风格、削弱通用能力）。
3. `stage3.lora.gguf` 是累计适配器；**不要**再叠加 stage2（会重复计入 stage2 增量）。
4. 训练期间会自动关闭权重 repack（反向传播需要原生量化 buffer），前向略慢，仅训练阶段。
5. 上下文 `-c` 会向上取整到 256 的倍数（本文的 8192/2048 已对齐），且必须满足 `-c % -b == 0` 与 `-b % -ub == 0`（如 `-c 8192 -b 2048 -ub 128`）。数据总量必须显著大于 `-c`，否则会报“训练数据过短”。
6. **长上下文能力**：主要由 stage1/stage2 的 `-c` 决定，本文已用官方配置的 `-c 8192`；若内存允许可提高到 `-c 16384` 甚至 `-c 32768`（模型上限 32768，stage1 的 p99 ≈ 16.9K）。提高 `-c` 会增加注意力计算与 KV 缓存，但不会成倍改变每步耗时。
7. 若目录非默认，检查点会写在 `-o` 所在目录；可用 `--checkpoint-dir` 单独指定。
