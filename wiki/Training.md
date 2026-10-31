# 训练引擎：量化模型微调（QLoRA）

> 🌐 **中文** | [English](Training.en.md)

本仓库新增了一个 **C++ 训练引擎** `Lumina-Engine-train`，用于对 **量化模型**
（`Q4_K` / `Q8_0` / `Q1_0` 等，任意 GGUF 量化类型）以及 `.gguf.lumina` 压缩容器进行微调。

> 结论先行：**量化权重不可直接训练**。本引擎采用 **QLoRA** 方案——基座权重保持冻结（仍以
> 量化形式存放），只训练低秩适配器（LoRA 的 A/B 矩阵）。这样既不需要解量化再训练再重新量化，
> 也把显存/内存需求压到很小。

> ⚠️ **Build 26 修复了训练梯度累加器从不清零的严重问题**：旧版每个优化器步实际应用
> “训练开始至今的全部历史梯度之和”，更新步长无界增长、loss 线性攀升。Build 26 之前
> 产出的 LoRA 检查点/适配器均建议废弃并从基座重新训练。

---

## 1. 为什么不是“直接微调量化权重”

上游 `llama-finetune`（`examples/training/finetune.cpp`）只支持 **F32 模型**，原因是：

- `llama_set_param()` 只接受 `GGML_TYPE_F32` 的张量，量化权重无法注册为可训练参数。
- 反向传播需要 `GGML_OP_OUT_PROD`，而当基座权重被 **repack** 到额外 buffer 类型时，该算子
  不被支持（见 §5）。
- 即便能训练，全参数微调 8B 模型需要“权重 + 梯度 + AdamW 动量”约 4×参数量，本机 8 GB 内存不可行。

因此本引擎走 **LoRA/QLoRA**：冻结的量化权重只参与前向，梯度只回传到新增的低秩矩阵。

---

## 2. 快速开始

工具位于交付目录 `Lumina-Engine-Bin\Lumina-Engine-train.exe`。

```bat
REM 用 Q4_K / Q8_0 / Q1_0 等量化模型做 QLoRA 微调（rank=8, alpha=16）
Lumina-Engine-train.exe ^
  -m model.Q4_K_M.gguf ^
  -f train.txt ^
  -c 512 -b 512 -ub 128 ^
  -t 6 -lr 1e-4 --epochs 3 ^
  -lora-rank 8 -lora-alpha 16 ^
  -o model.lora.gguf

REM 使用数据集（.jsonl，自动识别 text / messages / conversations）+ 独立验证集
Lumina-Engine-train.exe ^
  -m model.Q4_K_M.gguf ^
  --dataset data/stage2_sft/train.jsonl --dataset-val data/stage2_sft/val.jsonl ^
  -c 512 -b 512 -ub 128 --cpu-cores 8 -lr 5e-5 --epochs 2 ^
  -lora-rank 64 -lora-alpha 128 ^
  --save-steps 500 --save-every-epochs 1 --keep-last 3 ^
  -o model.lora.gguf

REM 压缩容器同样可用（首次加载会解压到 %TEMP% 并缓存）
Lumina-Engine-train.exe -m model.gguf.lumina -f train.txt -c 512 -b 128 -ub 128 -t 6 -o model.lora.gguf
```

训练完成后得到一个 **标准 LoRA 适配器 GGUF**，可直接用现有推理工具加载：

```bat
Lumina-Engine-cli.exe -m model.Q4_K_M.gguf --lora model.lora.gguf -p "..." -n 128 -t 6
```

也可以（可选）用 `Lumina-Engine-export-lora` 把适配器合并回基座模型。

---

## 3. 参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `-m, --model` | 基座模型（`.gguf` 或 `.gguf.lumina`，可量化） | — |
| `-f, --file` / `-p, --prompt` | 单一训练文本 | — |
| `--dataset` | 训练数据集（`.jsonl`，见 §2.1） | — |
| `--dataset-val` | 独立验证数据集（`.jsonl`，同格式） | 空（改用 `--val-split` 切分） |
| `--dataset-limit` / `--dataset-max-tokens` | 每个数据集最多加载的记录数 / token 数（`0` = 不限制） | `0` |
| `--dataset-chat-template` | 覆盖用于渲染对话的 chat template | 模型自带 |
| `--mask-history` / `--no-mask-history` | 对话数据是否只对 assistant 回复计算 loss | `--mask-history` |
| `-o, --out-file` | 输出 LoRA 适配器文件名 | `lumina-lora.gguf` |
| `--cpu-cores` | 训练使用的 CPU 核心（线程）数；`0` = 全部可用 | `0` |
| `-lora-rank` | 低秩维度；`0` = 全参数微调（仅 F32 模型可用） | `8` |
| `-lora-alpha` | LoRA 缩放；有效缩放 = `alpha / rank` | `16` |
| `-lora-targets` | 逗号分隔的基座张量子串，命中即挂适配器 | `attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down` |
| `-lr, --learning-rate` | 学习率（AdamW） | `1e-5` |
| `-epochs` | 训练轮数 | `2` |
| `-wd, --weight-decay` | AdamW 权重衰减 | `0` |
| `--val-split` | 未提供 `--dataset-val` 时，从训练集切出的验证集比例 | `0.05` |
| `--save-steps` / `--save-every-steps` | 每 N 个优化步保存一次检查点（`0` = 关闭） | `500` |
| `--save-every-epochs` | 每 N 个 epoch 保存一次检查点（`0` = 关闭） | `1` |
| `--keep-last` | 只保留最近 N 个检查点，旧的自删除（`0` = 全部保留） | `0` |
| `--checkpoint-dir` | 检查点目录 | 与 `-o` 同目录 |
| `--resume` / `--no-resume` | 从检查点目录中最新的检查点续训 | `--no-resume` |
| `-c, -b, -ub` | 上下文 / batch / micro-batch | 见 `--help` |

### 2.1 数据集格式（`--dataset` / `--dataset-val`）

数据集为 JSONL（每行一个 JSON 对象），逐行自动识别三种格式：

| 格式 | 示例 | 用途 |
|------|------|------|
| 纯文本 | `{"text": "一段训练文本"}` | 继续预训练（下一个 token 预测） |
| OpenAI 消息 | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` | 指令微调 / SFT |
| ShareGPT | `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}` | 指令微调 / SFT |

* 对话记录使用**模型自带的 chat template**（亦可用 `--dataset-chat-template` 覆盖）逐条渲染，
  因此与推理时的 chat 格式一致；
* 默认 **`--mask-history`**：只对 `assistant` 回合计算 loss，system / user 及模板标记被掩码
  （通过在标签里写入 `-1` 哨兵实现，掩码位置的 loss 与梯度严格为 0）；
* 纯文本记录始终对整个序列计算 loss；
* 训练集与验证集可分开指定（`--dataset-val`）；同一个 token 流内部的验证集用 `--val-split` 切分。

### 2.2 检查点与续训

每个检查点写出两个文件：`<out>.ckpt-<step>.gguf`（LoRA 权重）与 `<out>.ckpt-<step>.json`
（记录 `epoch` / `step` / rank / alpha / targets / 基座模型）。命名中的 `<out>` 为 `-o` 去扩展名后的
文件名。`--resume` 会扫描检查点目录、取 `step` 最大的一个，把 LoRA 权重拷回新适配器并从记录的 epoch
继续训练（`llama_adapter_lora_copy_weights()`）。

> `-lora-targets` 也会匹配 `wq/wk/wv/wo/w1/w2/w3` 等命名，适配不同架构。

---

## 4. 输出格式

输出是标准的 llama.cpp LoRA 适配器 GGUF：

```
general.type            = adapter
general.architecture    = <与基座一致>
adapter.type            = lora
adapter.lora.alpha      = <alpha>
tensors: <base_tensor_name>.lora_a / <base_tensor_name>.lora_b
```

张量形状与 `convert_lora_to_gguf.py` 一致（`lora_a` = `[n_in, rank]`，`lora_b` = `[rank, n_out]`），
因此可被 `llama_adapter_lora_init()` / `--lora` 直接加载，也能与上游工具互操作。

---

## 5. 实现要点

- **可训练适配器**：新增公共 API
  `llama_adapter_lora_init_trainable(model, rank, alpha, targets, n_targets)`，
  为命中的权重张量分配 F32 的 A/B（A ~ N(0, 1/√n_in)，B = 0，初始为恒等零贡献），
  并注册到 `model.loras`。保存用 `llama_adapter_lora_save(adapter, path)`。
- **参数注册**：`llama_context::opt_init()` 在原有模型参数之外，额外把已挂接适配器的
  A/B 通过 `ggml_set_param()` 注册为可训练参数；量化基座因为不是 F32 而被自动跳过（保持冻结）。
- **图尺寸**：训练反向图约为前向图的 2 倍，`graph_max_nodes()` 在训练期间（`opt_reserve` /
  `opt_ctx` 有效时）把节点预算翻倍，避免 `cgraph->n_nodes < cgraph->size` 断言。
- **关闭 repack**：训练工具自动设置 `no_extra_bufts`（等价 `-nr`）。repack 后的权重 buffer
  类型不实现 `GGML_OP_OUT_PROD`，而反向传播计算激活梯度时会用到
  `out_prod(量化权重, grad)`。原生量化 buffer 支持该算子。F32 模型不受影响。
- **上下文对齐**：`llama_context` 会把 `n_ctx` 向上取整到 256 的倍数，因此训练文本必须长于
  实际生效的上下文（工具会给出明确报错）。

---

## 6. 局限

- **MoE（混合专家）模型已支持训练**：为支持 `SmallThinker-4BA0.6B` 等 MoE，本次补齐了 ggml
  自动微分缺失的反向算子：`GGML_OP_MUL_MAT_ID_BACK_AS/_B`（专家权重的 3D 适配器与激活梯度）、
  `REGLU`/`SIGMOID`/`TANH` 激活、`CLAMP` 反向、broadcast `DIV` 反向，以及批量索引的
  `GET_ROWS_BACK`。MoE 的 router（`*_inp`）与专家（`*_exps`）都可以训练。
- 仅验证 CPU 路径；CUDA/Metal 等后端未测试（新反向算子仅在 CPU 后端实现）。
- 全参数微调（`-lora-rank 0`）仅对 F32 模型有意义，量化模型上回退为“无可训练参数”。
- 训练期间关闭 repack，前向会略慢（仅训练阶段）。
- 目前支持 `--file`/`-p` 单一文本，或 `--dataset`/`--dataset-val` 的 `.jsonl`
  （`text` / `messages` / `conversations` 三种格式，对话数据支持 assistant-only loss 掩码）。

---

## 7. 已验证

在 `Qwen2.5-0.5B-Q4_K_M.gguf`（量化基座）上：

- `Lumina-Engine-train` 训练 1 个 epoch，loss 约 `2.55 → 1.99`；
- 生成的适配器含 `288` 个张量（`144` 对 A/B），元数据与形状正确，权重非零；
- `Lumina-Engine-completion.exe --lora <adapter>` 正常加载并输出连贯文本（`... is Paris.`）；
- `.gguf.lumina` 作为基座可直接加载（透明解压）。

在 `SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf.lumina`（MoE 基座）上：

- `Lumina-Engine-train` 训练 1 个 epoch，loss 约 `1.93 → 1.44`；
- 适配器含 `384` 个张量（`192` 对）、其中 `192` 个为 3D 专家适配器（如 `[1536, 4, 32]`）；
- `Lumina-Engine-completion.exe --lora <adapter>` 能加载并推理（MoE 的 `MUL_MAT_ID` 已可反向）。

### Build 11：数据集 / 验证集 / 检查点

在 `SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf`（MoE 基座）上使用 `D:/Dataset/data` 的切片：

- `stage1_pt`（`text`）：数据集加载与下一个 token 预测训练正常；
- `stage2_sft`（`messages`）与 `stage2_sft_sharegpt`（`conversations`）：对话按 chat template 渲染，
  记录加载与 token 统计正确；
- `stage3_persona`（含 system/user/assistant）：`--dataset-val` 独立验证集生效（5 train / 5 val
  datapoints），`--mask-history` 掩码后约 `231 / 968` token 可训练（即 assistant 回复部分）；
- 检查点每 `2` 步保存一次且 `--keep-last 2` 自动清理旧检查点；`--resume` 从最新检查点恢复
  权重并继续训练；最终 LoRA 适配器可被推理工具加载（输出连贯）。

