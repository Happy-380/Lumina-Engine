# Kaggle 训练包（Lumina-Engine · CUDA 版）

把本机的 stage-1 续训搬到 Kaggle：

```
Lumina-Engine-train -m SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf \
  --dataset      data/stage1_pt/train.jsonl \
  --dataset-val  data/stage1_pt/val.jsonl \
  -c 32768 -b 2048 -ub 128 --cpu-cores 0 -lr 1e-4 --epochs 1 \
  -lora-rank 64 -lora-alpha 128 --save-steps 50 --save-every-epochs 1 --keep-last 3 \
  -o stage1.lora.gguf --resume
```

超参、上下文、stride、检查点命名**全部保持不变**，只把 Windows 路径换成了 Kaggle 上自动探测的路径。

---

## 1. 目录内容

| 文件 | 作用 |
|---|---|
| `Lumina-Engine-Kaggle-Train.ipynb` | **上传/运行的 notebook**（9 步：环境 → 找路径 → 编译 → 续训 → 打包） |
| `src/Lumina-Engine-src.zip` | **唯一需要上传的引擎包**：`llama.cpp-b10991/` 源码（含 CUDA 后端）+ `kaggle/` 下这些脚本 + 本 notebook，约 10 MB |
| `build_cuda.sh` | 在 Linux/Kaggle 上编译（`GGML_CUDA=ON`，静态链接，CUDA 配置失败自动回退 CPU-only） |
| `run_train.sh` | 探测路径、复制检查点、按原命令续训（支持时间预算 / dry-run） |
| `kaggle_paths.py` | 在 `/kaggle/input` 下递归定位基座模型 / train.jsonl / 最新 ckpt（不依赖解压层级） |
| `tools/pack_source.py` | 重新生成 `src/Lumina-Engine-src.zip` |
| `tools/make_notebook.py` | 重新生成 notebook |
| `windows/build_cuda_windows.bat` | 想在本机（Windows + NVIDIA）也编一个 CUDA 版时用 |

---

## 2. Kaggle 一次性准备

1. **上传数据集**：把 `D:\Dataset.zip` 上传为 dataset，名字 **`LuminaMoE-V1`**。
   Kaggle 会自动解压成 `/kaggle/input/lumamoe-v1/Dataset/…`（若没有解压，脚本会自动解压到
   `/kaggle/working/lumina/input/`，**两种都能跑**）。
2. **加入引擎包**：在 `LuminaMoE-V1` 页面点 *Edit → Add file*，把 `kaggle/src/Lumina-Engine-src.zip`
   加进同一个数据集（只有一个 11 MB 的文件）。
   - 也可以另外建一个 dataset 上传它；或把整个 `kaggle/` 目录作为一个 dataset 上传；
   - notebook 会在 `/kaggle/input` 里递归找 `kaggle_paths.py` / `Lumina-Engine-src.zip`；
   - 都找不到并且开了 Internet 时，会退化成 `git clone` 仓库。
3. **新建 Notebook**：*Create → Notebook → File → Import Notebook*，选
   `Lumina-Engine-Kaggle-Train.ipynb`。
4. **右侧设置**：Accelerator 选 **GPU T4 x2**（或 P100 / L4），Internet 可关。
5. *Add Data* → 挂上 `LuminaMoE-V1`。然后 **Run All**。

> 第一次运行会编译 20–60 分钟（4 vCPU）。编译完成后建议 **Save Version**，
> 把 Output 里的 `lumina/bin/` 保存成 dataset；下次挂载后第 4 格会直接复用，跳过编译。

---

## 3. 每一步在做什么

| 格子 | 内容 |
|---|---|
| 1 | 打印 vCPU / 内存 / 磁盘 / cmake / ninja / nvcc / GPU / torch，确认环境 |
| 2 | 找到脚本包 → 解压到 `/kaggle/working/lumina/src` → 确定引擎源码目录 |
| 3 | 运行 `kaggle_paths.py`：报告找到的基座模型 / train.jsonl / val.jsonl / 最新检查点 |
| 4 | `build_cuda.sh`：CUDA 编译（或复用 `/kaggle/input` 里旧的 `Lumina-Engine-train`） |
| 5 | `run_train.sh --dry-run`：打印最终命令行，便于和本机命令对照 |
| 6 | 正式续训（`TIME_BUDGET_HOURS` 默认 11 h） |
| 7 | 列出 `train_out/` 产物，并打包 `download/stage1-latest.zip`（最新 ckpt + 适配器） |
| 8 | 可选：CPU vs GPU 小样本 A/B（校验 GPU 路径数值是否可信） |
| 9 | 下一次会话怎么接着跑 + 常见问题 |

路径探测规则（`kaggle_paths.py`）：

- **基座模型**：`*.gguf` 且 ≥ 200 MiB、名字不是 `*.ckpt-*` / LoRA；优先名字含 `SmallThinker`，其次取最大；
- **训练集**：`train.jsonl`，优先路径含 `stage1_pt`、同目录有 `val.jsonl`，排除 `stage2/stage3`；
- **检查点**：`<stem>.ckpt-<step>.gguf`，取 **step 最大**（`stage1` 优先），并读取同名 `.json`
  打印 epoch / rank / alpha / targets。

---

## 4. 关于「CUDA 支持」的说明（重要）

这个引擎的训练栈（`ggml-opt` + `llama-train`）**反向传播和优化器只有 CPU 实现**：

- `ggml-cuda` 没有 `MUL_MAT_ID_BACK_AS/_BACK_B`（本项目为 MoE 自定义的两个反向算子），
  `OUT_PROD`（反向用到的、src0 为量化权重的乘）在 CUDA 里也只支持 F32；
- `HANDOFF.md` 的 Build 15 结论：GPU 训练「不可行」，`LUMINA_TRAIN_ALLOW_GPU=1` 能跑完但
  反向仍在 CPU，且指标可能损坏；因此 Build 13 起 `Lumina-Engine-train` **默认强制纯 CPU**
  （日志会打印 `restricting device list to CPU only`）。

所以本包里的 CUDA 构建：

- ✅ 正常编译、可运行，`Lumina-Engine-cli` 可 `-ngl 99` 用 GPU 推理；
- ✅ 训练默认走 CPU（**正确性优先**，和本机结果一致）；
- ⚠️ `ALLOW_GPU = "1"`（notebook 第 6 格）会给训练加 `-ngl 99` + `LUMINA_TRAIN_ALLOW_GPU=1`：
  能利用 GPU 做部分前向，但**不保证更快、不保证数值正确**，请先跑第 8 格的 A/B 对比 loss。

如果目标是「Kaggle GPU 上真正加速训练」，需要给 `ggml-cuda` 补反向算子（MoE 的
`MUL_MAT_ID_BACK_*`、量化 `OUT_PROD` 等）——这是另一个级别的工程，不在本包范围内。

### 速度预期

`-c 32768` 下每个微批（ubatch=128）在本机 6 核 / 12 线程约 30 s；Kaggle 4 vCPU 会更慢。
整个 epoch 是 ~2.6M 微批，**不可能一次跑完**。本包的设计就是：

```
每次会话跑 11 h → 存 ckpt-<step>.gguf → 下次 --resume 从该 step 继续
```

`--resume` 会打印 `resuming at datapoint N/10163, micro-batch M/256 (global step S)`，
`S` 只会随会话增多而增大。

---

## 5. 检查点 / 续训要点

- 检查点命名：`-o .../stage1.lora.gguf` → `stage1.lora.ckpt-<step>.gguf` + `.json`；
  `--resume` 在 **输出目录**里找 `<stem>.ckpt-*` 的最大 step。
- `/kaggle/input` 只读，所以 `run_train.sh` 会先把最新 `--keep-last 3` 个 ckpt（+ `.json`）
  复制到 `/kaggle/working/lumina/train_out/`，再 `--resume`。**`.json` 必须一起复制**，
  它记录了 `epoch`。
- **不要改** `-c / -b / -ub / --dataset-stride / 数据集文件`：它们决定
  `steps_per_epoch = idata_split × (n_ctx/n_ubatch)`；改了以后旧 ckpt 的步号就对不上
  （引擎会退回 epoch 开头，日志里表现为 `resuming at datapoint 0/…`）。
- `--save-steps 50` 是**训练微批**（不是优化步）。时间预算到点时，最后一个检查点之后
  （≤ 50 个微批）的进度会丢；想更细可以自己调小。

---

## 6. 跨会话怎么接

1. 第 7 格会生成 `/kaggle/working/lumina/download/stage1-latest.zip`（最新 ckpt + 当前适配器）。
   下载后把里面的 `stage1.lora.ckpt-*.gguf/.json` 放回 `LuminaMoE-V1` 数据集（*Edit → Add file*）。
2. 或者把本次 notebook 的 Output（`lumina/train_out/`）整体保存成 dataset，下次挂载，脚本会自动选最大 step。
3. 下次运行第 3 → 6 格即可（第 4 格会复用已编译产物）。

---

## 7. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| 第 3 格 `输入没有找齐` | 数据集里没有 ≥200 MiB 的 `.gguf` 或 `train.jsonl`；看报告里的“search roots” |
| 第 2 格 `找不到 kaggle_paths.py` | `Lumina-Engine-src.zip` 没加到数据集，或没挂载（*Add Data*） |
| 编译 20 分钟没动静 | 正常（ggml-cuda 有 160 个 `.cu`）；心跳行会每 30 s 打一行 |
| 编译失败 | 看 `/kaggle/working/lumina/logs/build.log`；CUDA 配置失败会自动回退 CPU-only |
| 启动报 `libcudart.so.12: cannot open…` | `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` |
| `--resume` 提示 `no checkpoint found` | ckpt 没进 `train_out/`：确认数据集里有 `stage1.lora.ckpt-*.gguf` **和** `.json` |
| loss 一直 `nan` | 学习率过大；试 `LUMINA_LR=5e-5`（第 6 格 env） |
| 会话被硬杀（超过 12 h） | `TIME_BUDGET_HOURS` 调小到 10.5–11 |

---

## 8. 维护：改了脚本后重新打包

```bat
python kaggle\tools\make_notebook.py     :: 重新生成 .ipynb
python kaggle\tools\pack_source.py       :: 重新生成 src\Lumina-Engine-src.zip（含脚本与本 notebook）
```

`pack_source.py` 只打包编译 Linux 版所需的内容（排除 `build-*`、`models/`、`docs/`、`tests/`、
`examples/`、`app/`、`.git` 等），源码 42.3 MiB → 压缩后 ~10 MB。

> **为什么排除 `tools/ui/src/`（635 个文件）**
> 1. 我们用 `-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF`，Svelte 前端不参与编译；
> 2. 该目录里有 `chat/[id]/+page.svelte` 这类含 `[` 的文件名，**Kaggle 数据集会拒收整个包**
>    （`contains a forbidden character in name ('[')`）。
>
> `pack_source.py` 现在还会对任何含 `[` `]` `?` `*` `:` `<` `>` `|` `"` 的路径直接报错退出，
> 避免再产出被 Kaggle 拒收的压缩包。
