# 压缩模型 & 使用压缩模型（用户指南）

> 🌐 **中文** | [English](ModelCompression.en.md)

本页只讲**怎么用**：如何把一个 GGUF 模型无损压小，以及如何用 Lumina-Engine 加载压缩后的模型。
原理与容器格式见 [LosslessCompression](LosslessCompression.md)。

一句话：**用 `tools/lumina_gguf_pack.py` 压小（默认产出 `A.gguf.lumina`，原文件不动），Lumina-Engine（Build 8+）自动识别并加载 `.gguf` 与 `.gguf.lumina`，权重逐字节不变。**

---

## 0. 适用条件

| 项目 | 要求 |
|------|------|
| 系统 | Windows 10 / 11 x86-64 |
| Python | 3.8+，并安装 `numpy` |
| 引擎 | **Lumina-Engine Build 8 或更新**（`Lumina-Engine-Bin\`） |
| 模型张量类型 | `F32`、`Q4_K`、`Q6_K`（本仓库的 `SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf` 即在此列） |

> 压缩是**离线一次性**操作，不需要 GPU；推理时的速度、内存、精度与未压缩模型完全一致。

---

## 1. 压缩模型

### 1.1 安装依赖

```bat
python -m pip install numpy
```

### 1.2 打包

默认输出为 `<输入>.lumina`（即 `A.gguf` → `A.gguf.lumina`），**原文件保持不变**：

```bat
python tools\lumina_gguf_pack.py  <输入.gguf>
REM 等价于： python tools\lumina_gguf_pack.py <输入.gguf> <输入.gguf>.lumina
```

也可以显式指定输出路径：

```bat
python tools\lumina_gguf_pack.py  <输入.gguf>  <输出.gguf.lumina>
```

示例（本仓库实际模型）：

```bat
python tools\lumina_gguf_pack.py SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf
```

> ⚠️ **注意**：本仓库根目录**同时保留两份**：`SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf`
> 是**原始未压缩**模型，`SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf.lumina` 是**压缩容器**。
> 请对原始 `.gguf` 打包；对压缩容器直接打包会报错 `input is already a Lumina compressed GGUF`。

输出示例：

```
packed SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf
  original : 2630213408 bytes (2.45 GiB)
  packed   : 2479779960 bytes (2.31 GiB)
  ratio    : 94.28%  (saved 143.5 MiB)
  output   : SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf.lumina
  original left untouched; load either file directly with Lumina-Engine.
```

可选第三个参数为 zlib 压缩级别（`1`–`9`，默认 `9`，越高越小越慢）：

```bat
python tools\lumina_gguf_pack.py model.gguf model.custom.lumina 9
```

### 1.3 校验（强烈建议）

确认解压结果与原始文件**逐字节一致**（SHA-256）：

```bat
python tools\lumina_gguf_unpack.py verify  <压缩容器.gguf.lumina>  <原始.gguf>
```

期望输出：

```
verify: OK fb584d0b0f79f9e9944f91ac061e24726a8d014b6702213bfd5585729db7ab61 fb584d0b...
```

`OK` 表示无损；`FAIL` 表示文件损坏，请勿使用。

### 1.4 原文件与压缩文件的取舍（可选）

默认打包**不会改动原文件**，因此两份并存、按需使用即可：

```
model.gguf          ← 原始模型（官方 llama.cpp / 其它工具也可直接读取）
model.gguf.lumina   ← 压缩容器（仅 Lumina-Engine 识别，靠魔数自动加载）
```

若确实想只保留一份，可先校验再替换（注意替换后普通工具将无法直接读取）：

```bat
python tools\lumina_gguf_unpack.py verify model.gguf.lumina model.gguf
move /y model.gguf model.gguf.bak          REM 或直接删除
move /y model.gguf.lumina model.gguf       REM 可选：让引擎仍从 .gguf 路径加载
```

> 引擎**只认文件头的 `LGGC` 魔数，不依赖扩展名**：`.gguf` 与 `.gguf.lumina` 均可直接加载。

---

## 2. 使用压缩后的模型

**不需要任何额外参数**：把压缩文件（`.gguf.lumina`）当成普通模型路径传给引擎即可。首次加载时引擎会自动解压并缓存。原始 `.gguf` 也照常可用。

```bat
REM 加载压缩容器（推荐）
Lumina-Engine-cli.exe -m model.gguf.lumina -p "Explain quantum computing simply." -n 256 -t 6

REM 加载原始 .gguf 亦可（引擎靠魔数兼容两种扩展名）
Lumina-Engine-cli.exe -m model.gguf -p "Explain quantum computing simply." -n 256 -t 6

REM 基准测试 / 困惑度 / 服务同样直接传路径
Lumina-Engine-bench.exe       -m model.gguf.lumina -p 512 -n 128 -t 6
Lumina-Engine-perplexity.exe  -m model.gguf.lumina -f docx\whitepaper.txt -c 512 -t 6
Lumina-Engine-server.exe      -m model.gguf.lumina -c 4096
```

### 加载时发生了什么

1. 引擎读取文件头，发现 `LGGC` 魔数 → 判定为无损压缩容器；
2. 流式解压，按张量重建出**与原始文件完全相同**的 GGUF；
3. 写到系统临时目录：`%TEMP%\lumina-<hash>.gguf`；
4. 之后走原有加载/推理路径。临时文件按路径+大小命名，**再次加载会直接复用**，不会重复解压。

### 磁盘与耗时

| 项目 | 说明 |
|------|------|
| 首次加载 | 需要一次性写出临时文件（约等于原始模型大小）。本机 2.63 GB 约 30 秒。 |
| 后续加载 | 直接复用临时文件，无额外耗时。 |
| 清理 | 可随时删除 `%TEMP%\lumina-*.gguf`，下次加载会自动重建。 |

想看解压日志可设置环境变量 `LUMINA_VERBOSE=1`。

> 若磁盘空间紧张：可先压缩替换原文件（省下 ~143.5 MiB），但首次运行时临时目录仍需 ~2.63 GB 可用空间。

---

## 3. 原理（一句话）

k-quant 的每个 256 权重块内混有低熵的 `d/dmin/scales` 和高熵的 4/6-bit 权重流。打包器把它们
**分字段拆开**（并把 Q4_K 的量化值按 `scale`/`min` 上下文稳定排序）再 DEFLATE，避免互相干扰；
加载时逆向合并。**不量化、不剪枝**，权重值不变。

本模型实测：`2,630,213,408 B → 2,479,779,960 B`（**94.28%**，省 **143.5 MiB**），解压后 SHA-256 一致。

---

## 4. 常见问题（FAQ）

**Q1：压缩后的文件能直接用官方 llama.cpp / 其它工具打开吗？**
不能。它是 Lumina 私有容器（魔数 `LGGC`），只有 **Build 8+ 的 Lumina-Engine** 能透明加载。
需要原始 GGUF 时用 `lumina_gguf_unpack.py` 还原。

**Q2：会损失精度/影响输出吗？**
不会。解压结果与原文件**逐字节相同**（可用 `verify` 校验），输出与未压缩模型一字不差。

**Q3：为什么只小了百分之几？**
权重已经量化过：Q6_K 量化值接近满熵（~6 bit），Q4_K 量化值的条件熵也约 3.82/4 bit，
无损压缩的理论上限就只有几个百分点。v2 已把 Q4_K 流压到 95.5%（接近其条件熵上限），
因此在不量化、不剪枝的前提下无法再小很多。

**Q4：支持所有模型吗？**
目前打包器支持张量类型 `F32` / `Q4_K` / `Q6_K`。其它类型会报错；如需支持更多类型，扩展
`tools/lumina_gguf_pack.py` 的 `split_tensor` 与对应解码即可。

**Q5：临时文件在哪？安全吗？**
`%TEMP%\lumina-<hash>.gguf`，由引擎自动生成，可随时删除；不会修改你的模型文件。

**Q6：压缩比原文件还大怎么办？**
极小模型（如只有几 MB 的词表文件）可能出现这种情况，因为容器头本身有开销。此时直接用原文件即可。

**Q7：打包很慢？**
纯 Python + zlib，2.6 GB 模型约 3 分钟（主要耗时在 DEFLATE 与上下文排序）。可用 `level 6` 换取更快速度。

**Q8：为什么用 `.gguf.lumina` 后缀？**
便于一眼区分原始模型与压缩容器。引擎**按魔数识别，后缀只是约定**：`.gguf` 和 `.gguf.lumina`
都能直接加载；用其它后缀（如 `.bin`）同样可以，只是不推荐。

---

## 5. 一键脚本（可选）

在仓库根目录新建 `compress_model.bat`：

```bat
@echo off
setlocal
if "%~1"=="" ( echo usage: compress_model.bat ^<model.gguf^> & exit /b 1 )
python tools\lumina_gguf_pack.py "%~1"
if errorlevel 1 ( echo [compress_model] pack FAILED & exit /b 1 )
python tools\lumina_gguf_unpack.py verify "%~1.lumina" "%~1"
if errorlevel 1 ( echo [compress_model] verify FAILED & exit /b 1 )
echo [compress_model] done: "%~1.lumina"   (original kept)
endlocal
```

用法：`compress_model.bat SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf`

---

## 6. 附：脚本命令行速查

所有脚本都是纯 Python（仅需 `numpy`），在仓库根目录运行。

### 6.1 打包（压缩）

```bat
python tools\lumina_gguf_pack.py <输入.gguf> [输出] [level]
```

| 参数 | 说明 |
|------|------|
| `<输入.gguf>` | 原始未压缩 GGUF；若已是 `LGGC` 容器会报错 |
| `[输出]` | 可选；默认 `<输入>.lumina`（`A.gguf` → `A.gguf.lumina`），**原文件保留** |
| `[level]` | 可选；zlib 级别 `1`–`9`，默认 `9`（越大越小越慢） |

示例：

```bat
python tools\lumina_gguf_pack.py model.gguf
python tools\lumina_gguf_pack.py model.gguf model.custom.lumina 6
```

### 6.2 还原 / 校验（解包）

```bat
REM 还原为普通 GGUF 文件
python tools\lumina_gguf_unpack.py <压缩容器.gguf.lumina> <输出.gguf>

REM 逐字节校验（推荐）：输出 OK / FAIL，OK 时退出码为 0
python tools\lumina_gguf_unpack.py verify <压缩容器.gguf.lumina> <原始.gguf>
```

示例：

```bat
python tools\lumina_gguf_unpack.py verify model.gguf.lumina model.gguf
```

> 解包器同时兼容容器 v1 与 v2。

### 6.3 检查 GGUF 元数据 / 张量（辅助）

```bat
REM 打印 GGUF 元数据键值
python tools\read_gguf.py <模型.gguf>

REM 打印每个张量的名称 / 类型 / 维度，以及类型计数
python tools\tensors.py <模型.gguf>
```

> 这两个脚本用于查看普通 GGUF（或还原后的文件）；直接读 `LGGC` 容器没有意义，请先
> 用 `lumina_gguf_unpack.py` 还原。

---

## 相关页面

- [LosslessCompression](LosslessCompression.md) — 容器格式与内部原理
- [Building](Building.md) — 从源码编译（含 Build 8）
- [Delivery](Delivery.md) — 交付目录 `Lumina-Engine-Bin`
