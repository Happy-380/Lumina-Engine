#!/usr/bin/env python3
"""Generate kaggle/Lumina-Engine-Kaggle-Train.ipynb.

Kept in the repo so the notebook can be regenerated after edits to the shell
scripts (the notebook itself is the deliverable that gets uploaded to Kaggle).

    python kaggle/tools/make_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "Lumina-Engine-Kaggle-Train.ipynb"

MD = "markdown"
CODE = "code"

cells: list[tuple[str, str]] = []


def md(src: str) -> None:
    cells.append((MD, src.strip("\n")))


def code(src: str) -> None:
    cells.append((CODE, src.strip("\n")))


# --------------------------------------------------------------------------- 0
md(r"""
# Lumina-Engine · SmallThinker MoE QLoRA 训练（Kaggle / CUDA 版）

把本机的 stage-1 继续预训练（`-c 32768`，`--resume`）搬到 Kaggle 上跑。本 notebook 会：

1. 检查 GPU / CUDA / 内存；
2. 找到你上传的 `LuminaMoE-V1` 数据集里的 **基座模型 / stage1 数据集 / 最新检查点**（自动递归搜索，不依赖解压后的目录层级）；
3. 编译 **CUDA 版** Lumina-Engine（Build 32，llama.cpp b-10991 分支）；
4. 用**与本地完全一致**的参数续训（只改路径），按时间预算自动收尾；
5. 打包产物，并说明下一次会话如何接着跑。

---

### 运行前准备（一次性）

| 需要的东西 | 怎么放 |
|---|---|
| 数据集 `Dataset.zip` | 已在 Kaggle 上传为 dataset，名字 **`LuminaMoE-V1`**（里面含 `Dataset/…` 的基座 gguf、`data/stage1_pt/train.jsonl`、`output/stage1.lora.ckpt-*.gguf`） |
| 引擎源码 `Lumina-Engine-src.zip` | 见下 |
| Notebook 设置 | Accelerator 选 **GPU T4 x2 / P100 / L4**（无所谓，训练只用到 CPU，见下面的“重要说明”）；Internet 可关 |

**源码包怎么上传**（三选一，本 notebook 都会自动找）：

- **推荐**：在 `LuminaMoE-V1` 数据集页面点 *Edit → Add file*，把 `Lumina-Engine-src.zip`（约 10 MB）加进同一个数据集；
- 或者另外建一个 dataset 上传 `Lumina-Engine-src.zip`；
- 或者把 `D:\Lumina-Engine\kaggle\` 整个文件夹作为一个 dataset 上传（脚本 + `src/` 都在里面）。

> 如果都找不到、且会话开了 Internet，最后会尝试 `git clone https://github.com/Happy-380/Lumina-Engine`。

---

### 重要说明（请先读）

- **本引擎的反向传播 / 优化器只有 CPU 实现**（`HANDOFF.md` Build 15 的结论：Vulkan/CUDA 都没有反向算子）。
  CUDA 版会正常编译、`cli` 推理可以吃 GPU，但 `Lumina-Engine-train` 默认仍强制纯 CPU 训练
  （日志里会打印 `restricting device list to CPU only`）。这是**正确性优先**的默认值。
- 因此 **训练速度 ≈ Kaggle 的 4 vCPU**，比本机 12 线程更慢；`-c 32768` 下每个微批是分钟级。
  全量 epoch（约 2.6M 微批）在 Kaggle 上不可能一次跑完 —— 本 notebook 的设计是
  **每 11 小时收一次，靠 `--save-steps 50` + `--resume` 跨会话接续**。
- 想试 GPU 训练：`ALLOW_GPU = "1"`（第 6 格），会加 `-ngl 99` + `LUMINA_TRAIN_ALLOW_GPU=1`。
  这是**实验性**的：不支持的算子会回退 CPU，可能更慢，也可能算错，务必先跑第 8 格的 A/B 校验。
- 检查点/上下文/stride **必须与本地一致**：`-c 32768 -b 2048 -ub 128`，`--dataset-stride` 用默认
  （`n_ctx/2`）。改了任何一个，`steps_per_epoch` 变化，`ckpt-10400` 的续训位置就对不上了。
""")

# --------------------------------------------------------------------------- 1
code(r"""
#@title 1. 环境检查 { vertical-output: true }
import os, re, shutil, subprocess, sys, time, platform
from pathlib import Path

def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return (r.stdout or r.stderr).strip()
    except Exception as e:
        return f"<{e}>"

print("python :", sys.version.split()[0])
print("os     :", platform.platform())
print("cpu    :", sh("nproc"), "vCPU |", sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip())
print("ram    :", sh("free -g | awk '/^Mem:/{print $2\" GiB total, \"$7\" GiB available\"}'"))
print("disk   :", sh("df -h /kaggle/working | tail -1"))
print("cmake  :", sh("cmake --version | head -1") or "not found")
print("ninja  :", sh("ninja --version") or "not found")
print("g++    :", sh("g++ --version | head -1") or "not found")
print("nvcc   :", sh("nvcc --version | tail -1") or ("/usr/local/cuda/bin/nvcc --version | tail -1" if os.path.exists("/usr/local/cuda/bin/nvcc") else "not found"))
print("gpu    :", sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader") or "no GPU visible")
try:
    import torch
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    print("torch  :", torch.__version__, "| cuda:", torch.cuda.is_available(), "| compute capability:", cc)
except Exception as e:
    print("torch  : unavailable (", e, ")")

if not shutil.which("nvcc") and os.path.exists("/usr/local/cuda/bin/nvcc"):
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    print("\n已把 /usr/local/cuda/bin 加入 PATH")
""")

# --------------------------------------------------------------------------- 2
md(r"""
## 2. 定位脚本与源码

把所有 `kaggle/` 下的脚本复制到 `/kaggle/working/lumina/run`（`/kaggle/input` 只读），
顺便准备好后面各步要用的路径。
""")

code(r"""
#@title 2. 定位脚本 / 源码 { vertical-output: true }
import json, os, re, shutil, subprocess, sys, time, zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK  = Path("/kaggle/working/lumina")
RUN_DIR = WORK / "run"
LOG_DIR = WORK / "logs"
for d in (WORK, RUN_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

def find_in_input(name):
    hits, seen = [], set()
    for root, dirs, files in os.walk(INPUT, followlinks=False):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules", "site-packages")]
        if name in files:
            p = Path(root) / name
            if str(p) not in seen:
                seen.add(str(p))
                hits.append(p)
    return hits

# ---- helper: run a command, stream output live, tee to a log file ----------
_dec = __import__("codecs").getincrementaldecoder("utf-8")("replace")

def _env(extra=None):
    e = os.environ.copy()
    e.update({k: str(v) for k, v in (extra or {}).items()})
    return e

def run_stream(cmd, extra_env=None, log=None, cwd=None):
    # echo the command output live (handles \r progress bars) and tee it to `log`
    print("$", " ".join(str(c) for c in cmd), flush=True)
    fh = open(log, "a", encoding="utf-8", errors="replace") if log else None
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, cwd=cwd, env=_env(extra_env), bufsize=0)
    buf = ""
    while True:
        chunk = p.stdout.read(4096)
        if not chunk:
            break
        text = _dec.decode(chunk)
        if fh:
            fh.write(text); fh.flush()
        buf += text
        parts = re.split(r"[\r\n]", buf)
        buf = parts.pop()
        for line in parts:
            if line.strip():
                print(line.rstrip(), flush=True)
    if buf.strip():
        print(buf.rstrip(), flush=True)
    rc = p.wait()
    if fh: fh.close()
    print(f"[exit {rc}]", flush=True)
    return rc

def run_quiet(cmd, extra_env=None, log=None, tail=30, heartbeat=20):
    # run a command writing its output to `log`; print a heartbeat + the tail at the end
    log = Path(log) if log else (LOG_DIR / "build.log")
    print("$", " ".join(str(c) for c in cmd), f"\n  (log: {log})", flush=True)
    fh = open(log, "a", encoding="utf-8", errors="replace")
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, env=_env(extra_env), bufsize=0)
    last_print, alive, text = time.time(), True, ""
    while True:
        chunk = p.stdout.read(4096)
        if not chunk:
            break
        t = _dec.decode(chunk)
        fh.write(t); fh.flush()
        text = (text + t)[-8000:]
        if time.time() - last_print > heartbeat:
            lines = [l for l in re.split(r"[\r\n]", text) if l.strip()]
            print(f"  … {time.time()-last_print:4.0f}s  {lines[-1][:160] if lines else ''}", flush=True)
            last_print = time.time()
    rc = p.wait()
    fh.close()
    lines = [l for l in re.split(r"[\r\n]", text) if l.strip()]
    print("\n".join(lines[-tail:]), flush=True)
    print(f"[exit {rc}]", flush=True)
    return rc

# ---- scripts ---------------------------------------------------------------
# Preferred: the bundle zip (or the kaggle/ folder) was extracted by Kaggle.
paths_py = find_in_input("kaggle_paths.py")
print("kaggle_paths.py :", [str(p) for p in paths_py] or "not found in /kaggle/input")

SCRIPT_SRC = paths_py[0].parent if paths_py else None

if SCRIPT_SRC is None:
    # Lumina-Engine-src.zip holds both the engine source and these scripts, but
    # Kaggle does not always auto-extract a zip that sits inside a dataset.
    cands, seen = [], set()
    for p in find_in_input("Lumina-Engine-src.zip"):
        cands.append(p); seen.add(str(p))
    for root, dirs, files in os.walk(INPUT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for f in files:
            if not f.lower().endswith(".zip"):
                continue
            z = Path(root) / f
            if str(z) in seen or z.stat().st_size < (1 << 20):
                continue
            try:
                with zipfile.ZipFile(z) as zf:
                    if any(n.endswith("kaggle_paths.py") for n in zf.namelist()):
                        cands.append(z); seen.add(str(z))
            except Exception:
                continue
    for z in sorted(cands, key=lambda p: len(str(p))):
        print("extracting bundle:", z, "->", WORK / "src")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(WORK / "src")
        hit = list((WORK / "src").rglob("kaggle_paths.py"))
        if hit:
            SCRIPT_SRC = hit[0].parent
            break

if SCRIPT_SRC is None:
    print("没有找到脚本包，尝试 git clone（需要 Internet）…")
    run_quiet(["git", "clone", "--depth", "1",
               "https://github.com/Happy-380/Lumina-Engine", str(WORK / "clone")],
              log=LOG_DIR / "clone.log")
    hit = list((WORK / "clone").rglob("kaggle_paths.py"))
    SCRIPT_SRC = hit[0].parent if hit else None

if SCRIPT_SRC is None:
    raise RuntimeError("找不到 kaggle_paths.py / Lumina-Engine-src.zip。"
                       "请把 Lumina-Engine-src.zip 加入 LuminaMoE-V1 数据集后重新运行本格。")

shutil.copytree(SCRIPT_SRC, RUN_DIR, dirs_exist_ok=True)
print("scripts       :", SCRIPT_SRC, "->", RUN_DIR)

# ---- engine source ---------------------------------------------------------
ENV = dict(LUMINA_WORK=str(WORK), LUMINA_INPUT_DIRS="/kaggle/input")
src_dir = None
for cand in (SCRIPT_SRC.parent / "llama.cpp-b10991",
             SCRIPT_SRC / "src" / "llama.cpp-b10991",
             SCRIPT_SRC.parent.parent / "llama.cpp-b10991"):
    if (cand / "CMakeLists.txt").is_file():
        src_dir = cand
        break
if src_dir is None:
    hit = [p for p in find_in_input("CMakeLists.txt") if p.parent.name == "llama.cpp-b10991"]
    if hit:
        src_dir = hit[0].parent

if src_dir is not None:
    ENV["LUMINA_SRC"] = str(src_dir)
    print("engine source :", src_dir)
else:
    zips = find_in_input("Lumina-Engine-src.zip")
    if zips:
        ENV["LUMINA_SRC"] = str(sorted(zips, key=lambda p: len(str(p)))[0])
        print("engine source :", ENV["LUMINA_SRC"], "(build_cuda.sh 会解压)")
    else:
        print("engine source : NOT FOUND — 编译会失败，请把 Lumina-Engine-src.zip 加入数据集")

print("work dir      :", WORK)
""")

# --------------------------------------------------------------------------- 3
md(r"""
## 3. 定位基座模型 / 数据集 / 检查点

不假设解压结构：在 `/kaggle/input` 下递归找

- 基座模型：`*.gguf` 且 ≥ 200 MiB、名字里不是 LoRA/ckpt（优先 `SmallThinker`）；
- 训练集：`train.jsonl`（优先路径含 `stage1_pt`，且同目录有 `val.jsonl`）；
- 检查点：`<stem>.ckpt-<step>.gguf`，取 **step 最大** 的那个（`stage1` 优先）。

如果数据集上传成了未被解压的 `Dataset.zip`，会自动解压到 `/kaggle/working/lumina/input`。
""")

code(r"""
#@title 3. 输入定位报告 { vertical-output: true }
rc = run_stream([sys.executable, str(RUN_DIR / "kaggle_paths.py")], extra_env=ENV,
                log=LOG_DIR / "paths.log")
if rc != 0:
    raise SystemExit("输入没有找齐，请检查 LuminaMoE-V1 数据集内容（见上方报告）")
""")

# --------------------------------------------------------------------------- 4
md(r"""
## 4. 编译 CUDA 版

- `GGML_CUDA=ON`，`CMAKE_CUDA_ARCHITECTURES` 按当前 GPU 的 compute capability 自动选（T4 → 75，P100 → 60，L4 → 89 …）；
- 静态链接，产物只有两个可执行文件，方便把 `/kaggle/working` 存成 dataset 复用；
- 4 vCPU 上首次约 **20–60 分钟**；`/kaggle/input` 里若有上次会话保存的 `Lumina-Engine-train`，会自动复用、跳过编译；
- CUDA 配置失败会自动回退 CPU-only 构建（训练本来就在 CPU 上跑，功能不受影响）。
""")

code(r"""
#@title 4. 编译（或复用已编译产物）
USE_CACHED_BIN = "auto"   # "auto" | "never"  —— "auto": 复用 /kaggle/input 里的旧产物
LUMINA_JOBS    = 0        # 0 = nproc

BIN_DIR = WORK / "bin"
TRAINER = BIN_DIR / "Lumina-Engine-train"
BIN_DIR.mkdir(parents=True, exist_ok=True)

def binary_ok(p):
    try:
        r = subprocess.run([str(p), "--version"], capture_output=True, text=True, timeout=180)
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode == 0 and "Build 32" in out
    except Exception:
        return False

cached = None
if USE_CACHED_BIN == "auto":
    for p in find_in_input("Lumina-Engine-train"):
        if binary_ok(p):
            cached = p
            break

if cached is not None:
    print("复用已编译产物:", cached)
    shutil.copy2(cached, TRAINER)
    os.chmod(TRAINER, 0o755)
    print(subprocess.run([str(TRAINER), "--version"], capture_output=True, text=True).stdout)
else:
    env = dict(ENV)
    env["LUMINA_BIN_DIR"] = str(BIN_DIR)
    env["LUMINA_JOBS"] = str(LUMINA_JOBS or (os.cpu_count() or 4))
    env["LUMINA_CUDA"] = "1"
    rc = run_quiet(["bash", str(RUN_DIR / "build_cuda.sh")], extra_env=env,
                   log=LOG_DIR / "build.log", tail=40, heartbeat=30)
    if rc != 0 or not TRAINER.exists():
        raise SystemExit("编译失败，请看上面的日志尾部 / logs/build.log")

print("\n=== build-info ===")
info = BIN_DIR / "build-info.txt"
print(info.read_text() if info.exists() else "(no build-info.txt)")
""")

# --------------------------------------------------------------------------- 5
md(r"""
## 5. 训练命令预览

这一步只把最终命令行打印出来（不执行），用来对照：

```
Lumina-Engine-train -m <基座模型> --dataset <stage1_pt/train.jsonl> --dataset-val <.../val.jsonl> \
  -c 32768 -b 2048 -ub 128 --cpu-cores 0 -lr 1e-4 --epochs 1 -lora-rank 64 -lora-alpha 128 \
  --save-steps 50 --save-every-epochs 1 --keep-last 3 -o <out>/stage1.lora.gguf --resume
```

和本机命令相比**只改了路径**：超参、上下文、stride、检查点命名全部一致，
所以 `ckpt-10400` 能被正确续接（`--resume` 会打印 `resuming at datapoint N/M …`）。
""")

code(r"""
#@title 5. 命令预览（dry-run）
env = dict(ENV)
env["LUMINA_BIN"] = str(TRAINER)
rc = run_stream(["bash", str(RUN_DIR / "run_train.sh"), "--dry-run"], extra_env=env,
                log=LOG_DIR / "dryrun.log")
assert rc == 0, "dry-run 失败"
""")

# --------------------------------------------------------------------------- 6
md(r"""
## 6. 开始训练

- `TIME_BUDGET_HOURS`：本次会话最多训练多久。会话上限 12 h，留出编译/打包时间；
  时间到了用 `timeout` 停掉，**最后一个检查点之后（≤ 50 个微批）的进度会丢**。
- 想跑 GPU 实验：`ALLOW_GPU = "1"`（会加 `-ngl 99` + `LUMINA_TRAIN_ALLOW_GPU=1`，风险见开头）。
- 断点信息：检查点被复制到 `/kaggle/working/lumina/train_out/`，`--resume` 从那里取最新的
  `stage1.lora.ckpt-*.gguf`，`--keep-last 3` 留 3 个（约 200 MB）。

> 第一次运行时，日志里应出现（数值取决于 ckpt）：
> `resuming from checkpoint '…/stage1.lora.ckpt-10400.gguf' (epoch 1, step 10400)`
> `resuming at datapoint 40/10159 of epoch 1, micro-batch 160/256 (global step 10400)`
""")

code(r"""
#@title 6. 训练 { vertical-output: true }
TIME_BUDGET_HOURS = 11.0     # 0 = 不限制（不推荐，会话会被硬杀）
ALLOW_GPU         = "0"      # "1" = 尝试 GPU offload（实验性）
RESUME            = "1"

budget_s = int(TIME_BUDGET_HOURS * 3600) if TIME_BUDGET_HOURS else 0
env = dict(ENV)
env.update({
    "LUMINA_BIN": str(TRAINER),
    "LUMINA_RESUME": RESUME,
    "LUMINA_ALLOW_GPU": ALLOW_GPU,
    "LUMINA_TIME_BUDGET": str(budget_s),
    "LUMINA_PYTHON": sys.executable,
})
if budget_s:
    print(f"训练预算 {TIME_BUDGET_HOURS} h，预计 {time.strftime('%H:%M', time.localtime(time.time()+budget_s))} 前后停止（UTC {time.strftime('%H:%M', time.gmtime(time.time()+budget_s))}）")
print("(Kaggle 会在 ~12 h 硬性回收会话，请留足余量)\n", flush=True)

LOG = LOG_DIR / f"train-{time.strftime('%Y%m%d-%H%M%S')}.log"
rc = run_stream(["bash", str(RUN_DIR / "run_train.sh")], extra_env=env, log=LOG)
print("\nrc =", rc, "(124/137 = 到达时间预算，正常)")

# checkpoint progress of this session
steps = []
for p in (WORK / "train_out").glob("*.ckpt-*.gguf"):
    m = re.search(r"\.ckpt-(\d+)$", p.stem)
    if m:
        steps.append(int(m.group(1)))
print("checkpoint steps now:", sorted(steps) or "(none)")
""")

# --------------------------------------------------------------------------- 7
md(r"""
## 7. 产物

`/kaggle/working` 下的文件会出现在 notebook 的 **Output** 面板，可以逐个下载；
如果 Output 太大，可以只下载本格打出的 `stage1-latest.zip`（最新检查点 + 当前适配器）。
""")

code(r"""
#@title 7. 列出并打包产物
import zipfile
from pathlib import Path

OUT = WORK / "train_out"
DL  = WORK / "download"
DL.mkdir(parents=True, exist_ok=True)

files = sorted(OUT.glob("*")) if OUT.exists() else []
for f in files:
    print(f"{f.stat().st_size/2**20:9.1f} MiB  {f.name}")

def step_of(p):
    m = re.search(r"\.ckpt-(\d+)$", p.stem)
    return int(m.group(1)) if m else -1

ck = sorted([f for f in files if f.suffix == ".gguf" and ".ckpt-" in f.name], key=step_of)
final = OUT / "stage1.lora.gguf"

zpath = DL / "stage1-latest.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:   # gguf 不可压缩，直接 store
    picked = []
    if ck:
        picked.append(ck[-1])
        js = ck[-1].with_suffix(".json")
        if js.exists():
            picked.append(js)
    if final.exists():
        picked.append(final)
    for f in picked:
        zf.write(f, f.name)
print("\n打包:", zpath, f"({zpath.stat().st_size/2**20:.1f} MiB)")
print("内容:", [f.name for f in picked])

print()
print("下一步（重要）：")
print("  · 下载 stage1-latest.zip，下次把里面的 ckpt-*.gguf / .json 放回 LuminaMoE-V1 数据集")
print("    （或把本次 notebook 的 Output 保存成 dataset 并挂载），本 notebook 会自动选最新 step；")
print("  · 编译产物在 /kaggle/working/lumina/bin，也可以在 Output 里保存成 dataset 复用，省掉 20–60 分钟编译。")
""")

# --------------------------------------------------------------------------- 8
md(r"""
## 8. （可选）GPU 训练 A/B 校验

引擎的反向/优化器是 CPU 实现，GPU 只能加速一部分前向算子。这个格子用**同一份小数据、
同一超参**分别跑 CPU 与 GPU，比较 loss 序列：

- 两次 loss 一致（差 < 1e-2）→ GPU 路径结果可信，再看耗时是否真的更快；
- loss 不一致或明显更快但 loss 变差 / 运行报错 → 保持 CPU 训练（默认）。

只在 `-c 1024 -b 1024 -ub 128` 的小样本上跑，几分钟即可，不会影响正式训练。
""")

code(r"""
#@title 8. GPU / CPU 小样本 A/B（可选）
RUN_AB = False          # True 才执行（几分钟）
AB_LIMIT = 128          # 记录数
AB_BUDGET = 300         # 每组最多秒数

if RUN_AB:
    info = json.loads(subprocess.run(
        [sys.executable, str(RUN_DIR / "kaggle_paths.py"), "--json"],
        capture_output=True, text=True, env=_env(ENV)).stdout)
    ab_dir = WORK / "ab"
    ab_dir.mkdir(parents=True, exist_ok=True)
    for name, gpu in (("cpu", False), ("gpu", True)):
        cmd = [str(TRAINER), "-m", info["MODEL"], "--dataset", info["TRAIN"],
               "--dataset-limit", str(AB_LIMIT),
               "-c", "1024", "-b", "1024", "-ub", "128", "--cpu-cores", "0",
               "-lr", "1e-4", "--epochs", "1", "-lora-rank", "8", "-lora-alpha", "16",
               "--save-steps", "100000", "--save-every-epochs", "0",
               "-o", str(ab_dir / f"ab-{name}.lora.gguf")]
        if gpu:
            cmd += ["-ngl", "99"]
        if shutil.which("timeout"):
            cmd = ["timeout", str(AB_BUDGET)] + cmd
        env = dict(ENV)
        if gpu:
            env["LUMINA_TRAIN_ALLOW_GPU"] = "1"
        print(f"\n===== {name} =====", flush=True)
        run_stream(cmd, extra_env=env, log=LOG_DIR / f"ab-{name}.log")

    print("\n=== summary ===")
    for name in ("cpu", "gpu"):
        p = LOG_DIR / f"ab-{name}.log"
        txt = p.read_text(errors="replace") if p.exists() else ""
        losses = re.findall(r"loss=([0-9.]+)", txt)
        print(f"{name:4s}: {len(losses):3d} 次, 前几个 loss = {losses[:6]}")
    print("两次 loss 基本一致 -> GPU 路径数值可信；差异大或报错 -> 请保持 CPU 训练")
else:
    print("RUN_AB = False，跳过")
""")

# --------------------------------------------------------------------------- 9
md(r"""
## 9. 下一次会话怎么继续

1. **复用编译产物**：把本次 Output 里的 `lumina/bin/` 保存成 dataset（*New Dataset → From notebook output*），
   下次把它加进输入；第 4 格会打印 `复用已编译产物: …` 并跳过编译。
2. **接续训练**：检查点按 `stage1.lora.ckpt-<step>.gguf` 命名，`--resume` 自动选最大 step。
   - 方式 A：把 `stage1-latest.zip` 里的 ckpt 放回 `LuminaMoE-V1` 数据集；
   - 方式 B：把每次的 Output 保存成 dataset 挂载（里面有 `train_out/`）。
3. 重复运行第 3 → 6 格即可。每次会话都会打印 `resuming at datapoint N/M`，`N` 只会增大。

### 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `输入没有找齐` | `LuminaMoE-V1` 里没有 `*.gguf`(≥200 MiB) 或 `train.jsonl`；看第 3 格的报告 |
| 报 `training data is too short` | 数据集不是 stage1（别把 stage3 的人设集指过来） |
| 编译很慢 / 失败 | 看 `/kaggle/working/lumina/logs/build.log`；失败会自动回退 CPU-only |
| `--resume` 提示 `no checkpoint found` | ckpt 没复制进 `train_out/`：确认数据集里有 `stage1.lora.ckpt-*.gguf` 和同名 `.json` |
| `resuming at datapoint 0/…`（明明有 ckpt） | 训练配置与本地不一致（`-c`/`-b`/`-ub`/`--dataset-stride`/数据集文件），导致 `steps_per_epoch` 变了 |
| loss 一直是 `nan` | 学习率过大或数据异常；用 `LUMINA_LR=5e-5` 试 |
| loss 持续上升 / 不收敛 | 确认用的是 Build 32 源码包（或任何 >= Build 26 的包）：Build 26 之前的训练梯度累加器从不清零，loss 会线性攀升 |
| 想要更快 | 本机/云端换更多核；或把 `-c 32768` 降到 `-c 16384`（**会改变语义、旧 ckpt 失效，需要从基座重训**） |
""")

# --------------------------------------------------------------------------- build
nb = {
    "cells": [
        {
            "cell_type": t,
            "metadata": {},
            "source": (src + "\n").splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if t == CODE else {}),
        }
        for t, src in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isGpuEnabled": True,
                   "isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells, {OUT.stat().st_size/1024:.1f} KiB)")
