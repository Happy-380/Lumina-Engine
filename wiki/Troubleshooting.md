# 常见问题与排障

> 🌐 **中文** | [English](Troubleshooting.en.md)

## 性能

### Q: 为什么 tg（生成）没有变快？

先确认二值查表（Q1LUT）是否生效：**默认启用**，受控 A/B 约 **+30%**（~9.8 → ~13.0 t/s）。
若未提速，检查是否设置了 `GGML_Q1_LUT=0`、CPU 是否有 AVX2、DLL 是否为新版。

> ⚠️ 本机为**单通道内存**（8 GB DDR4-2666，实测流式读带宽约 14–17 GB/s）。
tg 终究受**内存带宽**约束：每生成一个 token 要读完约 1.06 GB 的 Q1_0 权重，理论上限约
15.7 tok/s；Q1LUT 已把内核推到接近该上限。双通道/更高带宽平台上 tg 会更高。
这与模型发布方白皮书的描述一致：1-bit 主要降低 tg 的带宽需求，而不是提升 CPU 的算术吞吐。

**进一步提高 tg 的办法**：换双通道内存、关闭后台负载、使用更多物理核（不超过物理核数）。

### Q: 为什么 pp512 只提升 28%，而 pp32 提升 42%？

分块内核的收益来自“符号展开在 4 列间复用”。列数越少，复用的绝对次数越少，
但通用路径在这些小 batch 下本身就慢（16×16 分块的固定开销占比更高），所以**相对**提升更大。
绝对吞吐仍然是 batch 越大越高。

### Q: 怎么确认分块路径真的启用了？

```bat
set GGML_NO_Q1_TILE=1
Lumina-Engine-bench.exe -m Bonsai-8B.gguf -p 128 -n 1 -t 6 -r 2
set GGML_NO_Q1_TILE=
Lumina-Engine-bench.exe -m Bonsai-8B.gguf -p 128 -n 1 -t 6 -r 2
```

如果两者明显不同（应差 ~25%），说明分块路径在工作。

### Q: tg（生成）怎么确认 GEMV 路径启用了？

```bat
set GGML_NO_Q1_GEMV=1
Lumina-Engine-bench.exe -m Bonsai-8B.gguf -p 64 -n 128 -t 12 -r 3
set GGML_NO_Q1_GEMV=
Lumina-Engine-bench.exe -m Bonsai-8B.gguf -p 64 -n 128 -t 12 -r 3
```

12 线程下新版应快约 13%；6 线程受单通道带宽限制，差异较小。
行分组大小可用 `GGML_Q1_GEMV_ROWS`（1/2/4，默认 1）调整。

### Q: 结果波动很大？

本机为办公机，性能随温度/负载漂移。请**同场交替**测量，并增加 `-r` 轮次：
```bat
Lumina-Engine-bench.exe -m Bonsai-8B.gguf -p 128 -n 1 -t 6 -r 5
```

---

## 正确性

### Q: 贪婪解码偶尔与关闭优化时不一致？

分块内核的浮点求和顺序与通用内核不同（先累加 4 个子块的 int32，再统一乘尺度），
会产生 1e-6 量级差异，长文本上可能导致贪婪解码分叉。但：

- **困惑度无损失**（5.7111 vs 5.7160）。
- 两条续写都语义连贯、质量相当。
- 上游的 repack/GEMM 优化路径同样有此现象。

如果必须逐位一致，设置 `GGML_NO_Q1_TILE=1`。

### Q: `iq1_s` / `iq1_m` 的 CPU 测试报错？

这是**上游既有现象**（IQP 面板路径的 CPU-vs-CPU 容差），与本次改动无关：
```bat
set GGML_NO_IQ_PANEL=1
test-backend-ops.exe test -b CPU -o MUL_MAT
```
之后应全部通过。

---

## 编译

### Q: MSVC 只构建了 9 个 CPU 变体？

正常。`ivybridge/piledriver/cooperlake/zen4/sapphirerapids` 需要 Clang（见
[Compatibility](Compatibility.md#2-为什么-msvc-只能构建-9-个变体)）。用 Clang 20 构建 `build-clang` 补齐。

### Q: Clang 报 `error STL1000: Unexpected compiler version, expected Clang 20 or newer`？

VS2026 的 MSVC STL 要求 Clang ≥ 20。请安装 LLVM 20+（本仓库使用 20.1.8）。

### Q: `GGML_CPU_ALL_VARIANTS requires GGML_BACKEND_DL`？

必须同时开 `-DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON`。

### Q: 链接错误 `undefined symbol g_q1p_*`？

那是早期已回退的实验代码残留。请确认 `ggml/src/ggml-cpu/` 下**没有** `q1p.cpp/h`，
且 `ggml-cpu.c` 中不含 `q1p` 引用。

### Q: 修改后如何快速重编？

只重编本机使用的变体：
```bat
cmake --build build-release --target ggml-cpu-haswell -j
```

---

## 运行

### Q: 提示 `unknown model architecture` 或加载失败？

确认模型是 Q1_0（`Lumina-Engine-bench` 的 `ftype` 列应显示 `Q1_0`）。本项目只优化 Q1_0，
其它模型由上游代码支持。

### Q: 内存占用？

Bonsai-8B 权重约 1.07 GiB，加上 KV cache 和运行时，8 GB 内存足够。
pp512 时 Q8_128 scratch 约 6.4 MB，可忽略。二值查表（默认启用）额外约 0.95 GiB；`GGML_Q1_LUT=0` 可释放。

### Q: 能否用 GPU？

本项目的优化只针对 CPU。上游的 CUDA/Metal 后端不受影响，`-ngl` 仍可用。

### Q: 怎么切换 CPU 变体？

运行时自动选择。若要强制，可只保留想要的 `ggml-cpu-*.dll`；或在 CMake 中不用
`GGML_CPU_ALL_VARIANTS`，直接针对本机编译。

---

## 对话 / CLI

### Q: MoE 模型对话时，第二轮及之后只回 `None`？

Build 30 已修。根因：CLI 把每轮 assistant 回复追加进历史时未检查是否为空；只要某一轮没有可见输出
（首 token 即 EOS、流被中断、content 解析为空、请求报错），「空 assistant 轮」就会永久留在历史里，
下一轮请求变成 `[user, assistant(""), user]`，严格 chat template 的 MoE 模型从这轮起每轮只输出占位串。
Build 30 起：空回复/失败的轮次不入历史（打印 `[empty response: nothing was added to the chat history]`）。
若仍在旧版上复现，先 `/clear` 清空历史即可恢复。

### Q: `echo "..." | Lumina-Engine-cli.exe` 或关掉输入管道后进程不退出、一直刷 `> `？

Build 30 已修（空读取 = EOF → 退出，与上游 llama.cpp #29722 一致）。旧版在 stdin 关闭后会无限循环，
只能 Ctrl+C 或 kill。

### Q: 模型的推理过程没有闭合的 `[End thinking]`？

Build 30 起每轮结束都会补 `[End thinking]`（含流中断/报错）；旧版只有「推理 → 正文」的转折才会闭合。

---

## 采样

推荐参数（与模型发布方一致）：

```
--temp 0.5        (0.5 ~ 0.7)
--top-k 20        (20 ~ 40)
--top-p 0.9       (0.85 ~ 0.95)
--repeat-penalty 1.0
```

示例：
```bat
Lumina-Engine-cli.exe -m Bonsai-8B.gguf -p "Write a haiku about autumn." ^
  -n 64 --temp 0.7 --top-k 20 --top-p 0.9 --seed 5 -t 6 -st
```

---

## 诊断清单

| 现象 | 排查 |
|------|------|
| pp 没提升 | `GGML_NO_Q1_TILE`/`GGML_Q1_LUT_PP` 是否被设置；CPU 是否有 AVX2（x64/sse42/sandybridge 无分块/查表） |
| tg 没提升 | `GGML_Q1_LUT`/`GGML_NO_Q1_GEMV`/`GGML_NO_Q1_TILE` 是否被设置；CPU 是否有 AVX2；DLL 是否为新版 |
| tg 很慢 | 带宽瓶颈；检查内存是否双通道；可用 `GGML_Q1_GEMV_ROWS` 调优 |
| 输出乱码 | 是否为 Q1_0 模型；CPU 变体是否匹配；换 `haswell` 变体试 |
| 崩溃/加载失败 | 运行库是否齐全（`libomp140/vcomp140/msvcp140/vcruntime140*`） |
| 结果不稳定 | 同场交替测量并增加 `-r` |

---

## 二值查表（Q1LUT）

### Q: 第一个 token 变慢了？

查表**默认启用**：首次使用某个权重张量时会把它置换为查表码（全模型合计约 0.7 s），并额外分配约
0.95 GiB 内存。这是**一次性**开销，之后每个 token 都使用缓存；长时间生成会被摊薄。若不接受这份
内存/首次延迟，设 `GGML_Q1_LUT=0` 关闭整个查表路径。

### Q: 查表（默认开启）能提速吗？

能。受控交替 A/B（`-p 64 -n 128 -t 6`）：打包 GEMV 9.70/9.99 → 查表 13.04/13.01 t/s（约 **+30%**，
已到单通道带宽上限）；pp 从 `ne11>=8` 起全面快于 2×4 分块 GEMM（pp512 +26%、pp2048 +57%）。
K 分块（`GGML_Q1_LUT_KB`）无收益，保持默认 0。详见
[Optimizations](Optimizations.md#优化-7二值查表q1lut默认启用)。

### Q: 如何确认查表真的生效？

```bat
set GGML_Q1_LUT_DEBUG=1
Lumina-Engine-bench.exe -m Bonsai-8B.gguf -p 64 -n 128 -t 6 -r 2
```
会打印 `q1lut alloc` 行；无输出说明未启用（如 CPU 无 AVX2、`GGML_Q1_LUT=0`，或 `ne01 < 32`）。
