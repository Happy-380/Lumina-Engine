# 主要代码更改

> 🌐 **中文** | [English](Changes.en.md)

本页列出相对于原始 llama.cpp（b10991）的全部改动。Q1LUT（二值查表）部分涉及 **5 个文件**：2 个新增、3 个修改
（全部位于新增的 `q1t.cpp` 内），只影响 `GGML_TYPE_Q1_0` + AVX2 路径；另有 Q4_K 解码内核的预取优化（Build 31，见 §0）与各版 CLI/训练修复。

---

## 0. CPU Q4_K GEMV 软件预取（Build 31）

纯 CPU 单 token 生成的时间几乎全在 `ggml_gemv_q4_K_8x8_q8_K`（repack 后的 Q4_K 解码内核）上。
本机实测该内核单线程只有 6.0–7.5 GB/s，而同一线程纯流式读上限是 11.6–13.1 GB/s —— 说明瓶颈是
**访存延迟没被填满**，不是带宽。改法：在 block 循环开头把 +2 个 `block_q4_Kx8`（1152 B = 18 条
cache line）整块 `_mm_prefetch` 进 L1。**不要加 `if (b + 2 < nb)` 边界判断**（热循环里的分支比
末尾几次无用预取贵得多，实测 -13%；`PREFETCHh` 对无效地址不会 fault）。

| 文件 | 类型 | 说明 |
|------|------|------|
| `ggml/src/ggml-cpu/arch/x86/repack.cpp` | 修改 | `ggml_gemv_q4_K_8x8_q8_K()`：整块预取 +2 个 block（`sizeof` 驱动、编译期展开） |
| `CMakeLists.txt` / `tools/rpc/rpc-server.cpp` | 修改 | Build 30 → 31 |
| `tests/bench-cpu-kern.cpp` | 新增 | 内核级 A/B 基准：真实内核入口 + 独立标量参考 + 同进程交错测量（仅 `LLAMA_BUILD_TESTS=ON` 时编译，不进交付） |
| `tests/q4k_gemv_body.inc` / `tests/CMakeLists.txt` | 新增/修改 | 从 `repack.cpp` 原文抽取的内核体（基线副本）+ 基准目标 |
| `bench/gen_q4k_body.py` / `bench/build_bench_cpu_kern.bat` | 新增 | 生成基线内核体 / 一键构建独立 `build-bench` 树 |

实测（本机 i5-10400 / 8 GB，后台有训练任务，全部按同进程交错比取中位数）：

| 指标 | 改前 | 改后 |
|---|---:|---:|
| 内核单线程带宽（18 MiB 矩阵，DRAM 驻留） | 6.00–6.47 GB/s | 6.63–6.78 GB/s（**+13.0%**） |
| 端到端解码 `tg32`（Qwen3.5-0.8B Q4_K_M，`-t 4`，Clang） | 13.66 t/s | 15.53 t/s（**+13.7%**） |
| 端到端解码 `tg32`（MSVC 交付变体，逐对中位比） | — | **+13%** |
| 预填充 `pp128` | 79–89 t/s | 87–89 t/s（无变化，GEMM 路径未动） |
| L3 驻留小矩阵（0.56 MiB） | 10.07 GB/s | 10.30 GB/s（噪声内） |

正确性：改后内核输出与改前 **逐位相同**（`memcmp`）；真实引擎贪心解码（`--temp 0`）生成文本与
Build 30 完全一致；7 个实验变体对独立标量参考 `max_abs_err = 4.8e-07`（容差 2.9e-06）。

## 0.1 CLI 多轮对话修复（Build 30）

MoE 模型对话时第二轮及之后只回 `None`：CLI 每轮无条件把 assistant 消息（content 可能为空串）追加进
历史，一旦某轮没有可见输出，这条空 assistant 轮就永久留在历史里，后续请求变成
`[user, assistant(""), user]`，严格 chat template 的 MoE 模型从这一轮起只输出占位串。
同时修掉 stdin 关闭后的死循环（旧版会无限打印 `> `）。

| 文件 | 类型 | 说明 |
|------|------|------|
| `tools/cli/cli-context.cpp` | 修改 | 空回复 / 生成失败的轮次不再入历史（打印 `[empty response: ...]`，`--output` 不写空段）；空读取视为 EOF 退出；多模型选择空输入返回失败；每轮结束调 `a.finish()` |
| `tools/cli/cli-ui.h` | 修改 | `ui::assistant_turn::finish()`：回合结束时强制闭合 `[End thinking]` 并复位颜色/行尾状态 |
| `CMakeLists.txt` / `tools/rpc/rpc-server.cpp` | 修改 | Build 29 → 30 |

实测：`printf '你好\n1+1=?\n' | cli` 在 Build 29 下会持续输出 `> `（300 s / 1,483,183 行），
Build 30 正常两轮后 `Exiting...`（37 行，exit 0）。

## 0.2 `--mask-history` 训练路径问题 + 归一化口径（Build 29）

实测：同一 token 流、`-c 4096`，`--mask-history` 的每批 loss ≈ 11.9，`--no-mask-history` ≈ 1.47
（各跑两次逐位一致；标签核对无误；推理侧同文本 PPL 2.03）。**训练请改用 `--no-mask-history`。**

| 文件 | 类型 | 说明 |
|------|------|------|
| `ggml/src/ggml-opt.cpp` / `ggml/include/ggml-opt.h` | 修改 | 新增 `ggml_opt_result_add_rows()`；loss/accuracy 的按标注行归一化改用实际评估行数 `nrows`（原用 `ndata`=数据集点数） |
| `src/llama-context.cpp` | 修改 | 每评估一个 ubatch 上报 `n_ubatch` 行 |
| `tools/lumina-train/lumina-train.cpp` | 修改 | 提示 `--mask-history` 的路径问题与推荐配置 |
| `CMakeLists.txt` / `tools/rpc/rpc-server.cpp` | 修改 | Build 28 → 29 |

## 0.3 训练 loss 显示口径说明（Build 28）

进度条 loss 是累计均值；CE 核按全部行求均值，而 `ggml_opt_result_loss()` 的补偿项
`ndata/nlabeled` 用「数据集点数 / 已标注行数」（单位不一致）→ 已标注行很少时读数被放大约 8 倍
（同一文本同位置对照：掩码 11.88 vs 不掩码 1.44）。训练与梯度不受影响。
`--mask-history` 开启时会打印一条说明，建议用 `--dataset-val` 判断进度。

| 文件 | 类型 | 说明 |
|------|------|------|
| `tools/lumina-train/lumina-train.cpp` | 修改 | 删除 `-c > 2048` 告警（结论有误）；新增 `--mask-history` 口径说明 |
| `CMakeLists.txt` / `tools/rpc/rpc-server.cpp` | 修改 | Build 27 → 28 |

## 0.4 训练数据集磁盘流式落盘（Build 27）

`Lumina-Engine-train` 加载 `--dataset` 时，旧实现把整份语料追加进两个 `std::vector`
（`int32` token ids + `uint8` 可训练掩码 = **5 B/token**），加载完成后才序列化成临时文件再 mmap。
1.18 B tokens 的语料因此需要约 **5.9 GiB 追加内存**，小内存机器在加载阶段就失败。

| 文件 | 类型 | 说明 |
|------|------|------|
| `tools/lumina-train/lumina-train.cpp` | 修改 | `lumina_token_stream` 增加磁盘 sink（`FILE*` + `sink_n` / `sink_trainable` / `sink_dir`）；`lumina_append()` 直接 `fwrite` ids 与 mask；新增 `lumina_stream_begin()` / `lumina_stream_finish()` / `lumina_stream_discard()` / `lumina_append_padding()`；`lumina_stream_view` 抽出 `map_file()` 并新增 `map_streamed()`，`size()` / `n_trainable()` 统一两种模式 |
| 同上 | 修改 | 临时目录按进程隔离：`%TEMP%\lumina-train\<tag>-<pid>\`（Build 26 两个实例会写同一个 `train.stream.bin`） |
| `ggml/src/ggml-opt.cpp` | 修改 | 进度条追加实时 `tok/s`；`ggml_opt_result_loss()` 在 result 还没有 batch 时返回 `NAN`（显示 `loss=n/a`，此前为 0.0）。验证集首行偶见的 `loss=0.00000` 另有原因：该 micro-batch 只含掩码 prompt、无监督目标，Build 26 相同 |
| 构建脚本（10 个 `.bat`） | 修改 | 去掉硬编码 `C:\Lumina-Engine`，改用 `%~dp0` 相对路径；`build-clang.bat` 传参改为 `--target` |
| `CMakeLists.txt` / `tools/rpc/rpc-server.cpp` | 修改 | Build 26 → 27 |

写出布局不变（`[n×int32][pad 到 64 B][n×uint8]`），训练器读到的字节逐字节一致：

| 版本 | 加载峰值 RSS（300 M tokens） | 数据集统计 |
|---|---|---|
| Build 26 | 2074 MB | 29282 rec / 300,021,789 tok / 278,340 datapoints |
| Build 27 | **645 MB** | 完全相同 |

## 0.5 训练梯度累加器修复（Build 26）

`llama-finetune` / `llama_opt_epoch()` 路径（`Lumina-Engine-train`）在
`n_batch / n_ubatch > 1` 时，旧代码的梯度累加器从未清零：每个优化器步实际应用的是
**从训练开始到当前步所有历史梯度之和**，更新步长无界增长、loss 线性攀升。

| 文件 | 类型 | 说明 |
|------|------|------|
| `ggml/src/ggml-opt.cpp` | 修改 | 新增 `ggml_opt_zero_grad_accs()` + `grad_acc_is_loss`：直接清零持久累加器（loss 置 1），不再依赖非静态图路径下已被置空的 `gb_grad`；`ggml_opt_alloc()` 在 `backward && opt_period > 1 && opt_i == 0` 时调用 |
| 同上 | 修改 | `ggml_opt_eval()` 只在计算梯度的图上推进 `opt_i`，仅前向求值（验证 / `--resume` 重建 KV cache）不再打乱累加周期 |
| 同上 | 修改 | `ggml_opt_reset(opt_ctx, false)` 改用 `ggml_opt_zero_grad_accs()` |
| `tests/test-opt.cpp` | 修改 | 新增 `test_gradient_accumulation_nonstatic` 回归测试（非静态图 + `opt_period>1`） |
| `CMakeLists.txt` / `tools/rpc/rpc-server.cpp` | 修改 | Build 25 → 26 |

> Build 26 之前产出的 LoRA 检查点/适配器是在污染梯度下训练的，建议废弃并重新训练。

---

## 0.6 控制台统一启动画面（Build 24）

所有控制台工具启动时打印同一套 Lumina-Engine 启动画面（ASCII logo + Build 号 + 项目链接）：

| 文件 | 类型 | 说明 |
|------|------|------|
| `common/build-info.cpp.in` / `.h` | 修改 | 新增 `lumina_ascii_logo()`；`llama_print_lumina_banner()` 输出完整启动画面；Build 号升至 **Build 24** |
| `common/arg.cpp` | 修改 | `common_params_parse()` 成功后统一打印启动画面（`cli` 除外，由 CLI 打印带模型信息与命令列表的横幅） |
| `tools/cli/cli-context.cpp` | 修改 | 删除本地 logo 副本，改用共享 `lumina_ascii_logo()` |
| `tools/server/server.cpp`、`tools/cvector-generator/cvector-generator.cpp` | 修改 | 移除重复的横幅打印 |
| `tools/gguf-split/gguf-split.cpp` | 修改 | 显式打印启动画面，`--version` 复用 `llama_print_build_info()` |
| `tools/rpc/rpc-server.cpp` / `tools/rpc/CMakeLists.txt` | 修改 | 不链接 `llama-common`，用编译期 `LUMINA_RPC_BUILD_NUMBER` + 同一段 ASCII art 打印同一画面 |

---

## 1. `ggml/src/ggml-cpu/q1t.cpp`（新增，约 480 行）

Q1_0 专用推理内核。核心内容：

### 1.1 per-128 激活格式

```c
#define QK8_128 128
typedef struct {
    ggml_half d;        // 每 128 个值一个 fp16 尺度
    int8_t    qs[128];
} block_q8_128;
static_assert(sizeof(block_q8_128) == sizeof(ggml_half) + QK8_128, "...");
```

### 1.2 激活量化（AVX2 向量化 + 跨线程按块分配）

```c
static void q1t_quantize_block(const float * x, block_q8_128 * y) {
    // 4 × 32 个值：andnot+max 求 |amax|，round-to-nearest，packs 成 int8
    ...
}
```

量化在 compute 中按 128 值块跨线程分配：

```c
for (int64_t blk = ith; blk < total_blocks; blk += nth)
    q1t_quantize_block(...);
```

这样单列（tg）时也能被所有线程并行处理。早期按行分配会在 `ne11 == 1` 时只用一个线程，
其余线程在 barrier 上空等，显著拖慢 tg（已在早期版本中修正）。

### 1.3 VNNI 选择

```c
#if defined(__AVX512VNNI__)
#  define Q1T_VNNI 1
#  define Q1T_DPBUSD(acc, a, b) _mm256_dpbusd_epi32(acc, a, b)
#elif defined(__AVXVNNI__)
#  define Q1T_VNNI 1
#  define Q1T_DPBUSD(acc, a, b) _mm256_dpbusd_avx_epi32(acc, a, b)
#endif
```

### 1.4 寄存器分块内核（pp）

```c
#define Q1T_ROWS 2
#define Q1T_COLS 4

template <int R, int T>
Q1T_NOINLINE void q1t_dot(int n, float * out,
                          const block_q1_0 * const * x,
                          const block_q8_128 * const * y) {
    // ... 初始化 acc[R][T]
    for (int ib = 0; ib < nb; ++ib) {
        __m256 d0v[R];                          // 每行权重尺度广播
        for (int r = 0; r < R; ++r)
            d0v[r] = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[r][ib].d));

        __m256i t32[R][T];                      // int32 累加器（跨 4 个子块）
        // ... 清零

        for (int k = 0; k < 4; ++k) {
            __m256i sg[R];
            for (int r = 0; r < R; ++r)         // 符号展开一次
                sg[r] = q1t_expand(((const uint32_t *) x[r][ib].qs)[k]);

            for (int t = 0; t < T; ++t) {       // 复用给 T 列
                const __m256i qy = _mm256_loadu_si256(
                        (const __m256i *) (y[t][ib].qs + k * 32));
                for (int r = 0; r < R; ++r) {
                    const __m256i sy = _mm256_sign_epi8(qy, sg[r]);
#if defined(Q1T_VNNI)
                    t32[r][t] = Q1T_DPBUSD(t32[r][t], ones, sy);       // VNNI
#else
                    t32[r][t] = _mm256_add_epi32(t32[r][t],
                            _mm256_madd_epi16(_mm256_maddubs_epi16(ones, sy), ones16));
#endif
                }
            }
        }

        for (int t = 0; t < T; ++t) {           // 每 128 权重一次浮点转换
            const __m256 d1v = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[t][ib].d));
            for (int r = 0; r < R; ++r) {
                const __m256 dscale = _mm256_mul_ps(d0v[r], d1v);
                acc[r][t] = _mm256_fmadd_ps(dscale, _mm256_cvtepi32_ps(t32[r][t]), acc[r][t]);
            }
        }
    }
    // ... 水平求和输出
}
```

### 1.5 单列 GEMV 内核（tg）

```c
#define Q1T_GEMV_ROWS_DEFAULT 1   // 可用 GGML_Q1_GEMV_ROWS=1|2|4 微调
#define Q1T_GEMV_ROWS_MAX     4

template <int R>
Q1T_NOINLINE void q1t_gemv(int n, float * out,
                           const block_q1_0 * const * x,
                           const block_q8_128 * y) {
    for (int ib = 0; ib < nb; ++ib) {
        __m256i t32[R] = {0};
        for (int k = 0; k < 4; ++k) {
            const __m256i qy = _mm256_loadu_si256((const __m256i *) (y[ib].qs + k * 32));
            for (int r = 0; r < R; ++r) {
                const __m256i sg = q1t_expand(((const uint32_t *) x[r][ib].qs)[k]);
                t32[r] = Q1T_DPBUSD(t32[r], ones, _mm256_sign_epi8(qy, sg));
            }
        }
        // 4 个子块的 int32 先累加，每 128 权重只做 1 次浮点转换
        const __m256 d1v = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib].d));
        for (int r = 0; r < R; ++r)
            acc[r] = _mm256_fmadd_ps(_mm256_mul_ps(d0v[r], d1v),
                                     _mm256_cvtepi32_ps(t32[r]), acc[r]);
    }
}
```

判定：`src1->ne[1] == 1` 且未设置 `GGML_NO_Q1_GEMV`。

### 1.6 判定、scratch 与 compute

```c
bool ggml_cpu_q1t_supports_mul_mat(const struct ggml_tensor * dst);  // Q1_0 + AVX2 + ...

// 共享 scratch：所有 src1 行的 per-128 激活
size_t ggml_cpu_q1t_scratch_size(const struct ggml_tensor * dst);

// 量化 src1 → barrier → 分块矩阵乘
void ggml_compute_forward_mul_mat_q1t(const struct ggml_compute_params * params,
                                      struct ggml_tensor * dst);
```

`compute` 关键流程：

```c
// 1. 并行把 F32 激活量化为 Q8_128（按 128 值块分给各线程）
for (int64_t blk = ith; blk < total_blocks; blk += nth)
    q1t_quantize_block(src + bi*QK8_128, scratch + row*qrow_size + bi);

// 2. 同步
ggml_barrier(params->threadpool);

// 3a. ne11 == 1：单列 GEMV（tg）
// 3b. 否则：2 行 × 4 列动态分块 GEMM（pp）
//     行尾 / 列尾回退到通用 vec_dot（Q8_0）
```

---

## 2. `ggml/src/ggml-cpu/q1t.h`（新增）

```c
bool   ggml_cpu_q1t_supports_mul_mat(const struct ggml_tensor * dst);
size_t ggml_cpu_q1t_scratch_size   (const struct ggml_tensor * dst);
void   ggml_compute_forward_mul_mat_q1t(const struct ggml_compute_params * params,
                                        struct ggml_tensor * dst);
```

---

## 3. `ggml/src/ggml-cpu/ggml-cpu.c`（修改）

### 3.1 include

```c
#include "traits.h"
#include "iqp.h"
#include "q1t.h"      // ← 新增
```

### 3.2 mul_mat 中的 hook（在 src1→Q8_0 转换 + barrier 之后）

```c
// IQ panel gemm (iqp.h)
if (ggml_cpu_iqp_supports_mul_mat(dst) && !params->use_ref) {
    ggml_compute_forward_mul_mat_iqp(params, dst);
    return;
}

// 1-bit Q1_0 register-tiled gemm (q1t.h)   ← 新增
if (ggml_cpu_q1t_supports_mul_mat(dst) && !params->use_ref) {
    ggml_compute_forward_mul_mat_q1t(params, dst);
    return;
}
```

### 3.3 work buffer 大小

```c
// the IQ panel path needs one scratch panel per thread past the q8_K rows
if (ggml_cpu_iqp_supports_mul_mat(node)) {
    cur = GGML_PAD(cur, 64) + n_tasks * ggml_cpu_iqp_scratch_size(node);
}

// the Q1_0 register-tiled path needs the q8_128 activation past the q8_0 rows  ← 新增
if (ggml_cpu_q1t_supports_mul_mat(node)) {
    cur = GGML_PAD(cur, 64) + ggml_cpu_q1t_scratch_size(node);
}
```

---

## 4. `ggml/src/ggml-cpu/CMakeLists.txt`（修改）

```cmake
ggml-cpu/iqp.cpp
ggml-cpu/iqp.h
ggml-cpu/q1t.cpp      # ← 新增
ggml-cpu/q1t.h        # ← 新增
```

---

## 5. `ggml/src/ggml-cpu/arch/x86/quants.c`（修改）

### 5.1 VNNI 宏（文件级）

```c
#if defined(__AVX512VNNI__)
#  define GGML_Q1_0_VNNI 1
#  define GGML_Q1_0_DPBUSD(acc, a, b) _mm256_dpbusd_epi32(acc, a, b)
#elif defined(__AVXVNNI__)
#  define GGML_Q1_0_VNNI 1
#  define GGML_Q1_0_DPBUSD(acc, a, b) _mm256_dpbusd_avx_epi32(acc, a, b)
#endif
```

### 5.2 `ggml_vec_dot_q1_0_q8_0` 的 AVX2 分支

**改进前：**

```c
const __m256i sm = _mm256_cmpeq_epi8(
        _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) qs32[K]), byte_shuf), bit_masks), zero);
const __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
```

**改进后：**

```c
// 0xFF where the bit is 0, 0x01 where it is 1; vpsignb then negates the corresponding lanes
const __m256i sg = _mm256_or_si256(_mm256_cmpeq_epi8(
        _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) qs32[K]), byte_shuf), bit_masks), zero), ones_8);
const __m256i sy = _mm256_sign_epi8(qy, sg);
#if defined(GGML_Q1_0_VNNI)
const __m256i s32 = GGML_Q1_0_DPBUSD(_mm256_setzero_si256(), ones_8, sy);
#else
const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
#endif
```

同一改动应用在首个子块与 `K=1..3` 循环中（共 2 处）。

---

## 6. `q1t.cpp` 内的二值查表（Q1LUT，默认启用）

新增于一文件内（无新文件）：

- `q1lut_get_or_alloc` / `q1lut_fill_group`：把 Q1_0 权重惰性置换为查表码，并收集每 128 权重的
  `d0` 尺度（直接拷贝原 fp16，tg 权重流量与打包格式持平）；按张量缓存于进程内、按行组并行填充。
- `q1lut_build_group`：由激活构建 16 项 `int16` 表（低字节 + 高字节）。利用真二值对称性
  `T[c ^ 0xf] = -T[c]`，只累加 `c = 0..7`，其余直接取负填入，生成量减半。
- `q1lut_acc128` / `q1lut_gemv32`：AVX2 内核，一次处理 32 个输出行，用 `vpshufb` 查表（整表直写，默认）。
- `q1lut_gemv32_add`：实验性 K 分块累加内核，由 `GGML_Q1_LUT_KB` 启用（默认 0=整表）；实测更慢。
- 批处理 pp 路径（默认启用，`GGML_Q1_LUT_PP=0` 关闭）：按 32 列分块构建表，`q1lut_gemv32` 一次算一列。
- `ggml_cpu_q1t_scratch_size` / `ggml_compute_forward_mul_mat_q1t`：`ne11 == 1` 走 tg 查表路径；
  `ne11 >= Q1LUT_PP_MIN(8)` 走 pp 查表（均默认启用，受 `GGML_Q1_LUT` / `GGML_Q1_LUT_PP` 控制）。
- `q1t_env_bool`：布尔环境变量解析（`0/false/off/no` 为假），供上述开关与 `GGML_Q1_LUT_DEBUG` 使用。

由 `GGML_Q1_LUT`（总开关）与 `GGML_Q1_LUT_PP`（仅 pp）控制；关闭时对上游和其它路径零影响。测试工具：`bench/bench_lut2.cpp`。

---

## 7. 模型无损压缩容器（Build 6/7/8，新增）

在**不量化、不剪枝**的前提下减小模型文件。新增/修改：

| 文件 | 类型 | 说明 |
|------|------|------|
| `src/lumina-gguf.cpp` / `.h` | 新增 | 识别 `LGGC` 容器，流式解压到临时 `.gguf`，不量化/不剪枝，逐字节还原 |
| `src/miniz/miniz.{c,h}` | 新增 | 内置单文件 DEFLATE（MIT / 公有领域），无外部依赖 |
| `src/llama.cpp` | 修改 | `llama_model_load_from_file()` 调用 `lumina_gguf_resolve_path()`（非压缩文件原样返回） |
| `src/CMakeLists.txt` | 修改 | 编译 `lumina-gguf.cpp` 与 `miniz.c` |
| `tools/lumina_gguf_pack.py` | 新增 | 打包器：k-quant 字段拆分 + 上下文排序 + DEFLATE（容器 v2） |
| `tools/lumina_gguf_unpack.py` | 新增 | 参考解包器 / `verify` 逐字节校验 |
| `common/build-info.cpp.in` | 修改 | 版本号升至 **Build 8** |
| `common/preset.cpp` | 修改 | 模型目录列表识别 `.gguf.lumina`（名字去掉后缀） |
| `common/arg.cpp` | 修改 | `-m` shell 补全同时匹配 `.gguf` 与 `.gguf.lumina` |

**Build 7（容器 v2）**：在 Build 6 的字段拆分之上，对 Q4_K 的 `d/dmin` 做 fp16 字节平面拆分、
把 6-bit `scale`/`min` 解包为字节流，并把 4-bit 量化值按上下文键 `(scale << 6 | min)` 稳定排序；
Q6_K 的 `d` 同样字节平面拆分、`scales` 奇偶拆分。解码器同时兼容 v1 与 v2。

**Build 8（扩展名约定）**：打包器默认把 `A.gguf` 输出为 `A.gguf.lumina`（原文件保留），
便于区分原模型与压缩容器；引擎按魔数加载，`.gguf` 与 `.gguf.lumina` 均可直接使用，模型目录
列表与命令行补全会把 `.gguf.lumina` 当作模型。

本模型实测：`2,630,213,408 B → 2,479,779,960 B`（**94.28%**，节省 143.5 MiB；容器 v1 为
95.30% / 117.9 MiB），解压后 SHA-256 与原文件一致。原理与格式见
[LosslessCompression](LosslessCompression.md)。

---

## 8. 训练引擎与数据集支持（Build 9/10/11）

新增纯 C++ 训练工具 `tools/lumina-train/`（交付名 `Lumina-Engine-train`），在冻结的量化基座上
训练 LoRA 适配器（QLoRA）。Build 10 补齐 MoE 训练所需的 ggml 反向算子（`MUL_MAT_ID`、
`REGLU`/`SIGMOID`/`TANH`/`CLAMP`/broadcast `DIV`、批量 `GET_ROWS_BACK`）；Build 11 增加数据集与检查点支持。

| 文件 | 类型 | 说明 |
|------|------|------|
| `tools/lumina-train/lumina-train.cpp` | 新增/修改 | QLoRA 训练工具；Build 11 支持 `.jsonl` 数据集（`text`/`messages`/`conversations`）、独立验证集、检查点、`--cpu-cores` |
| `include/llama.h` / `src/llama-adapter.cpp` | 修改 | `llama_adapter_lora_init_trainable()` / `_save()` / Build 11 的 `_copy_weights()` |
| `src/llama-context.cpp` | 修改 | `opt_init()` 注册 LoRA 参数、图节点预算翻倍；Build 11 的负标签 loss 掩码 |
| `common/common.h` / `common/arg.cpp` | 修改 | `-lora-*` 参数；Build 11 的 `--dataset*` / `--mask-history` / `--cpu-cores` / `--save-steps` / `--keep-last` / `--resume` |
| `ggml/include/ggml.h` / `ggml-rpc.h` / `ggml.c` / 后端 | 修改 | MoE 反向算子（Build 10） |
| `common/build-info.cpp.in` / `CMakeLists.txt` | 修改 | 版本号升至 **Build 11** |

---

## 9. Vulkan 后端与默认设备策略（Build 12）

| 文件 | 类型 | 说明 |
|------|------|------|
| `configure-release.bat` | 修改 | 检测 `D:\VulkanSDK\sdk`，存在则 `-DGGML_VULKAN=ON`，否则自动回退 CPU-only |
| `common/common.h` | 修改 | `n_gpu_layers` 默认 `-1`（auto）→ **`0`**（层留 CPU + 允许 Vulkan 算子 offload） |
| `common/arg.cpp` | 修改 | 移除旧的 `GGML_ASSERT(n_gpu_layers < 0)`，修正 `-ngl` 帮助文案 |
| 交付目录 | 修改 | 新增 `ggml-vulkan.dll`（共 55 文件） |

实测（本机 Intel Iris Xe）：默认 `-ngl 0` 下 Qwen3-0.6B pp512 **≈2.4×**、phi-3-mini pp256 **≈1.6×**，tg 不降；
`-ngl auto/99` 整层 offload 时 pp 更高但 tg 变慢；困惑度 CPU vs op-offload 差 **+0.11%**。
完整 A/B 见 `OPTIMIZATION_LOG.md`。

**Build 13**：`tools/lumina-train/` 训练前强制 `n_gpu_layers = 0` 且 `no_op_offload = true`——
否则 Vulkan 算子 offload 会污染训练的准确率指标（acc ≈ 8.25e8%）；修复后恢复正常（~10%）。

**Build 14**：`common/common.cpp` 加入机器+模型感知的自动设备策略（无 GPU/IGPU → 层留 CPU；
dGPU → offload 整层；MoE 模型 → 自动关闭算子 offload）。一套预编译、程序自决。
MoE 走算子 offload 反而慢 ~30%（143 vs 207 t/s），关闭后恢复。

**Build 15**：GPU 训练实验结论——不可行（Vulkan 无反向算子）。保留 `LUMINA_TRAIN_ALLOW_GPU`
环境开关（默认关闭、实验性），默认训练强制纯 CPU。

**Build 16**：修复 GPU 训练下 `acc` 指标损坏 —— 根因是 Vulkan `count_equal` 内核固定写 32 位而
`ggml_count_equal` 输出 I64；改为仅 I32 目标走 Vulkan、I64 回退 CPU。
同时在 `--version` / CLI 横幅 / server 启动显示 Build 号与项目链接。
（另：Vulkan 标准反向算子其实早已实现，dense GPU 训练 loss 与 CPU 一致。）

---

## 改动规模

| 文件 | 类型 | 行数变化（约） |
|------|------|---------------|
| `q1t.cpp` | 新增 | +480（基础）+ ~190（Q1LUT） |
| `q1t.h` | 新增 | +30 |
| `ggml-cpu.c` | 修改 | +10 |
| `CMakeLists.txt` | 修改 | +2 |
| `arch/x86/quants.c` | 修改 | +20 |
| `bench/bench_lut2.cpp` | 新增 | 独立微基准 |
| `lumina-gguf.cpp` / `.h` | 新增 | +约 330（无损压缩容器解码器，兼容 v1/v2） |
| `miniz/miniz.{c,h}` | 新增 | 内置 DEFLATE（vendored，MIT/公有领域） |
| `tools/lumina_gguf_pack.py` / `_unpack.py` | 新增 | 打包 / 校验脚本 |

**总改动 < 620 行**，且几乎全部集中在新文件里，对上游代码的侵入极小。

---

## 构建 / 交付脚本

除内核改动外，仓库根目录与 `bench/` 下新增/更新了构建与交付脚本（不涉及推理代码，
详见 [Delivery](Delivery.md)）：

| 脚本 | 作用 |
|------|------|
| `build-bin.bat` | 一键：MSVC + Clang 编译 + 组装到 `Lumina-Engine-Bin` |
| `bench/build_all_msvc.bat` / `bench/build_all_clang.bat` | 编译两棵构建树 |
| `bench/assemble_bin.bat` | clang 基底 + 9 个 MSVC 变体覆盖 + 运行库 → `Lumina-Engine-Bin` |
| `bench/build_haswell.bat` / `bench/build_x64.bat` | 单变体增量编译（开发用） |
| `bench/bench_lut2.cpp` | Q1LUT 独立内核微基准（含正确性校验） |

**约定**：`Lumina-Engine-Bin` 是唯一的交付目录；旧入口 `D:\Lumina-Engine\llama` 已删除。

---

## 未保留的探索性改动

以下方案经过评估后**未纳入**，仓库中无残留：

| 方案 | 结论 |
|------|------|
| 内存面板（IQP 风格） | 额外 1 字节/权重的读写，抵消收益，实测变慢 |
| AVX-512 512-bit tile | 布局不匹配 + d1 向量构造开销，理论收益 ~1.1×，且无法本机验证 |
| 多行 GEMV + Q8_0 激活（tg） | 激活尺度仍为 per-32，浮点转换次数不变，实测无收益；tg 改用 Q8_128 单列 GEMV |
| 新 GGML 类型 Q8_128 | 需改动 175+ 后端文件，风险过大；改为 q1t 内部局部格式 |
| 软件预取 | 无收益 |
| 2-block 展开 | 收益不稳定，寄存器压力上升 |
