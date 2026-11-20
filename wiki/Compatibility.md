# 兼容性与分发

> 🌐 **中文** | [English](Compatibility.en.md)

## 1. CPU 变体总表

`Lumina-Engine-Bin/` 目录包含 14 个 CPU 后端 DLL。运行时会根据 CPUID 选择**满足条件且特性最高**的一个。

| DLL | 关键特性 | 编译器 | 分块 GEMM | VNNI | 典型 CPU |
|-----|----------|--------|:---------:|:----:|----------|
| `ggml-cpu-x64` | 基线（SSE2） | MSVC | — | — | 极老 x64 |
| `ggml-cpu-sse42` | SSE4.2 | MSVC | — | — | Nehalem+ |
| `ggml-cpu-sandybridge` | AVX | MSVC | — | — | Sandy Bridge |
| `ggml-cpu-ivybridge` | AVX + F16C | Clang | — | — | Ivy Bridge |
| `ggml-cpu-piledriver` | AVX + F16C + FMA | Clang | — | — | AMD Piledriver |
| `ggml-cpu-haswell` | AVX2 + FMA + BMI2 | MSVC | ✅ | — | Haswell ~ Comet Lake（**本机**） |
| `ggml-cpu-skylakex` | + AVX-512 | MSVC | ✅ | — | Skylake-X |
| `ggml-cpu-cannonlake` | + AVX-512 VBMI | MSVC | ✅ | — | Cannon Lake |
| `ggml-cpu-cascadelake` | + AVX-512 VNNI | MSVC | ✅ | ✅ | Cascade Lake |
| `ggml-cpu-icelake` | + AVX-512 VBMI + VNNI | MSVC | ✅ | ✅ | Ice Lake |
| `ggml-cpu-cooperlake` | + AVX-512 VNNI + BF16 | Clang | ✅ | ✅ | Cooper Lake |
| `ggml-cpu-alderlake` | AVX2 + AVX-VNNI | MSVC | ✅ | ✅ | Alder Lake ~ Rocket Lake |
| `ggml-cpu-zen4` | AVX-512 + VNNI + BF16 | Clang | ✅ | ✅ | AMD Zen 4/5 |
| `ggml-cpu-sapphirerapids` | AVX-512 + VNNI + BF16 + AMX | Clang | ✅ | ✅ | Sapphire Rapids |

> “分块 GEMM”要求 AVX2；“VNNI”在支持时额外启用 `vpdpbusd`。
> 早期 CPU 回退到通用 Q1_0 内核，**仍然正确**，只是较慢。

---

## 2. 为什么 MSVC 只能构建 9 个变体

CMake 中 `GGML_CPU_ALL_VARIANTS` 对 MSVC 跳过了 5 个变体：

```cmake
if (NOT MSVC)
    # __FMA__ / __F16C__ 在 MSVC 中未定义（AVX2/AVX512 隐含）
    ggml_add_cpu_backend_variant(ivybridge      SSE42 AVX F16C)
    ggml_add_cpu_backend_variant(piledriver     SSE42 AVX F16C FMA)
    ...
    # MSVC 不支持 BF16 / AMX intrinsic
    ggml_add_cpu_backend_variant(cooperlake     ... AVX512_BF16)
    ggml_add_cpu_backend_variant(zen4           ... AVX512_BF16)
    ggml_add_cpu_backend_variant(sapphirerapids ... AVX512_BF16 AMX_TILE AMX_INT8)
endif()
```

因此完整 14 变体需要 Clang。本仓库用 **Clang 20.1.8** 补齐了这 5 个。
（另外 VS2026 的 MSVC STL 要求 Clang ≥ 20；Clang 16 的 Emscripten 版本会因
`error STL1000: Unexpected compiler version, expected Clang 20 or newer` 而失败。）

---

## 3. 混合组装

由于：

- MSVC 对分块内核的代码生成更好（pp128 ≈ 16.4 vs Clang 14.9）；
- 只有 Clang 能构建全部 14 个变体；

最终交付采用混合：**Clang 工具链 + 14 变体，其中 9 个常用变体替换为 MSVC 版本**。

CPU 后端是独立 DLL、通过 C ABI 与 `ggml-base.dll` 交互，已验证混用正常：

```
[MSVC ggml-cpu-haswell.dll] + [Clang Lumina-Engine-cli.exe / ggml-base.dll]  →  pp128 = 15.41 ✅
```

---

## 4. 运行时选择逻辑

`ggml` 的 DL 加载器（`ggml_backend_load_all_from_path`）会：

1. 扫描 `ggml-cpu-*.dll`。
2. 对每个读取其自描述的 CPU 特性要求。
3. 用 `cpuid` 检查当前 CPU 是否满足。
4. 选择**满足条件中特性最高**的一个。

因此同一份 `Lumina-Engine-Bin/` 目录可直接拷贝到不同 CPU 的 Windows 机器上运行，无需重新编译。

---

## 5. 本机实际加载

```
load_backend: loaded CPU backend from ...\ggml-cpu-haswell.dll
```

Comet Lake 不支持 AVX-512 / VNNI，因此选择 `haswell`（AVX2）。
如需确认某台机器加载了哪个变体，可查看 `Lumina-Engine-cli` 的启动日志或观察
`Lumina-Engine-bench` 的 `backend` 列。

---

## 6. Windows 版本要求

- Windows 10 1607+ / Windows 11（x64）。
- 需要 Universal CRT（`api-ms-win-crt-*.dll`，Win10 自带）。
- VC++ 运行库与 OpenMP 运行库已随 `Lumina-Engine-Bin/` 一起分发。

---

## 7. 分发清单

```
Lumina-Engine-Bin/
├─ 14 × ggml-cpu-*.dll
├─ ggml-base.dll  ggml.dll  ggml-rpc.dll  ggml-vulkan.dll
├─ Lumina-Engine.dll  Lumina-Engine-common.dll  mtmd.dll
├─ 20 × Lumina-Engine-*.exe  +  ggml-rpc-server.exe
├─ 8 × *-impl.dll
├─ libomp140.x86_64.dll   (Clang 后端/工具)
├─ vcomp140.dll           (MSVC 后端)
└─ msvcp140.dll  vcruntime140.dll  vcruntime140_1.dll
```

共 **55** 个文件（21 exe + 8 impl + 14 CPU 变体 + 4 ggml DLL + 3 引擎 DLL + 5 运行库）。
直接整体拷贝即可在任意兼容 Windows x86-64 机器上运行。

---

## 8. 二值查表（Q1LUT）

- 需要 AVX2（`haswell` 及以上）；`x64` / `sse42` / `sandybridge` 变体回退到通用内核。
- 对所有 AVX2 变体（MSVC 与 Clang）可用，**默认启用**；`GGML_Q1_LUT=0` 整体关闭，`GGML_Q1_LUT_PP=0` 仅关闭 pp 查表。
- 建议 `-t` 不超过物理核心数（本机 6 物理核：`-t 6` 获益，`-t 12` 反降）。
- 本机为**单通道内存**，tg 受带宽上限约束（理论 ~15.7 tok/s）；双通道/更高带宽平台上 tg 提升更接近内核层面。
- 需要约 0.95 GiB 额外内存存放置换后的权重；首次使用有约 0.7 s 的一次性填充。
- 关闭后行为与改动前完全一致。

---

## 9. 模型格式与量化类型支持

| 项目 | 支持情况 |
|------|----------|
| 量化类型 | `Q1_0`（Bonsai 1-bit，专用内核）、`Q2_0` / `TQ2_0`（三元）、`Q4_K` / `Q6_K`、`Q4_0` / `Q5_0` / `Q8_0`、`MXFP4` / `NVFP4`、`IQ*` 系列等（CPU / Vulkan / CUDA 后端按各自支持表） |
| 架构 | `qwen35` / `qwen35moe` 等全部上游架构；`qwen35` 额外支持 M-RoPE 与 mmproj 多模态 |
| PrismML 旋转权重 | **Build 33 起支持**：GGUF 带 `prism.hadamard.*` 元数据时自动读出分块 512 归一化 Sylvester–Hadamard + ±1 符号，并在推理时对激活做对应旋转（`Ternary-Bonsai-2-2B-Q2_0` 实测通过） |
| PrismML mmproj | `qwen3vl_merger` 投影（`Ternary-Bonsai-2-2B-mmproj-Q8_0`）用 `Lumina-Engine-mtmd-cli` 加载；视觉向量走原始 embedding 分支，**不做**旋转 |

- 无 `prism.hadamard.*` 元数据的模型行为完全不变（相关代码全部按元数据关闭）。
- 只支持元数据里写明的三种取值：`transform = normalized-sylvester-walsh-hadamard`、
  `axis = input-last-dimension`、`sign_mode = explicit`；其它取值只告警并**忽略**旋转（不会崩）。
- 旋转复用 FWHT 快速路径（CPU / Vulkan / CUDA / Metal / SYCL）；不支持 `GGML_HINT_SRC0_IS_HADAMARD`
  的后端退化为真实矩阵乘，结果相同但更慢。
- 排障开关：`LUMINA_PRISM_DISABLE=1`（完全关闭）、`LUMINA_PRISM_ORDER=hd|dh`、
  `LUMINA_PRISM_EMBD=inv|fwd|none`、`LUMINA_PRISM_OUT=fwd|inv|none`、`LUMINA_PRISM_NO_FWHT=1`
  （改用真实矩阵乘）、`LUMINA_PRISM_DEBUG=1`（打印旋转书签，需 `-v`）、`LUMINA_NO_FUSED_GDN=1`。
- 验收要点：**批处理困惑度正常不代表自回归生成正常** —— 必须同时比较 `-ub 512` 与 `-ub 1` 的生成结果，
  并用 `perplexity -b 1` 确认不出现 `nan`。
