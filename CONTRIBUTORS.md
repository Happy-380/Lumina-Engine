# Contributors

> 🌐 [中文](#中文) | [English](#english)

## 中文

Lumina-Engine 由 **Happy-380** 发起并维护。感谢每一位提交代码、复现问题、补充文档，以及在各种 CPU / GPU 上跑基准的人。

### 维护者

| 贡献者 | 主要贡献 |
|---|---|
| [Happy-380](https://github.com/Happy-380) | 项目发起与维护；Q1_0 CPU 内核（Q1LUT / per-128 激活 / 单列 GEMV / 分块 GEMM）；Vulkan Q1_0 DP4A 移植；QLoRA 训练引擎；`.gguf.lumina` 无损压缩；14 变体构建与交付；文档 |

### 社区贡献者

完整名单（按提交统计）见 GitHub 的 [Contributors 图表](https://github.com/Happy-380/Lumina-Engine/graphs/contributors)。

如果你希望在本文件中留下署名，欢迎在 PR 中按以下格式追加到本节：

```markdown
- 名字 (@handle) — 贡献摘要
```

贡献不限于代码，以下任何一项都值得被记录：

- **代码 / 内核**：Q1_0 内核、Vulkan、训练算子、工具链。
- **基准与正确性**：在不同 CPU / GPU 上跑 A/B、困惑度、`test-backend-ops`，提交可复现的数据。
- **文档与翻译**：wiki 中英文页面、使用教程、排障记录。
- **问题复现**：附带命令、模型和硬件信息的高质量 issue。

### 上游与特别致谢

| 项目 / 个人 | 致谢原因 |
|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | 上游推理框架、工具链与全部基础算子（MIT） |
| [PrismML](https://prismml.com) | Q1_0 量化格式与 Bonsai 模型 |
| [Rich Geldreich / miniz](https://github.com/richgel999/miniz) | `.gguf.lumina` 使用的单文件 DEFLATE 实现 |
| 提交 issue、基准数据与硬件测试的社区成员 | 帮助定位平台差异、性能回归与正确性问题 |

### 说明

- 名单按首次贡献排序；如发现遗漏或希望修改署名，请提交 PR 或 issue。
- 所有贡献默认以本仓库的 [AGPL-3.0](LICENSE) 许可发布。
- 提交前建议阅读 README 的[贡献](README.md#贡献)一节。

---

## English

Lumina-Engine was started and is maintained by **Happy-380**. Thanks to everyone who contributes code, reproduces issues, improves documentation, or runs benchmarks on different CPUs and GPUs.

### Maintainers

| Contributor | Main contributions |
|---|---|
| [Happy-380](https://github.com/Happy-380) | Project founder and maintainer; Q1_0 CPU kernels (Q1LUT / per-128 activations / single-column GEMV / tiled GEMM); Vulkan Q1_0 DP4A port; QLoRA training engine; `.gguf.lumina` lossless compression; 14-variant build and delivery; documentation |

### Community contributors

The complete list (by commit count) is available in GitHub's [Contributors graph](https://github.com/Happy-380/Lumina-Engine/graphs/contributors).

If you would like to be listed in this file, add an entry to this section in your PR using this format:

```markdown
- Name (@handle) — contribution summary
```

Contributions are not limited to code; any of the following is worth recording:

- **Code / kernels**: Q1_0 kernels, Vulkan, training operators, tooling.
- **Benchmarks and correctness**: A/B runs, perplexity, and `test-backend-ops` on different CPUs / GPUs, with reproducible data.
- **Docs and translation**: Chinese/English wiki pages, tutorials, troubleshooting notes.
- **Issue reproduction**: high-quality issues with commands, model, and hardware details.

### Upstream and special thanks

| Project / person | Why we thank them |
|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Upstream inference framework, tooling, and all foundational operators (MIT) |
| [PrismML](https://prismml.com) | Q1_0 quantization format and Bonsai models |
| [Rich Geldreich / miniz](https://github.com/richgel999/miniz) | Single-file DEFLATE implementation used by `.gguf.lumina` |
| Community members who filed issues, shared benchmarks, or tested hardware | Helped locate platform differences, performance regressions, and correctness issues |

### Notes

- The list is ordered by first contribution. If something is missing or you want your name changed, open a PR or issue.
- All contributions are released under this repository's [AGPL-3.0](LICENSE) license by default.
- Before submitting, please read the [Contributing](README.en.md#contributing) section of the README.
