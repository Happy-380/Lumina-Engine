# Training engine: fine-tuning quantized models (QLoRA)

> 🌐 [中文](Training.md) | **English**

This repository ships a **C++ training engine**, `Lumina-Engine-train`, for fine-tuning
**quantized models** (`Q4_K` / `Q8_0` / `Q1_0`, i.e. any GGUF quantization) as well as
`.gguf.lumina` compressed containers.

> tl;dr: **quantized weights cannot be trained directly.** The engine implements **QLoRA**:
> the base weights stay frozen (and stay in their quantized storage) while only low-rank
> adapters (LoRA A/B matrices) are trained. No dequantize → train → requantize round trip is
> needed, and the memory footprint stays small.

> ⚠️ **Build 26 fixed a critical bug: the gradient accumulators were never cleared.** Older
> builds applied the sum of all gradients since the start of training at every optimizer step,
> so the update magnitude grew without bound and the loss climbed approximately linearly.
> LoRA checkpoints/adapters produced before Build 26 should be discarded and retrained.

---

## 1. Why not "directly fine-tune the quantized weights"

The upstream `llama-finetune` (`examples/training/finetune.cpp`) only supports **F32 models**:

- `llama_set_param()` only accepts `GGML_TYPE_F32` tensors, so quantized weights cannot be
  registered as trainable parameters.
- Backprop needs `GGML_OP_OUT_PROD`, which is not supported when the base weights have been
  **repacked** into an extra buffer type (see §5).
- Even if it worked, full fine-tuning of an 8B model needs ~4× the parameter count
  (weights + grads + AdamW momenta) — infeasible in 8 GB of RAM.

Hence this engine uses **LoRA/QLoRA**: frozen quantized weights run the forward pass, and
gradients flow only into the newly allocated low-rank matrices.

---

## 2. Quick start

The tool lives in the delivery directory as `Lumina-Engine-Bin\Lumina-Engine-train.exe`.

```bat
REM QLoRA fine-tuning of a Q4_K / Q8_0 / Q1_0 model (rank=8, alpha=16)
Lumina-Engine-train.exe ^
  -m model.Q4_K_M.gguf ^
  -f train.txt ^
  -c 512 -b 512 -ub 128 ^
  -t 6 -lr 1e-4 --epochs 3 ^
  -lora-rank 8 -lora-alpha 16 ^
  -o model.lora.gguf

REM datasets (.jsonl, text / messages / conversations auto-detected) + a separate val set
Lumina-Engine-train.exe ^
  -m model.Q4_K_M.gguf ^
  --dataset data/stage2_sft/train.jsonl --dataset-val data/stage2_sft/val.jsonl ^
  -c 512 -b 512 -ub 128 --cpu-cores 8 -lr 5e-5 --epochs 2 ^
  -lora-rank 64 -lora-alpha 128 ^
  --save-steps 500 --save-every-epochs 1 --keep-last 3 ^
  -o model.lora.gguf

REM compressed containers work too (decompressed to %TEMP% on first load)
Lumina-Engine-train.exe -m model.gguf.lumina -f train.txt -c 512 -b 128 -ub 128 -t 6 -o model.lora.gguf
```

The result is a **standard LoRA adapter GGUF** that existing inference tools can load:

```bat
Lumina-Engine-cli.exe -m model.Q4_K_M.gguf --lora model.lora.gguf -p "..." -n 128 -t 6
```

Optionally merge the adapter back into the base model with `Lumina-Engine-export-lora`.

---

## 3. Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | base model (`.gguf` or `.gguf.lumina`, may be quantized) | — |
| `-f, --file` / `-p, --prompt` | single training text | — |
| `--dataset` | training dataset (`.jsonl`, see §3.1) | — |
| `--dataset-val` | separate validation dataset (`.jsonl`, same format) | empty (use `--val-split`) |
| `--dataset-limit` / `--dataset-max-tokens` | max records / tokens per dataset file (`0` = unlimited) | `0` |
| `--dataset-chat-template` | chat template override used to render chat records | model's own |
| `--mask-history` / `--no-mask-history` | chat data: only train on assistant turns | `--mask-history` |
| `-o, --out-file` | output LoRA adapter file | `lumina-lora.gguf` |
| `--cpu-cores` | CPU cores (threads) used for training; `0` = all available | `0` |
| `-lora-rank` | low-rank dimension; `0` = full fine-tune (F32 models only) | `8` |
| `-lora-alpha` | LoRA scaling; effective scale = `alpha / rank` | `16` |
| `-lora-targets` | comma-separated base tensor substrings to attach adapters to | `attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down` |
| `-lr, --learning-rate` | learning rate (AdamW) | `1e-5` |
| `-epochs` | number of training epochs | `2` |
| `-wd, --weight-decay` | AdamW weight decay | `0` |
| `--val-split` | validation fraction carved out of the training stream (when no `--dataset-val`) | `0.05` |
| `--save-steps` / `--save-every-steps` | save a checkpoint every N optimizer steps (`0` = off) | `500` |
| `--save-every-epochs` | save a checkpoint every N epochs (`0` = off) | `1` |
| `--keep-last` | keep only the last N checkpoints, delete older ones (`0` = keep all) | `0` |
| `--checkpoint-dir` | checkpoint directory | same as `-o` |
| `--resume` / `--no-resume` | resume from the newest checkpoint in the checkpoint directory | `--no-resume` |
| `-c, -b, -ub` | context / batch / micro-batch | see `--help` |

> `-lora-targets` also matches `wq/wk/wv/wo/w1/w2/w3` to cover other naming schemes.

### 3.1 Dataset format (`--dataset` / `--dataset-val`)

The dataset is JSONL (one JSON object per line); three shapes are auto-detected:

| Shape | Example | Use |
|-------|---------|-----|
| plain text | `{"text": "some training text"}` | continued pre-training (next-token prediction) |
| OpenAI messages | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` | instruction tuning / SFT |
| ShareGPT | `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}` | instruction tuning / SFT |

* Chat records are rendered with the **model's own chat template** (override with
  `--dataset-chat-template`) so they match inference-time formatting;
* `--mask-history` (default) computes the loss only on `assistant` turns; system / user
  turns and template markers are masked (a `-1` sentinel label makes their loss and gradient
  exactly zero);
* plain-text records always contribute over the whole sequence;
* train and validation sets can be separate files (`--dataset-val`); within a single stream use
  `--val-split`.

### 3.2 Checkpoints and resume

Every checkpoint writes two files: `<out>.ckpt-<step>.gguf` (LoRA weights) and
`<out>.ckpt-<step>.json` (metadata: `epoch` / `step` / rank / alpha / targets / base model),
where `<out>` is `-o` without its extension. `--resume` scans the checkpoint directory, picks
 the highest `step`, copies the LoRA weights into the fresh adapter
(`llama_adapter_lora_copy_weights()`) and continues from the recorded epoch.

---

## 4. Output format

The output is a standard llama.cpp LoRA adapter GGUF:

```
general.type            = adapter
general.architecture    = <same as base>
adapter.type            = lora
adapter.lora.alpha      = <alpha>
tensors: <base_tensor_name>.lora_a / <base_tensor_name>.lora_b
```

Tensor shapes match `convert_lora_to_gguf.py` (`lora_a` = `[n_in, rank]`, `lora_b` =
`[rank, n_out]`), so the file loads directly via `llama_adapter_lora_init()` / `--lora`
and stays interoperable with upstream tooling.

---

## 5. Implementation notes

- **Trainable adapter**: new public API
  `llama_adapter_lora_init_trainable(model, rank, alpha, targets, n_targets)` allocates F32
  A/B for the matching weight tensors (A ~ N(0, 1/√n_in), B = 0, so the adapter starts as a
  no-op) and registers it in `model.loras`. Saving uses
  `llama_adapter_lora_save(adapter, path)`.
- **Parameter registration**: besides the model's own parameters,
  `llama_context::opt_init()` now also registers the attached adapter's A/B via
  `ggml_set_param()`; the quantized base tensors are skipped automatically because they are
  not F32 (so they stay frozen).
- **Graph size**: the backward graph is roughly twice the forward graph, so
  `graph_max_nodes()` doubles the node budget while training (`opt_reserve` / live
  `opt_ctx`), avoiding the `cgraph->n_nodes < cgraph->size` assertion.
- **Repack disabled**: the tool automatically sets `no_extra_bufts` (same as `-nr`).
  Repacked weight buffer types do not implement `GGML_OP_OUT_PROD`, which the backward pass
  needs for the activation gradient (`out_prod(quantized_weight, grad)`); native quantized
  buffers support it. F32 models are unaffected.
- **Context alignment**: `llama_context` rounds `n_ctx` up to a multiple of 256, so the
  training text must be longer than the effective context (the tool reports a clear error).

---

## 6. Limitations

- **MoE (mixture-of-experts) models are now supported**: to support `SmallThinker-4BA0.6B`, this
  work added the ggml backward ops that were missing: `GGML_OP_MUL_MAT_ID_BACK_AS/_B` (3D expert
  adapters and activation gradients), `REGLU`/`SIGMOID`/`TANH` activations, `CLAMP` backward,
  broadcast `DIV` backward, and batched-index `GET_ROWS_BACK`. Both the MoE router (`*_inp`) and
  the experts (`*_exps`) can be trained.
- Only the CPU path has been validated; CUDA/Metal/etc. are untested (the new backward ops are
  implemented for the CPU backend only).
- Full fine-tuning (`-lora-rank 0`) is only meaningful for F32 models; on quantized models it
  degrades to "no trainable parameters".
- Repacking is disabled during training (slightly slower forward, training only).
- Training supports a single `--file`/`-p` text, or `.jsonl` datasets via
  `--dataset`/`--dataset-val` (`text` / `messages` / `conversations`, with assistant-only loss
  masking for chat data).

---

## 7. Verified

On `Qwen2.5-0.5B-Q4_K_M.gguf` (quantized base):

- `Lumina-Engine-train` for 1 epoch: loss ≈ `2.55 → 1.99`;
- the produced adapter has `288` tensors (`144` A/B pairs), correct metadata/shapes, non-zero weights;
- `Lumina-Engine-completion.exe --lora <adapter>` loads it and generates coherent text (`... is Paris.`);
- `.gguf.lumina` works as a base directly (transparent decompression).

On `SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf.lumina` (MoE base):

- `Lumina-Engine-train` for 1 epoch: loss ≈ `1.93 → 1.44`;
- the adapter has `384` tensors (`192` pairs), of which `192` are 3D expert adapters (e.g. `[1536, 4, 32]`);
- `Lumina-Engine-completion.exe --lora <adapter>` loads it and runs inference (MoE `MUL_MAT_ID` is backpropagatable).

### Build 11: datasets / validation split / checkpoints

On `SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf` (MoE base) using slices from `D:/Dataset/data`:

- `stage1_pt` (`text`): dataset loading and next-token training work;
- `stage2_sft` (`messages`) and `stage2_sft_sharegpt` (`conversations`): conversations are rendered
  with the chat template and the record/token statistics are correct;
- `stage3_persona` (system/user/assistant): the separate `--dataset-val` takes effect (5 train / 5 val
  datapoints) and with `--mask-history` about `231 / 968` tokens are trainable (the assistant turns);
- checkpoints are written every `2` steps with `--keep-last 2` pruning the older ones; `--resume`
  restores the newest checkpoint and continues training; the final LoRA adapter loads in the
  inference tool and generates coherent text.

