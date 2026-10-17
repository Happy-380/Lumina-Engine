# Compressing a model & using the compressed model (User guide)

> 🌐 [中文](ModelCompression.md) | **English**

This page is purely practical: **how to losslessly shrink a GGUF file**, and **how to load the
compressed file with Lumina-Engine**.  For the container format and internals see
[LosslessCompression](LosslessCompression.en.md).

In one sentence: **shrink it with `tools/lumina_gguf_pack.py` (default output `A.gguf.lumina`, the
original file is left untouched); Lumina-Engine (Build 8+) detects and loads both `.gguf` and
`.gguf.lumina` automatically, byte-for-byte unchanged.**

---

## 0. Requirements

| Item | Requirement |
|------|-------------|
| OS | Windows 10 / 11 x86-64 |
| Python | 3.8+ with `numpy` |
| Engine | **Lumina-Engine Build 8 or newer** (`Lumina-Engine-Bin\`) |
| Tensor types | `F32`, `Q4_K`, `Q6_K` (this repo's `SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf` qualifies) |

> Packing is an **offline, one-time** step and needs no GPU.  Inference speed, memory and accuracy
> are identical to the uncompressed model.

---

## 1. Compress a model

### 1.1 Install the dependency

```bat
python -m pip install numpy
```

### 1.2 Pack

The default output is `<input>.lumina` (i.e. `A.gguf` -> `A.gguf.lumina`), and the **original file
is left untouched**:

```bat
python tools\lumina_gguf_pack.py  <input.gguf>
REM equivalent to: python tools\lumina_gguf_pack.py <input.gguf> <input.gguf>.lumina
```

You can also give an explicit output path:

```bat
python tools\lumina_gguf_pack.py  <input.gguf>  <output.gguf.lumina>
```

Example (the model shipped in this repo):

```bat
python tools\lumina_gguf_pack.py SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf
```

> ⚠️ **Note**: the repo root keeps **both** files:
> `SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf` is the **original, uncompressed** model and
> `SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf.lumina` is the **compressed container**.  Pack the
> original `.gguf`; packing the container itself fails with
> `input is already a Lumina compressed GGUF`.

Example output:

```
packed SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf
  original : 2630213408 bytes (2.45 GiB)
  packed   : 2479779960 bytes (2.31 GiB)
  ratio    : 94.28%  (saved 143.5 MiB)
  output   : SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf.lumina
  original left untouched; load either file directly with Lumina-Engine.
```

The optional third argument is the zlib level (`1`–`9`, default `9`; higher = smaller but slower):

```bat
python tools\lumina_gguf_pack.py model.gguf model.custom.lumina 9
```

### 1.3 Verify (strongly recommended)

Confirm the restored file is **byte-for-byte identical** (SHA-256) to the original:

```bat
python tools\lumina_gguf_unpack.py verify  <packed.gguf.lumina>  <original.gguf>
```

Expected output:

```
verify: OK fb584d0b0f79f9e9944f91ac061e24726a8d014b6702213bfd5585729db7ab61 fb584d0b...
```

`OK` means lossless; `FAIL` means a corrupt file - do not use it.

### 1.4 Keep both, or replace the original (optional)

Packing **does not touch the original**, so you can simply keep both and use whichever you need:

```
model.gguf          <- original model (also readable by stock llama.cpp / other tools)
model.gguf.lumina   <- compressed container (Lumina-Engine only; loaded via the magic)
```

If you really want only one file, verify first and then replace (note that stock tools will no
longer read the result):

```bat
python tools\lumina_gguf_unpack.py verify model.gguf.lumina model.gguf
move /y model.gguf model.gguf.bak          REM or just delete it
move /y model.gguf.lumina model.gguf       REM optional: keep loading from the .gguf path
```

> The engine detects the `LGGC` magic in the file header, **not the extension**: both `.gguf` and
> `.gguf.lumina` load directly.

---

## 2. Use the compressed model

**No extra flags are needed.**  Just pass the packed path (`.gguf.lumina`) like any other model.
On the first load the engine decompresses and caches it automatically.  The original `.gguf` works
as before.

```bat
REM load the compressed container (recommended)
Lumina-Engine-cli.exe -m model.gguf.lumina -p "Explain quantum computing simply." -n 256 -t 6

REM loading the original .gguf also works (the engine keys off the magic, not the extension)
Lumina-Engine-cli.exe -m model.gguf -p "Explain quantum computing simply." -n 256 -t 6

REM benchmark / perplexity / server take the same path
Lumina-Engine-bench.exe      -m model.gguf.lumina -p 512 -n 128 -t 6
Lumina-Engine-perplexity.exe -m model.gguf.lumina -f docx\whitepaper.txt -c 512 -t 6
Lumina-Engine-server.exe     -m model.gguf.lumina -c 4096
```

### What happens on load

1. the engine reads the header, sees the `LGGC` magic → treats the file as a lossless container;
2. streams it out and rebuilds a GGUF that is **exactly the original file**;
3. writes it to `%TEMP%\lumina-<hash>.gguf`;
4. continues through the normal loading / inference path.  The temp file is keyed by path + size,
   so **later loads reuse it** without decompressing again.

### Disk and timing

| Item | Note |
|------|------|
| First load | writes the temp file once (~the original model size).  For 2.63 GB this is about 30 seconds here. |
| Later loads | reuse the temp file; no extra cost. |
| Cleanup | `%TEMP%\lumina-*.gguf` can be deleted at any time; it is rebuilt on the next load. |

Set `LUMINA_VERBOSE=1` to print decompression logs.

> Tight on disk?  Replacing the original saves ~143.5 MiB, but the first run still needs ~2.63 GB
> free in the temp directory.

---

## 3. How it works (one paragraph)

Every 256-weight k-quant block interleaves low-entropy `d/dmin/scales` with high-entropy 4/6-bit
weight streams.  The packer **splits the fields apart** (and, for the 4-bit quants, sorts sub-blocks
by their `scale`/`min` context) before DEFLATE so they do not pollute each other, and the engine
merges them back on load.  **No quantization, no pruning** - the weight values are untouched.

Measured on this model: `2,630,213,408 B → 2,479,779,960 B` (**94.28%**, **143.5 MiB saved**);
the restored SHA-256 matches.

---

## 4. FAQ

**Q1: Can stock llama.cpp or other tools open the packed file?**
No. It is a Lumina-specific container (magic `LGGC`) that only **Lumina-Engine Build 8+** can load
transparently.  Use `lumina_gguf_unpack.py` to get the original GGUF back.

**Q2: Does it lose precision or change output?**
No. The restored file is **byte-for-byte identical** (check with `verify`), so output is unchanged.

**Q3: Why only a few percent smaller?**
The weights are already quantized: the Q6_K quants are close to full entropy (~6 bit) and the Q4_K
quants have a conditional entropy of ~3.82/4 bit, so the lossless ceiling is only a few percent.
v2 already pushes the Q4_K stream to 95.5% (near its conditional-entropy limit), so with
"no quantization, no pruning" there is little left to gain.

**Q4: Does it support every model?**
The packer currently supports tensor types `F32` / `Q4_K` / `Q6_K`; anything else raises an error.
To add more, extend `split_tensor` in `tools/lumina_gguf_pack.py` and the matching decoder.

**Q5: Where is the temp file, is it safe?**
`%TEMP%\lumina-<hash>.gguf`, created by the engine, safe to delete.  Your model file is never modified.

**Q6: What if the packed file is bigger?**
For very small files (e.g. a few-MB vocab-only GGUF) the container header overhead can outweigh the
savings - just keep the original.

**Q7: Packing feels slow.**
It is pure Python + zlib; a 2.6 GB model takes ~3 minutes (mostly DEFLATE and the context sort).  Use `level 6` for speed.

**Q8: Why the `.gguf.lumina` suffix?**
So the original model and the compressed container are easy to tell apart.  The engine keys off the
`LGGC` magic, **not the extension**: both `.gguf` and `.gguf.lumina` load directly (any other suffix
works too, it is just not the convention).

---

## 5. One-click helper (optional)

Create `compress_model.bat` in the repo root:

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

Usage: `compress_model.bat SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf`

---

## 6. Appendix: script CLI cheat sheet

All scripts are pure Python (only `numpy` is required) and are run from the repo root.

### 6.1 Pack (compress)

```bat
python tools\lumina_gguf_pack.py <input.gguf> [output] [level]
```

| Argument | Meaning |
|----------|---------|
| `<input.gguf>` | original, uncompressed GGUF; an `LGGC` container is rejected |
| `[output]` | optional; defaults to `<input>.lumina` (`A.gguf` -> `A.gguf.lumina`), **original kept** |
| `[level]` | optional; zlib level `1`–`9`, default `9` (higher = smaller but slower) |

Examples:

```bat
python tools\lumina_gguf_pack.py model.gguf
python tools\lumina_gguf_pack.py model.gguf model.custom.lumina 6
```

### 6.2 Restore / verify (unpack)

```bat
REM restore to a plain GGUF file
python tools\lumina_gguf_unpack.py <packed.gguf.lumina> <output.gguf>

REM byte-exact check (recommended): prints OK / FAIL, exit code 0 on OK
python tools\lumina_gguf_unpack.py verify <packed.gguf.lumina> <original.gguf>
```

Example:

```bat
python tools\lumina_gguf_unpack.py verify model.gguf.lumina model.gguf
```

> The unpacker reads both container v1 and v2.

### 6.3 Inspect GGUF metadata / tensors (helper)

```bat
REM print GGUF metadata key/value pairs
python tools\read_gguf.py <model.gguf>

REM print each tensor's name / type / dims, plus type counts
python tools\tensors.py <model.gguf>
```

> These two read a plain GGUF (or a restored file); reading an `LGGC` container directly is
> meaningless - restore it with `lumina_gguf_unpack.py` first.

---

## See also

- [LosslessCompression](LosslessCompression.en.md) - container format and internals
- [Building](Building.en.md) - building from source (incl. Build 8)
- [Delivery](Delivery.en.md) - the `Lumina-Engine-Bin` delivery directory
