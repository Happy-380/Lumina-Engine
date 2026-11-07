#!/usr/bin/env python3
r"""Locate the Lumina-Engine training inputs inside a Kaggle /kaggle/input tree.

Kaggle extracts an uploaded `Dataset.zip` without telling us how deep the members
end up, and the notebook may see

    /kaggle/input/<dataset>/Dataset/SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf
    /kaggle/input/<dataset>/SmallThinker-4BA0.6B-Instruct.Q4_K_M.gguf
    /kaggle/input/<dataset>/data/stage1_pt/train.jsonl
    /kaggle/input/<dataset>/Dataset.zip                      (nested, not extracted)
    ...

so instead of hard-coding a single layout this script searches for the pieces by
*content* (file name, size, sibling files) and reports what it found.

    python3 kaggle_paths.py                 # human readable report
    python3 kaggle_paths.py --shell         # KEY='value' lines, for `eval`
    python3 kaggle_paths.py --json          # machine readable

Any of MODEL / TRAIN / VAL / CKPT can be pinned with an environment variable
(LUMINA_MODEL, LUMINA_TRAIN, LUMINA_VAL, LUMINA_CKPT) and the search is skipped for
that item.

Exit code is 0 when the base model and the training dataset were found.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import zipfile
from pathlib import Path

CKPT_RE = re.compile(r"^(?P<prefix>.+)\.ckpt-(?P<step>\d+)\.gguf$", re.IGNORECASE)
MIN_MODEL_BYTES = 200 * 1024 * 1024          # a real base model is >= 200 MiB
SKIP_DIRS = {
    "__pycache__", ".git", ".ipynb_checkpoints", "node_modules", "site-packages",
    ".tools", "tokenizer", "persona_style",
}
MAX_DEPTH = 14


def log(msg: str = "") -> None:
    print(msg, flush=True)


def input_roots() -> list[Path]:
    env = os.environ.get("LUMINA_INPUT_DIRS", "/kaggle/input")
    return [Path(p) for p in env.split(os.pathsep) if p.strip() and Path(p).is_dir()]


def walk(roots: list[Path]):
    seen: set[str] = set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            cur = Path(dirpath)
            try:
                depth = len(cur.relative_to(root).parts)
            except ValueError:
                depth = 0
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and depth < MAX_DEPTH)
            for fn in sorted(filenames):
                p = cur / fn
                if str(p) in seen:
                    continue
                seen.add(str(p))
                yield p


def size_of(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return -1


def human(n: int) -> str:
    return f"{n/2**30:.2f} GiB" if n >= 2**30 else f"{n/2**20:.1f} MiB"


# --------------------------------------------------------------------------- model

def find_model(files: list[Path]) -> tuple[Path | None, str]:
    cands = []
    for p in files:
        if p.suffix.lower() != ".gguf":
            continue
        name = p.name.lower()
        if CKPT_RE.match(p.name) or "lora" in name or ".ckpt-" in name:
            continue
        size = size_of(p)
        if size < MIN_MODEL_BYTES:
            continue
        score = 0
        path_s = str(p).lower()
        if "smallthinker" in name or "smallthinker" in path_s:
            score += 100
        if "instruct" in name:
            score += 5
        if "q4_k_m" in name:
            score += 5
        cands.append((score, size, p))
    if not cands:
        return None, "no .gguf >= 200 MiB that is not a LoRA / checkpoint"
    cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
    _, size, p = cands[0]
    return p, f"{human(size)}, {p.name}"


# ------------------------------------------------------------------------ datasets

def find_dataset(files: list[Path]) -> tuple[Path | None, Path | None, str]:
    """Return (train.jsonl, val.jsonl, note) for the stage-1 continued-pretraining set."""
    cands = [p for p in files if p.name.lower() == "train.jsonl"]
    if not cands:
        return None, None, "no file named train.jsonl"

    def score(p: Path) -> tuple:
        s = str(p).lower()
        sc = 0
        if "stage1_pt" in s:
            sc += 100
        elif "stage1" in s:
            sc += 40
        if "stage2" in s or "stage3" in s:
            sc -= 200
        if p.with_name("val.jsonl").is_file():
            sc += 20
        return (sc, size_of(p))

    cands.sort(key=score, reverse=True)
    best = cands[0]
    val = best.with_name("val.jsonl")
    if val.is_file():
        return best, val, f"{human(size_of(best))}, train + val in the same directory"
    for p in files:
        if p.name.lower() == "val.jsonl" and "stage1" in str(p).lower():
            return best, p, f"{human(size_of(best))}, val from a different directory"
    return best, None, f"{human(size_of(best))}, no val.jsonl (trainer falls back to --val-split)"


# --------------------------------------------------------------------- checkpoints

def list_checkpoints(files: list[Path]) -> list[tuple[Path, int, str]]:
    """All checkpoints as (path, step, prefix), best candidate last."""
    groups: dict[tuple[str, str], list[tuple[int, Path]]] = {}
    for p in files:
        m = CKPT_RE.match(p.name)
        if not m:
            continue
        key = (str(p.parent), m.group("prefix"))
        groups.setdefault(key, []).append((int(m.group("step")), p))

    out = []
    for (_, prefix), items in groups.items():
        items.sort()
        step, path = items[-1]
        out.append((path, step, prefix))
    # prefer stage1 checkpoints, then the highest step
    out.sort(key=lambda t: ("stage1" not in t[2].lower(),
                            Path(t[0]).with_suffix(".json").is_file() is False,
                            t[1]))
    return out


def find_checkpoint(files: list[Path]) -> tuple[Path | None, int, str]:
    cks = list_checkpoints(files)
    if not cks:
        return None, -1, "no <stem>.ckpt-<step>.gguf found"
    path, step, _ = cks[-1]
    note = f"step {step}, {human(size_of(path))}"
    js = path.with_suffix(".json")
    if js.is_file():
        try:
            meta = json.loads(js.read_text())
            note += (f", epoch {int(meta.get('epoch', 0)) + 1}/{meta.get('total_epochs', '?')}"
                     f", rank {meta.get('lora_rank')}, alpha {meta.get('lora_alpha')}"
                     f", targets {meta.get('lora_targets')}")
        except Exception:
            note += ", .json unreadable"
    else:
        note += ", WARNING: .json missing (resume will assume epoch 0)"
    return path, step, note


# ------------------------------------------------------------------- nested zips

def maybe_unpack(dest: Path, min_bytes: int = 0) -> bool:
    """Extract a nested .zip from the input tree into `dest` (Kaggle does not always
    auto-extract a zip that is itself stored inside a dataset folder)."""
    if dest.is_dir() and any(dest.iterdir()):
        return True

    cands = []
    for p in walk(input_roots()):
        if p.suffix.lower() != ".zip" or size_of(p) < min_bytes:
            continue
        try:
            with zipfile.ZipFile(p) as zf:
                names = zf.namelist()
        except Exception:
            continue
        if not any(n.lower().endswith(".gguf") for n in names):
            continue
        cands.append((size_of(p), p))
    if not cands:
        return False
    cands.sort(reverse=True)
    z = cands[0][1]
    log(f"[paths] extracting nested archive {z} ({human(size_of(z))})")
    log(f"[paths]   -> {dest}  (a few minutes)")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        members = [i for i in zf.infolist() if not i.is_dir()]
        total = sum(i.file_size for i in members) or 1
        done = 0
        for i in members:
            zf.extract(i, dest)
            done += i.file_size
            if done * 10 // total != (done - i.file_size) * 10 // total:
                log(f"[paths]   {done*100//total:3d}%")
    return True


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--shell", action="store_true", help="emit KEY='value' lines for eval")
    g.add_argument("--json", action="store_true", help="emit a JSON object")
    ap.add_argument("--work", default=os.environ.get("LUMINA_WORK", "/kaggle/working/lumina"))
    args = ap.parse_args()

    work = Path(args.work)
    out_dir = Path(os.environ.get("LUMINA_OUT", work / "train_out"))
    unpack_dir = Path(os.environ.get("LUMINA_UNPACK_DIR", work / "input"))

    env_model = os.environ.get("LUMINA_MODEL") or None
    env_train = os.environ.get("LUMINA_TRAIN") or None
    env_val   = os.environ.get("LUMINA_VAL") or None
    env_ckpt  = os.environ.get("LUMINA_CKPT") or None

    roots = input_roots()
    files = list(walk(roots))

    model = Path(env_model) if env_model else None
    train = Path(env_train) if env_train else None
    val   = Path(env_val) if env_val else None
    ckpt  = Path(env_ckpt) if env_ckpt else None

    model_note = "pinned by $LUMINA_MODEL" if model else ""
    ds_note    = "pinned by $LUMINA_TRAIN/$LUMINA_VAL" if train else ""
    ck_note    = "pinned by $LUMINA_CKPT" if ckpt else ""
    ck_step    = -1

    def resolve() -> None:
        nonlocal model, train, val, ckpt, model_note, ds_note, ck_note, ck_step
        if model is None:
            model, model_note = find_model(files)
        if train is None:
            train, val, ds_note = find_dataset(files)
        elif val is None and train.with_name("val.jsonl").is_file():
            val = train.with_name("val.jsonl")
        if ckpt is None:
            ckpt, ck_step, ck_note = find_checkpoint(files)
        elif ckpt:
            m = CKPT_RE.match(ckpt.name)
            ck_step = int(m.group("step")) if m else -1
            ck_note = f"step {ck_step}"

    resolve()
    if model is None or train is None:
        # the zip was uploaded as-is and Kaggle did not extract it
        if maybe_unpack(unpack_dir):
            files = list(walk(roots + [unpack_dir]))
            resolve()

    ckpt_prefix = ""
    if ckpt:
        m = CKPT_RE.match(ckpt.name)
        ckpt_prefix = m.group("prefix") if m else ckpt.stem
    js = ckpt.with_suffix(".json") if ckpt else None

    info = {
        "MODEL": str(model) if model else "",
        "TRAIN": str(train) if train else "",
        "VAL": str(val) if val else "",
        "CKPT_GGUF": str(ckpt) if ckpt else "",
        "CKPT_JSON": str(js) if js and js.is_file() else "",
        "CKPT_DIR": str(ckpt.parent) if ckpt else "",
        "CKPT_PREFIX": ckpt_prefix,
        "CKPT_STEP": ck_step,
        "OUT_DIR": str(out_dir),
        "WORK": str(work),
        "UNPACK_DIR": str(unpack_dir),
    }

    if args.shell:
        for k, v in info.items():
            print(f"{k}={shlex.quote(str(v))}")
        return 0
    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    log("=" * 78)
    log("Lumina-Engine / Kaggle input resolution")
    log("=" * 78)
    log(f"search roots      : {', '.join(str(r) for r in roots) or '(none)'}")
    log(f"files considered  : {len(files)}")
    log(f"base model        : {info['MODEL'] or 'NOT FOUND'}")
    log(f"                    {model_note}")
    log(f"train dataset     : {info['TRAIN'] or 'NOT FOUND'}")
    log(f"val dataset       : {info['VAL'] or '(none)'}")
    log(f"                    {ds_note}")
    log(f"latest checkpoint : {info['CKPT_GGUF'] or '(none - fresh run)'}")
    log(f"                    {ck_note}")
    log(f"checkpoint dir    : {info['CKPT_DIR'] or '-'}  ->  copied to {out_dir}")
    log(f"output dir        : {out_dir}")
    log("=" * 78)
    ok = bool(info["MODEL"] and info["TRAIN"])
    if not ok:
        log("[paths] FAILED: base model and/or training dataset not found")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
