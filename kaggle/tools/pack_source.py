#!/usr/bin/env python3
"""Pack the Lumina-Engine source tree into kaggle/src/Lumina-Engine-src.zip.

The archive is what you upload to Kaggle (as part of the LuminaMoE-V1 dataset).  The
notebook only needs this one file: it contains

    llama.cpp-b10991/...        the engine source
    kaggle/...                  the helper scripts (build_cuda.sh, run_train.sh, ...)
                                and the notebook itself

so a single upload is enough.

Everything that is not needed to compile the engine on Linux is stripped:
pre-existing build trees, test models, docs, examples, benches and VCS metadata.

Usage:
    python kaggle/tools/pack_source.py [--src DIR] [--out ZIP]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KAGGLE = HERE.parent
REPO = KAGGLE.parent

DEFAULT_SRC = REPO / "llama.cpp-b10991"
DEFAULT_OUT = KAGGLE / "src" / "Lumina-Engine-src.zip"

# Top-level entries that are never needed by a Linux cmake build of the tools.
EXCLUDE_TOP_DIRS = {
    "build", "build-release", "build-clang", "build-debug", "build-native",
    "models", "docs", "media", "benches", "tests", "examples", "app",
    ".git", ".github", ".devops", ".gemini", ".pi", ".vs", ".cache",
    "__pycache__", ".pytest_cache", "node_modules",
}
EXCLUDE_TOP_FILES = {
    ".gitmodules", ".gitattributes", ".dockerignore", "AGENTS.md", "CLAUDE.md",
}
EXCLUDE_SUFFIXES = (
    ".pyc", ".pyo", ".o", ".obj", ".a", ".so", ".dll", ".exe", ".pdb",
    ".gguf", ".lumina", ".zip", ".log",
)
# everything in kaggle/ except the archive we are writing (and caches)
KAGGLE_EXCLUDE_DIRS = {"src", "__pycache__", ".ipynb_checkpoints"}

# Kaggle's dataset ingester rejects zip entries whose name contains '[' or ']'
# ("contains a forbidden character in name"); Windows-invalid characters are listed
# for good measure. Nothing the Linux build needs lives in such a path -- the only
# offenders are Svelte route files under tools/ui/src, and the UI is built with
# -DLLAMA_BUILD_UI=OFF anyway (tools/ui/CMakeLists.txt + scripts/ui-assets.cmake are
# kept, they generate an empty placeholder ui.cpp without npm).
FORBIDDEN_CHARS = set('[]?*:<>|"' + chr(92))

# Trees that exist only for the optional npm UI build.
EXCLUDE_TOP_SUBDIRS = {
    ("tools", "ui", "src"),
}

# x86 *build* trees are already covered by EXCLUDE_TOP_DIRS; guard against others.
def excluded(rel: Path) -> bool:
    parts = rel.parts
    if not parts:
        return False
    top = parts[0]
    if top in EXCLUDE_TOP_DIRS:
        return True
    if len(parts) == 1 and top in EXCLUDE_TOP_FILES:
        return True
    name = parts[-1]
    if name.endswith(EXCLUDE_SUFFIXES):
        return True
    if any(ch in FORBIDDEN_CHARS for ch in rel.as_posix()):
        return True
    for sub in EXCLUDE_TOP_SUBDIRS:
        if parts[:len(sub)] == sub:
            return True
    if "CMakeFiles" in parts or name == "CMakeCache.txt":
        return True
    if parts[0] == "build" or parts[0].startswith("build-"):
        return True
    return False


def collect_kaggle(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        cur = Path(dirpath)
        rel_dir = cur.relative_to(root)
        dirnames[:] = sorted(d for d in dirnames if d not in KAGGLE_EXCLUDE_DIRS)
        for fn in sorted(filenames):
            if fn.endswith((".pyc", ".pyo", ".log")):
                continue
            files.append(rel_dir / fn)
    return files


def collect(src: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(src):
        cur = Path(dirpath)
        rel_dir = cur.relative_to(src)
        dirnames[:] = sorted(d for d in dirnames if not excluded(rel_dir / d))
        for fn in sorted(filenames):
            rel = rel_dir / fn
            if not excluded(rel):
                files.append(rel)
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-kaggle", action="store_true",
                    help="only pack the engine source (default: also pack kaggle/ helper scripts)")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    with_kaggle = not args.no_kaggle

    if not (src / "CMakeLists.txt").is_file():
        print(f"[pack] not a source tree: {src}", file=sys.stderr)
        return 2

    files = collect(src)
    out.parent.mkdir(parents=True, exist_ok=True)

    kaggle_files = collect_kaggle(KAGGLE) if with_kaggle else []

    # sanity: the archive must not contain names Kaggle refuses to ingest
    for rel in list(files) + list(kaggle_files):
        if any(ch in FORBIDDEN_CHARS for ch in rel.as_posix()):
            print(f"[pack] ERROR forbidden character in name: {rel}", file=sys.stderr)
            return 3
    total = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as zf:
        for rel in files:
            path = src / rel
            arc = Path(src.name) / rel
            try:
                zf.write(path, arc.as_posix())
                total += path.stat().st_size
            except OSError as exc:  # vanished/symlink loops -> skip
                print(f"[pack] skip {rel}: {exc}", file=sys.stderr)
        for rel in kaggle_files:
            path = KAGGLE / rel
            try:
                zf.write(path, (Path("kaggle") / rel).as_posix())
                total += path.stat().st_size
            except OSError as exc:
                print(f"[pack] skip kaggle/{rel}: {exc}", file=sys.stderr)

    data = out.read_bytes()
    print(f"[pack] source : {src}")
    print(f"[pack] archive: {out}")
    ui_dir = src / "tools" / "ui" / "src"
    if ui_dir.is_dir():
        n_ui = sum(len(f) for _, _, f in os.walk(ui_dir))
        print(f"[pack] skipped {n_ui} files under tools/ui/src (npm UI app; not needed with "
              f"-DLLAMA_BUILD_UI=OFF and the source of the '[' names Kaggle rejects)")
    print(f"[pack] files  : {len(files)} source + {len(kaggle_files)} kaggle helper files  "
          f"raw {total/1048576:.1f} MiB  "
          f"zip {len(data)/1048576:.1f} MiB")
    print(f"[pack] sha256 : {hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
