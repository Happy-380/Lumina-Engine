#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build Lumina-Engine (llama.cpp b-10991 fork, Build 31) on Linux / Kaggle.
#
#   - CUDA backend ON (falls back to a CPU-only build if the CUDA toolchain is
#     missing or the CUDA configure step fails, so training can always start)
#   - static binaries: no LD_LIBRARY_PATH juggling, easy to cache between
#     Kaggle sessions (save the notebook output as a dataset and point
#     LUMINA_BIN_DIR at it next time)
#
# Produces:
#   $LUMINA_BIN_DIR/Lumina-Engine-train    QLoRA trainer (the tool we need)
#   $LUMINA_BIN_DIR/Lumina-Engine-cli      inference / smoke test
#   $LUMINA_BIN_DIR/build-info.txt         what was built
#
# Usage (on Kaggle, from a notebook cell):
#   bash build_cuda.sh
#
# Environment knobs:
#   LUMINA_SRC        source directory or .zip (auto-detected when unset)
#   LUMINA_WORK       scratch root                (default /kaggle/working/lumina)
#   LUMINA_BUILD      cmake build dir             (default $LUMINA_WORK/build)
#   LUMINA_BIN_DIR    where the binaries land     (default $LUMINA_WORK/bin)
#   LUMINA_CUDA       1 = CUDA build, 0 = CPU only (default 1)
#   LUMINA_CUDA_ARCH  e.g. "75"; "auto" detects from the GPU (default auto)
#   LUMINA_JOBS       parallel build jobs         (default nproc)
#   LUMINA_REBUILD    1 = ignore an existing build directory (default 0)
#   LUMINA_BUILD_CLI  1 = also build Lumina-Engine-cli (default 1)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

c_info() { printf '\n\033[1;36m[build]\033[0m %s\n' "$*"; }
c_ok()   { printf '\033[1;32m[build]\033[0m %s\n' "$*"; }
c_warn() { printf '\033[1;33m[build][warn]\033[0m %s\n' "$*"; }
c_err()  { printf '\033[1;31m[build][error]\033[0m %s\n' "$*" >&2; }
die()    { c_err "$*"; exit 1; }

LUMINA_WORK="${LUMINA_WORK:-/kaggle/working/lumina}"
BUILD_DIR="${LUMINA_BUILD:-$LUMINA_WORK/build}"
CPU_BUILD_DIR="${LUMINA_CPU_BUILD:-$LUMINA_WORK/build-cpu}"
BIN_DIR="${LUMINA_BIN_DIR:-$LUMINA_WORK/bin}"
JOBS="${LUMINA_JOBS:-$( (nproc 2>/dev/null || echo 4) )}"
WANT_CUDA="${LUMINA_CUDA:-1}"
CUDA_ARCH="${LUMINA_CUDA_ARCH:-auto}"
REBUILD="${LUMINA_REBUILD:-0}"
BUILD_CLI="${LUMINA_BUILD_CLI:-1}"
INPUT_DIRS="${LUMINA_INPUT_DIRS:-/kaggle/input}"

mkdir -p "$LUMINA_WORK" "$BIN_DIR"

PY="${LUMINA_PYTHON:-python3}"
if ! "$PY" -c 'import sys' >/dev/null 2>&1; then
    PY=python
fi

# --------------------------------------------------------------- source tree
find_source() {
    local cand
    if [[ -n "${LUMINA_SRC:-}" ]]; then
        printf '%s\n' "$LUMINA_SRC"; return 0
    fi
    for cand in \
        "$SCRIPT_DIR/src/llama.cpp-b10991" \
        "$SCRIPT_DIR/llama.cpp-b10991" \
        "$SCRIPT_DIR/../llama.cpp-b10991"
    do
        [[ -f "$cand/CMakeLists.txt" ]] && { printf '%s\n' "$cand"; return 0; }
    done
    local zip
    for zip in "$SCRIPT_DIR/src/Lumina-Engine-src.zip" "$SCRIPT_DIR/Lumina-Engine-src.zip"; do
        [[ -f "$zip" ]] && { printf '%s\n' "$zip"; return 0; }
    done
    # last resort: search the read-only input tree
    local hit
    hit=$(find "$INPUT_DIRS" -maxdepth 6 \( -name 'llama.cpp-b10991' -o -name 'Lumina-Engine-src.zip' \) 2>/dev/null | head -n 1 || true)
    [[ -n "$hit" ]] && { printf '%s\n' "$hit"; return 0; }
    return 1
}

SRC_RAW="$(find_source || true)"
[[ -n "$SRC_RAW" ]] || die "no source tree found. Put Lumina-Engine-src.zip next to this script (kaggle/src/) or set LUMINA_SRC."

if [[ -d "$SRC_RAW" ]]; then
    SRC="$SRC_RAW"
elif [[ -f "$SRC_RAW" && "${SRC_RAW,,}" == *.zip ]]; then
    SRC="$LUMINA_WORK/src/llama.cpp-b10991"
    if [[ ! -f "$SRC/CMakeLists.txt" ]]; then
        c_info "extracting source archive $(basename "$SRC_RAW")"
        mkdir -p "$LUMINA_WORK/src"
        rm -rf "$LUMINA_WORK/src/llama.cpp-b10991"
        "$PY" - "$SRC_RAW" "$LUMINA_WORK/src" <<'PY'
import sys, zipfile
zf, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zf) as z:
    z.extractall(out)
PY
    fi
else
    die "LUMINA_SRC / $SRC_RAW is neither a directory nor a .zip"
fi
[[ -f "$SRC/CMakeLists.txt" ]] || die "no CMakeLists.txt in $SRC"

# Kaggle images may not expose nvcc on PATH even when it is installed.
for cuda_bin in /usr/local/cuda/bin /usr/local/cuda-12*/bin /usr/local/cuda-13*/bin; do
    if [[ -d "$cuda_bin" ]]; then
        PATH="$cuda_bin:$PATH"
    fi
done
export PATH

c_info "source  : $SRC"
c_info "work    : $LUMINA_WORK"
c_info "build   : $BUILD_DIR"
c_info "bin     : $BIN_DIR"
c_info "jobs    : $JOBS"

# ------------------------------------------------------------------ toolchain
if ! command -v cmake >/dev/null 2>&1; then
    c_warn "cmake not found, installing with pip"
    "$PY" -m pip install --quiet cmake || die "could not install cmake"
fi
if ! command -v ninja >/dev/null 2>&1; then
    "$PY" -m pip install --quiet ninja >/dev/null 2>&1 || true
fi
command -v ninja >/dev/null 2>&1 && GEN="Ninja" || GEN="Unix Makefiles"
command -v g++ >/dev/null 2>&1 || command -v c++ >/dev/null 2>&1 || die "no C++ compiler (g++) available"
c_info "generator: $GEN   cmake: $(cmake --version | head -n1)"

# ------------------------------------------------------------------- CUDA arch
detect_arch() {
    "$PY" - <<'PY' 2>/dev/null || true
import sys
try:
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print(f"{major}{minor}")
        sys.exit(0)
except Exception:
    pass
PY
}
if [[ "$CUDA_ARCH" == "auto" ]]; then
    CUDA_ARCH="$(detect_arch || true)"
    if [[ -z "$CUDA_ARCH" ]] && command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .' || true)"
    fi
fi
if [[ -n "$CUDA_ARCH" && "$CUDA_ARCH" =~ ^[0-9]+$ ]] && command -v nvcc >/dev/null 2>&1; then
    nvcc_major="$(nvcc --version 2>/dev/null | sed -n 's/.*release //p' | cut -d. -f1 | tr -dc '0-9')"
    if [[ -n "$nvcc_major" && "$nvcc_major" -ge 13 && "$CUDA_ARCH" -lt 70 ]]; then
        c_warn "CUDA $nvcc_major no longer supports sm_$CUDA_ARCH (Pascal/Maxwell); using the CMake default arch list"
        CUDA_ARCH=""
    fi
fi
if [[ -n "$CUDA_ARCH" ]]; then
    c_info "CUDA arch: sm_$CUDA_ARCH"
else
    c_warn "could not detect the GPU compute capability; letting CMake pick a multi-arch default"
fi

# ----------------------------------------------------------------- configure
configure() {  # $1 = build dir, $2 = 1|0 cuda
    local dir="$1" cuda="$2"
    mkdir -p "$dir"
    local args=(
        -S "$SRC" -B "$dir" -G "$GEN"
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=OFF
        -DGGML_NATIVE=ON
        -DGGML_BACKEND_DL=OFF
        -DGGML_CPU_ALL_VARIANTS=OFF
        -DGGML_RPC=OFF
        -DLLAMA_CURL=OFF
        -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_EXAMPLES=OFF
        -DLLAMA_BUILD_APP=OFF
        -DLLAMA_BUILD_SERVER=ON
        -DLLAMA_BUILD_TOOLS=ON
        -DLLAMA_BUILD_UI=OFF
        -DLLAMA_USE_PREBUILT_UI=OFF
    )
    if [[ "$cuda" == "1" ]]; then
        args+=(
            -DGGML_CUDA=ON
            -DGGML_CUDA_FA=ON
            -DGGML_CUDA_FA_ALL_QUANTS=OFF
            -DGGML_CUDA_NCCL=OFF
            -DGGML_CUDA_GRAPHS=ON
        )
        [[ -n "$CUDA_ARCH" ]] && args+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
    else
        args+=(-DGGML_CUDA=OFF)
    fi
    cmake "${args[@]}"
}

HAVE_CUDA_TOOLCHAIN=0
if [[ "$WANT_CUDA" == "1" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
        HAVE_CUDA_TOOLCHAIN=1
        c_info "nvcc   : $(nvcc --version | tail -n1)"
    else
        c_warn "nvcc not found -> building a CPU-only binary (set LUMINA_CUDA=1 after installing CUDA)"
    fi
fi

if [[ "$REBUILD" == "1" ]]; then
    rm -rf "$BUILD_DIR" "$CPU_BUILD_DIR"
fi

BUILD_MODE=cpu
if [[ "$HAVE_CUDA_TOOLCHAIN" == "1" ]]; then
    c_info "configuring CUDA build (this is the slow part: 20-60 min on 4 vCPU)"
    if configure "$BUILD_DIR" 1 2>&1 | tail -n 25; then
        BUILD_MODE=cuda
    else
        c_warn "CUDA configure failed, retrying without CUDA"
        rm -rf "$BUILD_DIR"
    fi
fi
if [[ "$BUILD_MODE" != "cuda" ]]; then
    BUILD_DIR="$CPU_BUILD_DIR"
    c_info "configuring CPU-only build"
    configure "$BUILD_DIR" 0 2>&1 | tail -n 25 || die "CPU-only configure failed (see the cmake output above)"
fi

# --------------------------------------------------------------------- build
TARGETS=(llama-train)
[[ "$BUILD_CLI" == "1" ]] && TARGETS+=(llama-cli)

c_info "building targets: ${TARGETS[*]}"
if ! cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" -j "$JOBS"; then
    if [[ "$BUILD_MODE" == "cuda" ]]; then
        c_warn "CUDA build failed, falling back to a CPU-only build"
        BUILD_DIR="$CPU_BUILD_DIR"
        rm -rf "$BUILD_DIR"
        configure "$BUILD_DIR" 0 2>&1 | tail -n 15 || die "CPU-only configure failed"
        BUILD_MODE=cpu
        cmake --build "$BUILD_DIR" --target "${TARGETS[@]}" -j "$JOBS" || die "CPU build failed too"
    else
        die "build failed"
    fi
fi

# ------------------------------------------------------------------ install
copy_bin() {  # $1 = name
    local exe
    exe="$(find "$BUILD_DIR" -maxdepth 3 -type f -name "$1" -perm -u+x | head -n1 || true)"
    [[ -n "$exe" ]] || die "built executable $1 not found in $BUILD_DIR"
    cp -f "$exe" "$BIN_DIR/$1"
    chmod +x "$BIN_DIR/$1"
    command -v strip >/dev/null 2>&1 && strip "$BIN_DIR/$1" 2>/dev/null || true
}
copy_bin Lumina-Engine-train
[[ "$BUILD_CLI" == "1" ]] && copy_bin Lumina-Engine-cli

{
    echo "built        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "mode         : $BUILD_MODE"
    echo "cuda arch    : ${CUDA_ARCH:-auto}"
    echo "nvcc         : $(command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -n1 || echo none)"
    echo "cmake        : $(cmake --version | head -n1)"
    echo "compiler     : $( (c++ --version 2>/dev/null || g++ --version) | head -n1)"
    echo "source       : $SRC"
    echo "build dir    : $BUILD_DIR"
    echo "host         : $(uname -srm)"
} > "$BIN_DIR/build-info.txt"

c_ok "binaries:"
ls -lh "$BIN_DIR" | sed 's/^/         /'
c_info "build-info.txt:"
sed 's/^/         /' "$BIN_DIR/build-info.txt"

if command -v ldd >/dev/null 2>&1; then
    missing="$(ldd "$BIN_DIR/Lumina-Engine-train" 2>/dev/null | grep 'not found' || true)"
    if [[ -n "$missing" ]]; then
        c_warn "the trainer links against libraries that are not on the default search path:"
        printf '%s
' "$missing" | sed 's/^/         /'
        c_warn "if it fails to start, export LD_LIBRARY_PATH=/usr/local/cuda/lib64"
    fi
fi

c_info "verifying the trainer starts"
if ! "$BIN_DIR/Lumina-Engine-train" --version 2>&1 | head -n 20; then
    die "the freshly built trainer does not start"
fi
c_ok "done: $BIN_DIR/Lumina-Engine-train ($BUILD_MODE build)"
