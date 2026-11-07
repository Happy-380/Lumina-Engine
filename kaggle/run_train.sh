#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Resume the Lumina-Engine stage-1 QLoRA run on Kaggle.
#
# This is the *local* command with only the paths changed:
#
#   Lumina-Engine-train -m <base model> --dataset <stage1_pt/train.jsonl> \
#       --dataset-val <stage1_pt/val.jsonl> -c 32768 -b 2048 -ub 128 \
#       --cpu-cores 0 -lr 1e-4 --epochs 1 -lora-rank 64 -lora-alpha 128 \
#       --save-steps 50 --save-every-epochs 1 --keep-last 3 \
#       -o <out>/stage1.lora.gguf --resume
#
# Everything that identifies the run (context, batch, ubatch, rank, alpha,
# target layers, dataset) is kept byte-for-byte identical to the run that
# produced ckpt-10400; changing any of them would invalidate the checkpoint
# (steps_per_epoch / datapoint mapping would no longer line up).
#
# The checkpoint files are *copied* into the output directory first, because
# /kaggle/input is read-only and --resume looks for <out-stem>.ckpt-<step>.gguf
# next to the output file.
#
# Usage:
#   bash run_train.sh --dry-run      # print the resolved command, do nothing
#   bash run_train.sh                # train (honours LUMINA_TIME_BUDGET)
#
# Environment:
#   LUMINA_BIN           trainer binary (default: find Lumina-Engine-train)
#   LUMINA_WORK          scratch root            (default /kaggle/working/lumina)
#   LUMINA_OUT           output + checkpoint dir (default $LUMINA_WORK/train_out)
#   LUMINA_RESUME        1 = pass --resume, 0 = fresh run     (default 1)
#   LUMINA_TIME_BUDGET   seconds; 0 = no limit                (default 0)
#   LUMINA_ALLOW_GPU     1 = LUMINA_TRAIN_ALLOW_GPU + -ngl 99 (default 0, see README)
#   LUMINA_CTX / _BATCH / _UBATCH / _LR / _EPOCHS
#   LUMINA_RANK / _ALPHA / _SAVE_STEPS / _SAVE_EPOCHS / _KEEP_LAST
#   LUMINA_EXTRA_ARGS    extra trainer flags (word-split)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done

c_info() { printf '\n\033[1;36m[train]\033[0m %s\n' "$*"; }
c_ok()   { printf '\033[1;32m[train]\033[0m %s\n' "$*"; }
c_warn() { printf '\033[1;33m[train][warn]\033[0m %s\n' "$*"; }
die()    { printf '\033[1;31m[train][error]\033[0m %s\n' "$*" >&2; exit 1; }

LUMINA_WORK="${LUMINA_WORK:-/kaggle/working/lumina}"
export LUMINA_WORK
OUT_DIR="${LUMINA_OUT:-$LUMINA_WORK/train_out}"
export LUMINA_OUT="$OUT_DIR"
LOG_DIR="$LUMINA_WORK/logs"
RESUME="${LUMINA_RESUME:-1}"
TIME_BUDGET="${LUMINA_TIME_BUDGET:-0}"
ALLOW_GPU="${LUMINA_ALLOW_GPU:-0}"
CTX="${LUMINA_CTX:-32768}"
BATCH="${LUMINA_BATCH:-2048}"
UBATCH="${LUMINA_UBATCH:-128}"
LR="${LUMINA_LR:-1e-4}"
EPOCHS="${LUMINA_EPOCHS:-1}"
RANK="${LUMINA_RANK:-64}"
ALPHA="${LUMINA_ALPHA:-128}"
SAVE_STEPS="${LUMINA_SAVE_STEPS:-50}"
SAVE_EPOCHS="${LUMINA_SAVE_EPOCHS:-1}"
KEEP_LAST="${LUMINA_KEEP_LAST:-3}"
EXTRA_ARGS="${LUMINA_EXTRA_ARGS:-}"

# output stem must match the checkpoint stem: -o <stem>.gguf <=> <stem>.ckpt-<step>.gguf
OUT_STEM="stage1.lora"
OUT_GGUF="$OUT_DIR/$OUT_STEM.gguf"

mkdir -p "$LUMINA_WORK" "$OUT_DIR" "$LOG_DIR"

# python3 is the interpreter on Kaggle; allow an override for local testing
PY="${LUMINA_PYTHON:-python3}"
if ! "$PY" -c 'import sys' >/dev/null 2>&1; then
    PY=python
fi

# ------------------------------------------------------------------- inputs
c_info "resolving inputs (base model / dataset / checkpoint)"
eval "$("$PY" "$SCRIPT_DIR/kaggle_paths.py" --shell)"
[[ -n "$MODEL" ]] || die "base model not found (set LUMINA_MODEL)"
[[ -n "$TRAIN" ]] || die "training dataset not found (set LUMINA_TRAIN)"

# ------------------------------------------------------------------- binary
if [[ -z "${LUMINA_BIN:-}" ]]; then
    for cand in "$LUMINA_WORK/bin/Lumina-Engine-train" "$SCRIPT_DIR/bin/Lumina-Engine-train"; do
        [[ -x "$cand" ]] && { LUMINA_BIN="$cand"; break; }
    done
fi
[[ -n "${LUMINA_BIN:-}" && -x "$LUMINA_BIN" ]] || die "trainer binary not found - run build_cuda.sh first (or set LUMINA_BIN)"
c_info "trainer: $LUMINA_BIN"
if ! "$LUMINA_BIN" --version 2>&1 | sed -n '1p;/^build /p' | sed 's/^/         /'; then
    c_warn "the trainer did not start - rebuild it with build_cuda.sh"
    exit 1
fi

# -------------------------------------------------------------- checkpoints
# --resume picks the highest step of <out-stem>.ckpt-<step>.gguf found in the
# output directory. Copy the newest KEEP_LAST checkpoints (+ their .json) there.
if [[ "$RESUME" == "1" && -n "${CKPT_GGUF:-}" && -n "${CKPT_DIR:-}" ]]; then
    c_info "copying up to $KEEP_LAST checkpoint(s) from $CKPT_DIR"
    mapfile -t ck_list < <(find "$CKPT_DIR" -maxdepth 1 -type f -name "$CKPT_PREFIX.ckpt-*.gguf" 2>/dev/null | sort -V | tail -n "$KEEP_LAST")
    if [[ ${#ck_list[@]} -eq 0 ]]; then
        c_warn "no checkpoint files matched $CKPT_DIR/$CKPT_PREFIX.ckpt-*.gguf - starting from scratch"
    fi
    for f in "${ck_list[@]}"; do
        cp -f "$f" "$OUT_DIR/" 2>/dev/null || die "could not copy $f (is the output dir writable?)"
        [[ -f "${f%.gguf}.json" ]] && cp -f "${f%.gguf}.json" "$OUT_DIR/" || true
        printf '         %s\n' "$(basename "$f")"
    done

    # --resume only matches <out-stem>.ckpt-<step>.gguf, so rename checkpoints whose
    # stem differs (e.g. a bundle produced with another --out-file name).
    shopt -s nullglob
    for f in "$OUT_DIR"/*.ckpt-*.gguf; do
        base="$(basename "$f")"
        [[ "$base" == "$OUT_STEM.ckpt-"* ]] && continue
        step_and_ext="${base##*.ckpt-}"
        newbase="$OUT_STEM.ckpt-$step_and_ext"
        mv -f "$f" "$OUT_DIR/$newbase"
        if [[ -f "${f%.gguf}.json" ]]; then
            mv -f "${f%.gguf}.json" "$OUT_DIR/${newbase%.gguf}.json"
        fi
        c_warn "renamed $(basename "$f") -> $newbase (to match the output stem)"
    done
    shopt -u nullglob
fi

# A checkpoint interrupted mid-write leaves a .gguf without its .json (the trainer
# writes the adapter first and the metadata second). Loading such a file makes
# --resume fall back to a fresh run, so drop obviously truncated files and keep the
# previous checkpoint. A complete .gguf without .json is kept (resume then assumes
# epoch 0, which is what we want for stage 1).
verify_checkpoints() {
    shopt -s nullglob
    local f base max_size=0
    local list=()
    for f in "$OUT_DIR"/*.ckpt-*.gguf; do
        list+=("$f")
    done
    for f in "${list[@]}"; do
        [[ "$(stat -c %s "$f" 2>/dev/null || echo 0)" -gt "$max_size" ]] && max_size="$(stat -c %s "$f")"
    done
    for f in "${list[@]}"; do
        base="${f%.gguf}"
        local sz
        sz="$(stat -c %s "$f" 2>/dev/null || echo 0)"
        if [[ "$sz" -lt 1048576 ]]; then
            c_warn "removing suspiciously small checkpoint $(basename "$f")"
            rm -f "$f" "$base.json"
        elif [[ ! -f "$base.json" ]]; then
            if [[ "$max_size" -gt 1048576 && "$sz" -lt $((max_size * 99 / 100)) ]]; then
                c_warn "removing truncated checkpoint $(basename "$f") (no .json, $sz < $max_size bytes)"
                rm -f "$f"
            else
                c_warn "$(basename "$f") has no .json - resume will assume epoch 0"
            fi
        fi
    done
    shopt -u nullglob
    return 0
}
verify_checkpoints

# ------------------------------------------------------------------ command
CMD=("$LUMINA_BIN"
     -m "$MODEL"
     --dataset "$TRAIN")
[[ -n "${VAL:-}" ]] && CMD+=(--dataset-val "$VAL")
CMD+=(-c "$CTX" -b "$BATCH" -ub "$UBATCH"
      --cpu-cores 0
      -lr "$LR" --epochs "$EPOCHS"
      -lora-rank "$RANK" -lora-alpha "$ALPHA"
      --save-steps "$SAVE_STEPS" --save-every-epochs "$SAVE_EPOCHS" --keep-last "$KEEP_LAST"
      -o "$OUT_GGUF")
[[ "$RESUME" == "1" ]] && CMD+=(--resume)
if [[ "$ALLOW_GPU" == "1" ]]; then
    CMD+=(-ngl 99)
fi
# shellcheck disable=SC2206
[[ -n "$EXTRA_ARGS" ]] && CMD+=($EXTRA_ARGS)

RUN_ENV=()
if [[ "$ALLOW_GPU" == "1" ]]; then
    c_warn "LUMINA_ALLOW_GPU=1: the backward pass and the optimizer are CPU-only kernels;"
    c_warn "               the graph is only partially offloaded. Verify loss/acc before trusting it."
    RUN_ENV+=(LUMINA_TRAIN_ALLOW_GPU=1)
fi

c_info "command"
printf '         %s\n' "${CMD[*]}"
c_info "notes"
printf '         output dir : %s\n' "$OUT_DIR"
printf '         resume     : %s (checkpoint step %s)\n' "$RESUME" "${CKPT_STEP:--1}"
printf '         threads    : --cpu-cores 0 = %s logical cores\n' "$(nproc 2>/dev/null || echo '?')"
printf '         ctx/batch  : %s / %s / ubatch %s\n' "$CTX" "$BATCH" "$UBATCH"
printf '         stride     : default = n_ctx/2 = %s tokens (matches the local run)\n' "$((CTX / 2))"
if [[ "$TIME_BUDGET" != "0" ]]; then
    printf '         time budget: %s s (%.2f h)\n' "$TIME_BUDGET" "$("$PY" -c "print($TIME_BUDGET/3600)")"
fi

if [[ "$DRY_RUN" == "1" ]]; then
    c_ok "dry run - nothing executed"
    exit 0
fi

# --------------------------------------------------------------------- run
STAMP="$(date -u +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/train-$STAMP.log"
c_info "logging to $LOG"

set +e
if [[ "$TIME_BUDGET" != "0" ]] && command -v timeout >/dev/null 2>&1; then
    env "${RUN_ENV[@]}" timeout --signal=TERM --kill-after=120 "$TIME_BUDGET" \
        "${CMD[@]}" > >(tee -a "$LOG") 2>&1
    RC=$?
else
    env "${RUN_ENV[@]}" "${CMD[@]}" > >(tee -a "$LOG") 2>&1
    RC=$?
fi
set -e

echo
if [[ "$RC" == "124" || "$RC" == "137" ]]; then
    c_warn "time budget reached - the trainer was stopped."
    c_warn "progress since the last checkpoint (<= $SAVE_STEPS micro-batches) is lost."
elif [[ "$RC" != "0" ]]; then
    c_warn "trainer exited with code $RC"
else
    c_ok "trainer finished"
fi

verify_checkpoints

if grep -q "failed to load checkpoint adapter" "$LOG" 2>/dev/null; then
    c_warn "the trainer could not load the newest checkpoint and restarted from scratch!"
    c_warn "delete the bad stage1.lora.ckpt-*.gguf / .json pair and run again to fall back to the previous one."
fi

c_info "last log lines"
tail -c 4000 "$LOG" | tr '\r' '\n' | grep -v '^$' | tail -n 25 | sed 's/^/         /'

c_info "checkpoints in $OUT_DIR"
ls -lh "$OUT_DIR" 2>/dev/null | sed 's/^/         /'

exit "$RC"
