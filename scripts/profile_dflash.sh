#!/usr/bin/env bash
set -euo pipefail

# Usage:
# MODEL=... DRAFT=... DATASET=... ./scripts/profile_dflash.sh
# Optional env:
#   MAX_SAMPLES=20 BATCH_SIZE=1 MAX_NEW_TOKENS=512 BLOCK_SIZE=4 OUT_DIR=profiles

MODEL=${MODEL:?please set MODEL}
DRAFT=${DRAFT:?please set DRAFT}
DATASET=${DATASET:?please set DATASET}

MAX_SAMPLES=${MAX_SAMPLES:-20}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
BLOCK_SIZE=${BLOCK_SIZE:-}
OUT_DIR=${OUT_DIR:-profiles}

mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
NSYS_OUT="$OUT_DIR/dflash_nsys_${TS}"
NCU_OUT="$OUT_DIR/dflash_ncu_${TS}"

COMMON_ARGS=(
  --model-name-or-path "$MODEL"
  --draft-name-or-path "$DRAFT"
  --dataset "$DATASET"
  --max-samples "$MAX_SAMPLES"
  --batch-size "$BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
)

if [[ -n "$BLOCK_SIZE" ]]; then
  COMMON_ARGS+=(--block-size "$BLOCK_SIZE")
fi

echo "[1/3] Running nsys..."
nsys profile \
  -o "$NSYS_OUT" \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  python benchmark.py "${COMMON_ARGS[@]}"

echo "[2/3] Exporting nsys stats..."
nsys stats --report nvtx_kern_sum,cuda_gpu_kern_sum "$NSYS_OUT.nsys-rep" | tee "$OUT_DIR/dflash_nsys_stats_${TS}.txt"

echo "[3/3] Running ncu (kernel breakdown under NVTX ranges)..."
ncu \
  --target-processes all \
  --set full \
  --nvtx \
  --nvtx-include "draft.qkv/" \
  --nvtx-include "draft.attn/" \
  --nvtx-include "draft.ffn/" \
  --nvtx-include "target.verify.qkv/" \
  --nvtx-include "target.verify.attn/" \
  --nvtx-include "target.verify.ffn/" \
  --nvtx-include "kv_update.draft_cache/" \
  --nvtx-include "kv_update.target_cache/" \
  -o "$NCU_OUT" \
  python benchmark.py "${COMMON_ARGS[@]}"

echo "Done."
echo "NSYS report : $NSYS_OUT.nsys-rep"
echo "NSYS stats  : $OUT_DIR/dflash_nsys_stats_${TS}.txt"
echo "NCU report  : $NCU_OUT.ncu-rep"
