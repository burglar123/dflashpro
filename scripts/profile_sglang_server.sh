#!/usr/bin/env bash
set -euo pipefail

# SGLang profiling workflow with selectable stages:
#  - PROFILE_STAGE=nsys : timeline only (fastest)
#  - PROFILE_STAGE=ncu  : kernel deep-dive only (slow)
#  - PROFILE_STAGE=both : run nsys first, then ncu
#
# Usage:
#   TARGET_MODEL=Qwen/Qwen3-8B \
#   DRAFT_MODEL=z-lab/Qwen3-8B-DFlash-b16 \
#   SERVER_MODE=dflash \
#   PROFILE_STAGE=nsys \
#   ./scripts/profile_sglang_server.sh

TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-8B}
DRAFT_MODEL=${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}
DATASET_NAME=${DATASET_NAME:-math500}
SERVER_MODE=${SERVER_MODE:-dflash} # baseline|dflash|eagle
EAGLE_DRAFT_MODEL=${EAGLE_DRAFT_MODEL:-}
EAGLE_ALGORITHM=${EAGLE_ALGORITHM:-EAGLE3}
EAGLE_NUM_STEPS=${EAGLE_NUM_STEPS:-5}
EAGLE_NUM_DRAFT_TOKENS=${EAGLE_NUM_DRAFT_TOKENS:-5}
EAGLE_TOPK=${EAGLE_TOPK:-1}

SERVER_PORT=${SERVER_PORT:-31000}
SERVER_URL="http://127.0.0.1:${SERVER_PORT}"
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flashinfer}
DTYPE=${DTYPE:-bfloat16}
TP_SIZE=${TP_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-64}
DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-0}

CONCURRENCIES=${CONCURRENCIES:-1,8,32}
QUESTIONS_PER_CONCURRENCY_BASE=${QUESTIONS_PER_CONCURRENCY_BASE:-64}
MAX_QUESTIONS_PER_CONFIG=${MAX_QUESTIONS_PER_CONFIG:-512}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
TIMEOUT_S=${TIMEOUT_S:-3600}
BATCH_REQUESTS=${BATCH_REQUESTS:-0}
EXPECT_SPECULATIVE=${EXPECT_SPECULATIVE:-1}

PROFILE_STAGE=${PROFILE_STAGE:-both} # nsys|ncu|both

OUT_DIR=${OUT_DIR:-profiles/sglang}
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_DIR"

NSYS_OUT="$OUT_DIR/sglang_${SERVER_MODE}_${TS}"
NCU_OUT="$OUT_DIR/sglang_${SERVER_MODE}_${TS}"
BENCH_MD="$OUT_DIR/sglang_${SERVER_MODE}_${TS}.md"
SERVER_LOG="$OUT_DIR/sglang_${SERVER_MODE}_${TS}.server.log"
NCU_LOG="$OUT_DIR/sglang_${SERVER_MODE}_${TS}.ncu.log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[cleanup] stopping server pid=$SERVER_PID"
    kill -INT "$SERVER_PID" || true
    wait "$SERVER_PID" || true
  fi
  if [[ -n "${NCU_SERVER_PID:-}" ]] && kill -0 "$NCU_SERVER_PID" 2>/dev/null; then
    echo "[cleanup] stopping ncu server pid=$NCU_SERVER_PID"
    kill -INT "$NCU_SERVER_PID" || true
    wait "$NCU_SERVER_PID" || true
  fi
}
trap cleanup EXIT

SERVER_ARGS=(
  python -m sglang.launch_server
  --model-path "$TARGET_MODEL"
  --host 127.0.0.1
  --port "$SERVER_PORT"
  --trust-remote-code
  --attention-backend "$ATTENTION_BACKEND"
  --tp-size "$TP_SIZE"
  --dtype "$DTYPE"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
)

if [[ "$DISABLE_CUDA_GRAPH" == "1" ]]; then
  SERVER_ARGS+=(--disable-cuda-graph)
fi

case "$SERVER_MODE" in
  baseline)
    EXPECT_SPECULATIVE=0
    ;;
  dflash)
    SERVER_ARGS+=(
      --speculative-algorithm DFLASH
      --speculative-draft-model-path "$DRAFT_MODEL"
    )
    ;;
  eagle)
    if [[ -z "$EAGLE_DRAFT_MODEL" ]]; then
      echo "[error] SERVER_MODE=eagle requires EAGLE_DRAFT_MODEL"
      exit 1
    fi
    SERVER_ARGS+=(
      --speculative-algorithm "$EAGLE_ALGORITHM"
      --speculative-draft-model-path "$EAGLE_DRAFT_MODEL"
      --speculative-num-steps "$EAGLE_NUM_STEPS"
      --speculative-num-draft-tokens "$EAGLE_NUM_DRAFT_TOKENS"
      --speculative-eagle-topk "$EAGLE_TOPK"
    )
    ;;
  *)
    echo "[error] Unsupported SERVER_MODE=$SERVER_MODE"
    exit 1
    ;;
esac

case "$PROFILE_STAGE" in
  nsys|ncu|both)
    ;;
  *)
    echo "[error] Unsupported PROFILE_STAGE=$PROFILE_STAGE. Use: nsys|ncu|both"
    exit 1
    ;;
esac

BENCH_ARGS=(
  python benchmark_sglang.py
  --dataset-name "$DATASET_NAME"
  --target-model "$TARGET_MODEL"
  --concurrencies "$CONCURRENCIES"
  --questions-per-concurrency-base "$QUESTIONS_PER_CONCURRENCY_BASE"
  --max-questions-per-config "$MAX_QUESTIONS_PER_CONFIG"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --timeout-s "$TIMEOUT_S"
  --server-url "$SERVER_URL"
  --server-label "$SERVER_MODE"
  --output-md "$BENCH_MD"
)

if [[ "$EXPECT_SPECULATIVE" == "1" ]]; then
  BENCH_ARGS+=(--server-expect-speculative)
fi
if [[ "$BATCH_REQUESTS" == "1" ]]; then
  BENCH_ARGS+=(--batch-requests)
fi

wait_server_ready() {
  local name="$1"
  for _ in $(seq 1 120); do
    if curl -fsS "$SERVER_URL/get_model_info" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "[error] ${name} server failed to come up"
  return 1
}

run_nsys_stage() {
  echo "[nsys] Launching server under nsys..."
  nsys profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --sample=none \
    --output "$NSYS_OUT" \
    --force-overwrite true \
    --capture-range=none \
    "${SERVER_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  echo "[nsys] Waiting for server health..."
  wait_server_ready "nsys" || {
    echo "[error] see log: $SERVER_LOG"
    exit 1
  }

  echo "[nsys] Running load generator..."
  "${BENCH_ARGS[@]}"

  echo "[nsys] Stopping server gracefully for nsys flush..."
  kill -INT "$SERVER_PID" || true
  wait "$SERVER_PID" || true
  unset SERVER_PID

  echo "[nsys] report: ${NSYS_OUT}.nsys-rep"
}

run_ncu_stage() {
  echo "[ncu] Launching server under ncu (this may take a while)..."
  # Keep kernel filters broad and SGLang-compatible (do not depend on old Python NVTX names).
  ncu \
    --set full \
    --target-processes all \
    --kernel-name-base demangled \
    --kernel-name "regex:.*(attention|flash|gemm|decode|fused).*" \
    --launch-skip 20 \
    --launch-count 60 \
    --force-overwrite \
    --export "$NCU_OUT" \
    "${SERVER_ARGS[@]}" >"$NCU_LOG" 2>&1 &
  NCU_SERVER_PID=$!

  echo "[ncu] Waiting for server health..."
  wait_server_ready "ncu" || {
    echo "[error] see log: $NCU_LOG"
    exit 1
  }

  echo "[ncu] Running load generator..."
  "${BENCH_ARGS[@]}" >/dev/null

  echo "[ncu] Stopping server..."
  kill -INT "$NCU_SERVER_PID" || true
  wait "$NCU_SERVER_PID" || true
  unset NCU_SERVER_PID

  echo "[ncu] report: ${NCU_OUT}.ncu-rep"
}

if [[ "$PROFILE_STAGE" == "nsys" || "$PROFILE_STAGE" == "both" ]]; then
  run_nsys_stage
fi

if [[ "$PROFILE_STAGE" == "ncu" || "$PROFILE_STAGE" == "both" ]]; then
  run_ncu_stage
fi

echo "Done."
echo "Benchmark md: ${BENCH_MD}"
echo "Server log  : ${SERVER_LOG}"
if [[ "$PROFILE_STAGE" == "ncu" || "$PROFILE_STAGE" == "both" ]]; then
  echo "NCU log     : ${NCU_LOG}"
fi
