#!/usr/bin/env bash
set -euo pipefail

# Two-stage SGLang profiling workflow:
#  1) nsys captures server timeline while benchmark_sglang.py generates load.
#  2) ncu deep-dives a short run with kernel-name filters (no legacy Python NVTX filters).
#
# Usage:
#   TARGET_MODEL=Qwen/Qwen3-8B \
#   DRAFT_MODEL=z-lab/Qwen3-8B-DFlash-b16 \
#   SERVER_MODE=dflash \
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

CONCURRENCIES=${CONCURRENCIES:-1,8,32}
QUESTIONS_PER_CONCURRENCY_BASE=${QUESTIONS_PER_CONCURRENCY_BASE:-64}
MAX_QUESTIONS_PER_CONFIG=${MAX_QUESTIONS_PER_CONFIG:-512}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
TIMEOUT_S=${TIMEOUT_S:-3600}
BATCH_REQUESTS=${BATCH_REQUESTS:-0}
EXPECT_SPECULATIVE=${EXPECT_SPECULATIVE:-1}

OUT_DIR=${OUT_DIR:-profiles/sglang}
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_DIR"

NSYS_OUT="$OUT_DIR/sglang_${SERVER_MODE}_${TS}"
NCU_OUT="$OUT_DIR/sglang_${SERVER_MODE}_${TS}"
BENCH_MD="$OUT_DIR/sglang_${SERVER_MODE}_${TS}.md"
SERVER_LOG="$OUT_DIR/sglang_${SERVER_MODE}_${TS}.server.log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[cleanup] stopping server pid=$SERVER_PID"
    kill -INT "$SERVER_PID" || true
    wait "$SERVER_PID" || true
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

echo "[1/5] Launching server under nsys..."
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=none \
  --output "$NSYS_OUT" \
  --force-overwrite true \
  --capture-range=none \
  "${SERVER_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "[2/5] Waiting for server health..."
for _ in $(seq 1 120); do
  if curl -fsS "$SERVER_URL/get_model_info" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl -fsS "$SERVER_URL/get_model_info" >/dev/null 2>&1; then
  echo "[error] server failed to come up; see $SERVER_LOG"
  exit 1
fi

echo "[3/5] Running load generator (benchmark_sglang client mode)..."
"${BENCH_ARGS[@]}"

echo "[4/5] Stopping server gracefully for nsys flush..."
kill -INT "$SERVER_PID" || true
wait "$SERVER_PID" || true
unset SERVER_PID

echo "[5/5] Running short ncu deep-dive (kernel-name filters)..."
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
  "${SERVER_ARGS[@]}" >"$OUT_DIR/sglang_${SERVER_MODE}_${TS}.ncu.log" 2>&1 &
NCU_SERVER_PID=$!

for _ in $(seq 1 120); do
  if curl -fsS "$SERVER_URL/get_model_info" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl -fsS "$SERVER_URL/get_model_info" >/dev/null 2>&1; then
  echo "[error] ncu server failed to come up"
  kill -INT "$NCU_SERVER_PID" || true
  wait "$NCU_SERVER_PID" || true
  exit 1
fi

"${BENCH_ARGS[@]}" >/dev/null
kill -INT "$NCU_SERVER_PID" || true
wait "$NCU_SERVER_PID" || true

echo "Done."
echo "NSYS report : ${NSYS_OUT}.nsys-rep"
echo "NCU report  : ${NCU_OUT}.ncu-rep"
echo "Benchmark md: ${BENCH_MD}"
echo "Server log  : ${SERVER_LOG}"
