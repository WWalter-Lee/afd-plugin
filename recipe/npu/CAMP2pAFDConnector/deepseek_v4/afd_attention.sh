#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "${ROOT_DIR}/recipe/npu/deepseek_v4/common/activate_role_runtime.sh"
set -u

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8910}"
AFD_HOST="${AFD_HOST:-127.0.0.1}"
AFD_PORT="${AFD_PORT:-29761}"
ATTENTION_RANKS="${ATTENTION_RANKS:-8}"
FFN_RANKS="${FFN_RANKS:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_BATCHED_TOKENS="${ATTENTION_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS:-1024}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
EXECUTION_MODE="${EXECUTION_MODE:-eager}"
U_BATCHES="${U_BATCHES:-1}"
DBO_DECODE_TOKEN_THRESHOLD="${DBO_DECODE_TOKEN_THRESHOLD:-2}"
DBO_PREFILL_TOKEN_THRESHOLD="${DBO_PREFILL_TOKEN_THRESHOLD:-12}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-8}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-1 2 4 8}"
AFD_ASYNC_SCHEDULING="${AFD_ASYNC_SCHEDULING:-auto}"

export ASCEND_RT_VISIBLE_DEVICES="${ATTENTION_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export HCCL_IF_IP="${HCCL_IF_IP:-192.169.91.106}"
export HCCL_IF_BASE_PORT="${ATTENTION_HCCL_IF_BASE_PORT:-51000}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"

if [[ "$TENSOR_PARALLEL_SIZE" != "1" ]]; then
  echo "DeepSeek-V4 CAMP2p recipe supports only TENSOR_PARALLEL_SIZE=1" >&2
  exit 2
fi
if [[ "$ATTENTION_RANKS" != "$FFN_RANKS" ]]; then
  echo "DeepSeek-V4 CAMP2p requires equal Attention and FFN ranks" >&2
  exit 2
fi
source "${ROOT_DIR}/afd_plugin/_cann_ops_custom/vendors/afd-plugin/bin/set_env.bash"

ADDITIONAL_CONFIG="$(printf '{"afd":{"role":"attention","connector":"CAMP2pAFDConnector","host":"%s","port":%s,"num_attention_ranks":%s,"num_ffn_ranks":%s}}' "$AFD_HOST" "$AFD_PORT" "$ATTENTION_RANKS" "$FFN_RANKS")"

case "$EXECUTION_MODE" in
  eager)
    EXECUTION_ARGS=(--enforce-eager)
    ;;
  full-decode-only)
    read -r -a CAPTURE_SIZE_ARGS <<<"$CUDAGRAPH_CAPTURE_SIZES"
    EXECUTION_ARGS=(
      --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
      --cudagraph-capture-sizes "${CAPTURE_SIZE_ARGS[@]}"
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    )
    ;;
  *)
    echo "Unsupported EXECUTION_MODE=$EXECUTION_MODE" >&2
    exit 2
    ;;
esac

case "$U_BATCHES" in
  1)
    UBATCH_ARGS=()
    ;;
  2)
    if [[ "$EXECUTION_MODE" != "eager" ]]; then
      echo "DeepSeek-V4 CAMP2p U2 supports only eager execution" >&2
      exit 2
    fi
    UBATCH_ARGS=(
      --enable-dbo
      --dbo-decode-token-threshold "$DBO_DECODE_TOKEN_THRESHOLD"
      --dbo-prefill-token-threshold "$DBO_PREFILL_TOKEN_THRESHOLD"
    )
    ;;
  *)
    echo "DeepSeek-V4 CAMP2p supports U_BATCHES=1 or 2, got $U_BATCHES" >&2
    exit 2
    ;;
esac

case "$AFD_ASYNC_SCHEDULING" in
  auto)
    SCHEDULING_ARGS=()
    ;;
  on)
    SCHEDULING_ARGS=(--async-scheduling)
    ;;
  off)
    SCHEDULING_ARGS=(--no-async-scheduling)
    ;;
  *)
    echo "AFD_ASYNC_SCHEDULING must be auto, on, or off" >&2
    exit 2
    ;;
esac

exec vllm serve "$MODEL_PATH" \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --api-server-count 1 \
  --served-model-name dsv4-afd \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --data-parallel-size "$ATTENTION_RANKS" \
  --tensor-parallel-size 1 \
  --all2all-backend flashinfer_all2allv \
  --enable-expert-parallel \
  --seed 1024 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tokenizer-mode deepseek_v4 \
  --no-enable-prefix-caching \
  --safetensors-load-strategy lazy \
  --quantization ascend \
  --block-size 128 \
  --additional-config "$ADDITIONAL_CONFIG" \
  "${SCHEDULING_ARGS[@]}" \
  "${UBATCH_ARGS[@]}" \
  "${EXECUTION_ARGS[@]}"
