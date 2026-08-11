#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "${ROOT_DIR}/tools/dsv4/activate_runtime.sh"
source /mnt/workspace/code/vllm-ascend-afd-80d8c194f/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
source "${ROOT_DIR}/afd_plugin/_cann_ops_custom/vendors/afd-plugin/bin/set_env.bash"
set -u

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8911}"
AFD_HOST="${AFD_HOST:-127.0.0.1}"
AFD_PORT="${AFD_PORT:-29761}"
ATTENTION_RANKS="${ATTENTION_RANKS:-8}"
FFN_RANKS="${FFN_RANKS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
EXECUTION_MODE="${EXECUTION_MODE:-eager}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-8}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-1 2 4 8}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
export HCCL_IF_IP="${HCCL_IF_IP:-192.169.91.106}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2048}"
export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1
export SOC_VERSION=ascend910_9362
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-18000}"
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd
unset VLLM_ASCEND_ENABLE_FLASHCOMM1

ADDITIONAL_CONFIG="$(printf '{"afd":{"role":"ffn","connector":"CAMP2pAFDConnector","host":"%s","port":%s,"num_attention_ranks":%s,"num_ffn_ranks":%s}}' "$AFD_HOST" "$AFD_PORT" "$ATTENTION_RANKS" "$FFN_RANKS")"

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

shutdown_requested=0
vllm_pid=""

forward_shutdown() {
  shutdown_requested=1
  if [[ -n "$vllm_pid" ]]; then
    kill -TERM "$vllm_pid" 2>/dev/null || true
  fi
}

trap forward_shutdown TERM INT
set +e
vllm serve "$MODEL_PATH" \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --api-server-count 1 \
  --served-model-name dsv4-afd-ffn \
  --max-model-len 4096 \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --data-parallel-size "$FFN_RANKS" \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --seed 1024 \
  --gpu-memory-utilization 0.90 \
  --tokenizer-mode deepseek_v4 \
  --no-enable-prefix-caching \
  --safetensors-load-strategy lazy \
  --quantization ascend \
  --block-size 128 \
  --additional-config "$ADDITIONAL_CONFIG" \
  "${EXECUTION_ARGS[@]}" &
vllm_pid=$!
wait "$vllm_pid"
vllm_status=$?
while kill -0 "$vllm_pid" 2>/dev/null; do
  wait "$vllm_pid"
  vllm_status=$?
done
set -e

if ((shutdown_requested)); then
  exit 0
fi
exit "$vllm_status"
