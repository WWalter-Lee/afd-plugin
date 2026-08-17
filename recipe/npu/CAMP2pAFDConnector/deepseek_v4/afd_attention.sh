#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export DSV4_VLLM_VENV="${DSV4_RUNTIME_VENV:-/mnt/workspace/code/.venvs/afd-v026}"
source "${ROOT_DIR}/tools/dsv4/activate_runtime.sh"
DSV4_VLLM_ASCEND_ROOT="${DSV4_VLLM_ASCEND_ROOT:-/mnt/workspace/code/vllm-ascend-afd-80d8c194f}"
source "${DSV4_VLLM_ASCEND_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
set -u

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8910}"
AFD_HOST="${AFD_HOST:-127.0.0.1}"
AFD_PORT="${AFD_PORT:-29761}"
AFD_CONNECTOR="${AFD_CONNECTOR:-CAMP2pAFDConnector}"
ATTENTION_RANKS="${ATTENTION_RANKS:-8}"
FFN_RANKS="${FFN_RANKS:-8}"
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

export ASCEND_RT_VISIBLE_DEVICES="${ATTENTION_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export HCCL_IF_IP="${HCCL_IF_IP:-192.169.91.106}"
export HCCL_IF_BASE_PORT="${ATTENTION_HCCL_IF_BASE_PORT:-51000}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1
export SOC_VERSION=ascend910_9362
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-18000}"
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd
unset VLLM_ASCEND_ENABLE_FLASHCOMM1

case "$AFD_CONNECTOR" in
  CAMP2pAFDConnector)
    source "${ROOT_DIR}/afd_plugin/_cann_ops_custom/vendors/afd-plugin/bin/set_env.bash"
    ;;
  P2pHcclAFDConnector)
    ;;
  *)
    echo "Unsupported DeepSeek-V4 NPU connector: $AFD_CONNECTOR" >&2
    exit 2
    ;;
esac

ADDITIONAL_CONFIG="$(printf '{"afd":{"role":"attention","connector":"%s","host":"%s","port":%s,"num_attention_ranks":%s,"num_ffn_ranks":%s}}' "$AFD_CONNECTOR" "$AFD_HOST" "$AFD_PORT" "$ATTENTION_RANKS" "$FFN_RANKS")"

case "$EXECUTION_MODE" in
  eager)
    EXECUTION_ARGS=(--enforce-eager)
    ;;
  full-decode-only)
    if [[ "$AFD_CONNECTOR" == "P2pHcclAFDConnector" ]]; then
      echo "P2pHcclAFDConnector currently supports only EXECUTION_MODE=eager" >&2
      exit 2
    fi
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
      echo "DeepSeek-V4 U2 currently supports only EXECUTION_MODE=eager" >&2
      exit 2
    fi
    UBATCH_ARGS=(
      --enable-dbo
      --dbo-decode-token-threshold "$DBO_DECODE_TOKEN_THRESHOLD"
      --dbo-prefill-token-threshold "$DBO_PREFILL_TOKEN_THRESHOLD"
    )
    ;;
  *)
    echo "DeepSeek-V4 AFD supports U_BATCHES=1 or 2, got $U_BATCHES" >&2
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
  --enable-expert-parallel \
  --seed 1024 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tokenizer-mode deepseek_v4 \
  --no-enable-prefix-caching \
  --safetensors-load-strategy lazy \
  --quantization ascend \
  --block-size 128 \
  --additional-config "$ADDITIONAL_CONFIG" \
  "${UBATCH_ARGS[@]}" \
  "${EXECUTION_ARGS[@]}"
