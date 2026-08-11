#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "${ROOT_DIR}/tools/dsv4/activate_runtime.sh"
source /mnt/workspace/code/vllm-ascend-afd-80d8c194f/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
source "${ROOT_DIR}/afd_plugin/_cann_ops_custom/vendors/afd-plugin/bin/set_env.bash"
set -u

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8910}"
AFD_HOST="${AFD_HOST:-127.0.0.1}"
AFD_PORT="${AFD_PORT:-29761}"
ATTENTION_RANKS="${ATTENTION_RANKS:-8}"
FFN_RANKS="${FFN_RANKS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_IF_IP="${HCCL_IF_IP:-192.169.91.106}"
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

ADDITIONAL_CONFIG="$(printf '{"afd":{"role":"attention","connector":"CAMP2pAFDConnector","host":"%s","port":%s,"num_attention_ranks":%s,"num_ffn_ranks":%s}}' "$AFD_HOST" "$AFD_PORT" "$ATTENTION_RANKS" "$FFN_RANKS")"

exec vllm serve "$MODEL_PATH" \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --api-server-count 1 \
  --served-model-name dsv4-afd \
  --max-model-len 4096 \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --data-parallel-size "$ATTENTION_RANKS" \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --seed 1024 \
  --gpu-memory-utilization 0.90 \
  --tokenizer-mode deepseek_v4 \
  --no-enable-prefix-caching \
  --safetensors-load-strategy lazy \
  --quantization ascend \
  --block-size 128 \
  --additional-config "$ADDITIONAL_CONFIG"
