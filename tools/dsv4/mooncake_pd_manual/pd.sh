#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-help}"
CONFIG_FILE="${2:-${PD_CONFIG_FILE:-${SCRIPT_DIR}/config.env}}"

usage() {
  cat <<'EOF'
Usage: bash pd.sh <action> [config.env]

Actions:
  init          Copy config.env.example to the requested config path.
  print-config  Print the non-secret effective role/topology configuration.
  install       Install the M9 afd-plugin editable source and Mooncake wheel.
  check         Validate versions, runtime, network, NPUs, and local round-trip.
  start         Start the configured prefill, decode, or proxy role.
  status        Check owned processes, readiness, and fatal log markers.
  validate      Run proxy smoke, 30-request golden, batch, and cancellation gates.
  stop          Stop only process groups owned by this config.
  collect       Produce a redacted, size-capped support artifact.
EOF
}

log() {
  printf '[mooncake-pd:%s] %s\n' "${NODE_ROLE:-unconfigured}" "$*"
}

warn() {
  printf '[mooncake-pd:%s] WARNING: %s\n' "${NODE_ROLE:-unconfigured}" "$*" >&2
}

die() {
  printf '[mooncake-pd:%s] ERROR: %s\n' "${NODE_ROLE:-unconfigured}" "$*" >&2
  exit 2
}

if [[ "${ACTION}" == "help" || "${ACTION}" == "-h" || "${ACTION}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${ACTION}" == "init" ]]; then
  [[ ! -e "${CONFIG_FILE}" ]] || die "Config already exists: ${CONFIG_FILE}"
  mkdir -p "$(dirname "${CONFIG_FILE}")"
  cp "${SCRIPT_DIR}/config.env.example" "${CONFIG_FILE}"
  printf 'Created %s\n' "${CONFIG_FILE}"
  exit 0
fi

[[ -f "${CONFIG_FILE}" ]] || die "Missing config: ${CONFIG_FILE}; run: bash $0 init ${CONFIG_FILE}"
set -a
# shellcheck disable=SC1090
source "${CONFIG_FILE}"
set +a

: "${CODE_ROOT:=/mnt/workspace/code}"
: "${VENV_ROOT:=${CODE_ROOT}/.venvs/afd-v023-vllm-cann}"
: "${VLLM_ROOT:=${CODE_ROOT}/vllm-release-v0.23.0}"
: "${VLLM_ASCEND_ROOT:=${CODE_ROOT}/vllm-ascend-rfc-vllm-cann}"
: "${AFD_PLUGIN_ROOT:=${CODE_ROOT}/afd-plugin}"
: "${CANN_ROOT:=${CODE_ROOT}/.ascend/cann-9.0.1/cann-9.0.1}"
: "${MODEL_PATH:=/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
: "${RUN_ROOT:=/mnt/workspace/dsv4-afd-mooncake-pd}"
: "${STATE_ROOT:=${RUN_ROOT}/state/${NODE_ROLE}}"
: "${LOG_ROOT:=${RUN_ROOT}/logs/${NODE_ROLE}}"
: "${VALIDATION_ROOT:=${RUN_ROOT}/validation}"
: "${OUTPUT_ROOT:=${RUN_ROOT}/output}"
: "${VLLM_COMMIT:=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665}"
: "${VLLM_ASCEND_COMMIT:=3da28f9414583d2d0b672a8f06d1fae142404bda}"
: "${MOONCAKE_WHEEL_SHA256:=0f9964801b24fd683d6016e1196cc0606fc87b0285b45d89c433650b9477ca12}"
: "${PREFILL_API_PORT:=8100}"
: "${DECODE_API_PORT:=8910}"
: "${FFN_PROCESS_PORT:=8911}"
: "${PROXY_PORT:=9000}"
: "${AFD_PORT:=29761}"
: "${PREFILL_KV_PORT:=30000}"
: "${DECODE_KV_PORT:=30100}"
: "${PREFILL_HCCL_IF_BASE_PORT:=50000}"
: "${ATTENTION_HCCL_IF_BASE_PORT:=51000}"
: "${FFN_HCCL_IF_BASE_PORT:=52000}"
: "${MC_MIN_PRC_PORT:=15000}"
: "${MC_MAX_PRC_PORT:=17000}"
: "${PREFILL_DEVICES:=0,1,2,3,4,5,6,7}"
: "${ATTENTION_DEVICES:=0,1,2,3,4,5,6,7}"
: "${FFN_DEVICES:=8,9,10,11,12,13,14,15}"
: "${PREFILL_DP_SIZE:=2}"
: "${PREFILL_TP_SIZE:=4}"
: "${ATTENTION_RANKS:=8}"
: "${FFN_RANKS:=8}"
: "${DECODE_DP_SIZE:=8}"
: "${DECODE_TP_SIZE:=1}"
: "${MAX_MODEL_LEN:=4096}"
: "${MAX_NUM_BATCHED_TOKENS:=4096}"
: "${MAX_NUM_SEQS:=16}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${HCCL_BUFFSIZE:=2048}"
: "${OMP_NUM_THREADS:=10}"
: "${MODEL_NAME:=dsv4-afd}"
: "${INSTALL_SYSTEM_PACKAGES:=0}"
: "${RUN_LOCAL_ROUNDTRIP:=1}"
: "${ROUNDTRIP_PRODUCER_DEVICE:=0}"
: "${ROUNDTRIP_CONSUMER_DEVICE:=1}"
: "${WAIT_READY:=1}"
: "${STARTUP_TIMEOUT_SECONDS:=3600}"
: "${STOP_TIMEOUT_SECONDS:=300}"
: "${FORCE_KILL:=0}"
: "${ALLOW_NPU_PROCESSES:=0}"
: "${VALIDATION_ROUNDS:=3}"
: "${VALIDATION_BATCH_SIZES:=1 8 32}"
: "${RUN_CANCELLATION_TEST:=1}"
: "${ARTIFACT_LOG_TAIL_BYTES:=262144}"
: "${ARTIFACT_MAX_BYTES:=2097152}"

PYTHON_BIN="${VENV_ROOT}/bin/python"
PREFILL_SCRIPT="${AFD_PLUGIN_ROOT}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_pd/prefill.sh"
ATTENTION_SCRIPT="${AFD_PLUGIN_ROOT}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_attention.sh"
FFN_SCRIPT="${AFD_PLUGIN_ROOT}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_ffn.sh"
PROXY_SCRIPT="${AFD_PLUGIN_ROOT}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_pd/proxy.sh"
RUNTIME_CHECK="${AFD_PLUGIN_ROOT}/tools/dsv4/check_mooncake_runtime.sh"
ROUNDTRIP_TOOL="${AFD_PLUGIN_ROOT}/tools/dsv4/check_mooncake_npu_roundtrip.py"
GOLDEN_VALIDATOR="${AFD_PLUGIN_ROOT}/recipe/npu/deepseek_v4/common/validate_golden.py"
FATAL_PATTERN='EngineCore encountered a fatal error|AFD NPU FFN worker loop failed|Mooncake transfer failed|Communication_Error|507015|Traceback'

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

assert_integer() {
  [[ "$2" =~ ^[0-9]+$ ]] || die "$1 must be a non-negative integer: $2"
}

pid_is_alive() {
  [[ "$1" =~ ^[0-9]+$ ]] && kill -0 "$1" 2>/dev/null
}

read_pid() {
  local path="$1"
  local pid
  [[ -f "${path}" ]] || return 1
  read -r pid <"${path}"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${pid}"
}

port_is_listening() {
  local port="$1"
  ss -ltn | awk -v expected=":${port}" '$4 ~ expected "$" {found=1} END {exit !found}'
}

git_head() {
  git -C "$1" rev-parse HEAD
}

validate_role() {
  case "${NODE_ROLE:-}" in
    prefill|decode|proxy) ;;
    *) die "NODE_ROLE must be prefill, decode, or proxy: ${NODE_ROLE:-unset}" ;;
  esac
}

reject_placeholder() {
  local name="$1"
  local value="$2"
  [[ -n "${value}" && "${value}" != *CHANGE_ME* && "${value}" != *REPLACE_WITH* ]] \
    || die "${name} is not configured: ${value}"
}

validate_common_config() {
  validate_role
  reject_placeholder PREFILL_IP "${PREFILL_IP:-}"
  reject_placeholder DECODE_IP "${DECODE_IP:-}"
  [[ "${AFD_PD_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
    || die "AFD_PD_COMMIT must be the delivered 40-character M9 commit"
  assert_integer STARTUP_TIMEOUT_SECONDS "${STARTUP_TIMEOUT_SECONDS}"
  assert_integer STOP_TIMEOUT_SECONDS "${STOP_TIMEOUT_SECONDS}"
  assert_integer ARTIFACT_LOG_TAIL_BYTES "${ARTIFACT_LOG_TAIL_BYTES}"
  assert_integer ARTIFACT_MAX_BYTES "${ARTIFACT_MAX_BYTES}"
  require_command git
  require_dir "${AFD_PLUGIN_ROOT}"
  require_dir "${VLLM_ROOT}"
  require_dir "${VLLM_ASCEND_ROOT}"
  require_file "${PYTHON_BIN}"
  [[ "$(git_head "${VLLM_ROOT}")" == "${VLLM_COMMIT}" ]] \
    || die "vLLM commit mismatch"
  [[ "$(git_head "${VLLM_ASCEND_ROOT}")" == "${VLLM_ASCEND_COMMIT}" ]] \
    || die "vLLM-Ascend commit mismatch"
  [[ "$(git_head "${AFD_PLUGIN_ROOT}")" == "${AFD_PD_COMMIT}" ]] \
    || die "afd-plugin commit mismatch"
  [[ -z "$(git -C "${VLLM_ROOT}" status --short)" ]] \
    || die "vLLM worktree is dirty"
  [[ -z "$(git -C "${VLLM_ASCEND_ROOT}" status --short)" ]] \
    || die "vLLM-Ascend worktree is dirty"
  [[ -z "$(git -C "${AFD_PLUGIN_ROOT}" status --short)" ]] \
    || die "afd-plugin worktree is dirty"
}

local_role_ip() {
  case "${NODE_ROLE}" in
    prefill) printf '%s\n' "${PREFILL_IP}" ;;
    decode) printf '%s\n' "${DECODE_IP}" ;;
    proxy) printf '%s\n' "" ;;
  esac
}

validate_local_network() {
  [[ "${NODE_ROLE}" == "proxy" ]] && return 0
  require_command ip
  ip link show dev "${NIC_NAME}" >/dev/null 2>&1 \
    || die "Network interface not found: ${NIC_NAME}"
  local expected_ip
  expected_ip="$(local_role_ip)"
  ip -o -4 addr show dev "${NIC_NAME}" \
    | awk '{split($4, a, "/"); print a[1]}' \
    | grep -Fxq "${expected_ip}" \
    || die "${expected_ip} is not assigned to ${NIC_NAME}"
}

wheel_sha256() {
  sha256sum "${MOONCAKE_WHEEL}" | awk '{print $1}'
}

validate_wheel() {
  reject_placeholder MOONCAKE_WHEEL "${MOONCAKE_WHEEL:-}"
  require_file "${MOONCAKE_WHEEL}"
  [[ "$(wheel_sha256)" == "${MOONCAKE_WHEEL_SHA256}" ]] \
    || die "Mooncake wheel SHA256 mismatch"
}

npu_process_count() {
  npu-smi info | awk '
    /\| NPU +Chip +\| Process id/ {in_process_table=1; next}
    in_process_table && /^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+/ {count++}
    END {print count + 0}
  '
}

check_npus() {
  require_command npu-smi
  local expected=8
  [[ "${NODE_ROLE}" == "decode" ]] && expected=16
  local detected
  detected="$(npu-smi info -l | awk -F: '/Chip Count/ {gsub(/[[:space:]]/, "", $2); sum += $2} END {print sum + 0}')"
  (( detected >= expected )) || die "${NODE_ROLE} requires ${expected} NPUs, detected ${detected}"
  local process_count
  process_count="$(npu_process_count)"
  if (( process_count > 0 )) && ! is_true "${ALLOW_NPU_PROCESSES}"; then
    die "Detected ${process_count} existing NPU processes"
  fi
}

owned_pid_names() {
  case "${NODE_ROLE}" in
    prefill) printf '%s\n' prefill ;;
    decode) printf '%s\n' attention ffn ;;
    proxy) printf '%s\n' proxy ;;
  esac
}

owned_process() {
  local pid="$1"
  local cmdline
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${cmdline}" == *"${AFD_PLUGIN_ROOT}"* || "${cmdline}" == *"vllm"* ]]
}

ensure_not_running() {
  local name pid
  while read -r name; do
    if pid="$(read_pid "${STATE_ROOT}/${name}.pid" 2>/dev/null)" && pid_is_alive "${pid}"; then
      die "${name} is already running with PID ${pid}"
    fi
  done < <(owned_pid_names)
}

check_start_ports() {
  require_command ss
  local ports=()
  case "${NODE_ROLE}" in
    prefill) ports=("${PREFILL_API_PORT}" "${PREFILL_KV_PORT}" "${PREFILL_HCCL_IF_BASE_PORT}") ;;
    decode) ports=("${DECODE_API_PORT}" "${FFN_PROCESS_PORT}" "${AFD_PORT}" "${DECODE_KV_PORT}" "${ATTENTION_HCCL_IF_BASE_PORT}" "${FFN_HCCL_IF_BASE_PORT}") ;;
    proxy) ports=("${PROXY_PORT}") ;;
  esac
  local port
  for port in "${ports[@]}"; do
    port_is_listening "${port}" && die "Port is already listening: ${port}"
  done
}

export_runtime_env() {
  local local_ip
  local_ip="$(local_role_ip)"
  export DSV4_CANN_ROOT="${CANN_ROOT}"
  export DSV4_RUNTIME_VENV="${VENV_ROOT}"
  export DSV4_VLLM_ROOT="${VLLM_ROOT}"
  export DSV4_VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT}"
  export MODEL_PATH VLLM_HOST_IP="${local_ip}" HCCL_IF_IP="${local_ip}"
  export GLOO_SOCKET_IFNAME="${NIC_NAME}" HCCL_SOCKET_IFNAME="${NIC_NAME}"
  export MC_MIN_PRC_PORT MC_MAX_PRC_PORT MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS
  export MAX_NUM_SEQS GPU_MEMORY_UTILIZATION HCCL_BUFFSIZE OMP_NUM_THREADS
  export PREFILL_DEVICES PREFILL_DP_SIZE PREFILL_TP_SIZE DECODE_DP_SIZE DECODE_TP_SIZE
  export ATTENTION_DEVICES FFN_DEVICES ATTENTION_RANKS FFN_RANKS
  export PREFILL_HCCL_IF_BASE_PORT ATTENTION_HCCL_IF_BASE_PORT FFN_HCCL_IF_BASE_PORT
  export AFD_HOST=127.0.0.1 AFD_PORT TENSOR_PARALLEL_SIZE=1
  export EXECUTION_MODE=eager U_BATCHES=1 ENABLE_MTP=0
  export MOONCAKE_ENGINE_ID MOONCAKE_KV_PORT
}

wait_http() {
  local url="$1"
  shift
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  local pid
  while (( SECONDS < deadline )); do
    for pid in "$@"; do
      pid_is_alive "${pid}" || return 1
    done
    curl -fsS --max-time 5 "${url}" >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

new_log() {
  local name="$1"
  local path="${LOG_ROOT}/${name}-$(date +%Y%m%d_%H%M%S).log"
  ln -sfn "$(basename "${path}")" "${LOG_ROOT}/${name}.log"
  printf '%s\n' "${path}"
}

start_prefill() {
  export_runtime_env
  export API_HOST=0.0.0.0 API_PORT="${PREFILL_API_PORT}"
  export MOONCAKE_ENGINE_ID=dsv4-afd-prefill MOONCAKE_KV_PORT="${PREFILL_KV_PORT}"
  local log_path pid
  log_path="$(new_log prefill)"
  nohup setsid bash "${PREFILL_SCRIPT}" >"${log_path}" 2>&1 &
  pid=$!
  printf '%s\n' "${pid}" >"${STATE_ROOT}/prefill.pid"
  sleep 3
  pid_is_alive "${pid}" || die "Prefill exited immediately; inspect ${log_path}"
  if is_true "${WAIT_READY}"; then
    wait_http "http://127.0.0.1:${PREFILL_API_PORT}/health" "${pid}" \
      || die "Prefill readiness failed; inspect ${log_path}"
  fi
  log "Prefill started: pid=${pid}, log=${log_path}"
}

start_decode() {
  export_runtime_env
  export MOONCAKE_ENGINE_ID=dsv4-afd-decode MOONCAKE_KV_PORT="${DECODE_KV_PORT}"
  local ffn_log attention_log ffn_pid attention_pid deadline ready_count
  ffn_log="$(new_log ffn)"
  attention_log="$(new_log attention)"
  API_HOST=0.0.0.0 API_PORT="${FFN_PROCESS_PORT}" \
    nohup setsid bash "${FFN_SCRIPT}" >"${ffn_log}" 2>&1 &
  ffn_pid=$!
  printf '%s\n' "${ffn_pid}" >"${STATE_ROOT}/ffn.pid"
  sleep 2
  ENABLE_PD=1 API_HOST=0.0.0.0 API_PORT="${DECODE_API_PORT}" \
    nohup setsid bash "${ATTENTION_SCRIPT}" >"${attention_log}" 2>&1 &
  attention_pid=$!
  printf '%s\n' "${attention_pid}" >"${STATE_ROOT}/attention.pid"
  sleep 3
  if ! pid_is_alive "${ffn_pid}" || ! pid_is_alive "${attention_pid}"; then
    warn "A Decode role exited immediately"
    stop_action || true
    die "Decode startup failed; inspect ${LOG_ROOT}"
  fi
  if is_true "${WAIT_READY}"; then
    deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
    while (( SECONDS < deadline )); do
      pid_is_alive "${ffn_pid}" && pid_is_alive "${attention_pid}" \
        || die "Decode role exited before readiness"
      ready_count="$( { grep -o 'AFD FFN EngineCore started; workers run connector loop' "${ffn_log}" 2>/dev/null || true; } | wc -l)"
      if (( ready_count >= FFN_RANKS )) \
        && curl -fsS --max-time 5 "http://127.0.0.1:${DECODE_API_PORT}/health" >/dev/null 2>&1; then
        log "Decode ready: Attention health OK, FFN loops=${ready_count}/${FFN_RANKS}"
        return 0
      fi
      sleep 5
    done
    die "Decode readiness timed out; inspect ${LOG_ROOT}"
  fi
  log "Decode started: attention=${attention_pid}, ffn=${ffn_pid}"
}

start_proxy() {
  export PREFILL_HOSTS="${PREFILL_IP}" PREFILL_PORTS="${PREFILL_API_PORT}"
  export DECODE_HOSTS="${DECODE_IP}" DECODE_PORTS="${DECODE_API_PORT}"
  export PROXY_HOST=0.0.0.0 PROXY_PORT DSV4_RUNTIME_VENV="${VENV_ROOT}"
  export DSV4_VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT}"
  curl -fsS --max-time 10 "http://${PREFILL_IP}:${PREFILL_API_PORT}/health" >/dev/null \
    || die "Prefill backend is not healthy"
  curl -fsS --max-time 10 "http://${DECODE_IP}:${DECODE_API_PORT}/health" >/dev/null \
    || die "Decode backend is not healthy"
  local log_path pid
  log_path="$(new_log proxy)"
  nohup setsid bash "${PROXY_SCRIPT}" >"${log_path}" 2>&1 &
  pid=$!
  printf '%s\n' "${pid}" >"${STATE_ROOT}/proxy.pid"
  sleep 2
  pid_is_alive "${pid}" || die "Proxy exited immediately; inspect ${log_path}"
  if is_true "${WAIT_READY}"; then
    wait_http "http://127.0.0.1:${PROXY_PORT}/healthcheck" "${pid}" \
      || die "Proxy readiness failed; inspect ${log_path}"
  fi
  log "Proxy started: pid=${pid}, log=${log_path}"
}

status_action() {
  validate_role
  local overall=0 name pid cmdline log_path ready_count transfer_count
  while read -r name; do
    if pid="$(read_pid "${STATE_ROOT}/${name}.pid" 2>/dev/null)" && pid_is_alive "${pid}"; then
      cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
      log "${name}: RUNNING pid=${pid} command=${cmdline}"
    else
      warn "${name}: NOT RUNNING"
      overall=1
    fi
  done < <(owned_pid_names)
  case "${NODE_ROLE}" in
    prefill)
      curl -fsS --max-time 5 "http://127.0.0.1:${PREFILL_API_PORT}/health" >/dev/null 2>&1 \
        && log "Prefill health: OK" || { warn "Prefill health: NOT READY"; overall=1; }
      ;;
    decode)
      curl -fsS --max-time 5 "http://127.0.0.1:${DECODE_API_PORT}/health" >/dev/null 2>&1 \
        && log "Attention health: OK" || { warn "Attention health: NOT READY"; overall=1; }
      log_path="${LOG_ROOT}/ffn.log"
      ready_count="$( { grep -o 'AFD FFN EngineCore started; workers run connector loop' "${log_path}" 2>/dev/null || true; } | wc -l)"
      log "FFN connector loops: ${ready_count}/${FFN_RANKS}"
      (( ready_count >= FFN_RANKS )) || overall=1
      transfer_count="$( { grep -c 'KV cache transfer for request .* took .* remote_session_id' "${LOG_ROOT}/attention.log" 2>/dev/null || true; } )"
      log "Successful Mooncake KV transfer records: ${transfer_count}"
      ;;
    proxy)
      curl -fsS --max-time 5 "http://127.0.0.1:${PROXY_PORT}/healthcheck" >/dev/null 2>&1 \
        && log "Proxy health: OK" || { warn "Proxy health: NOT READY"; overall=1; }
      ;;
  esac
  local fatal=0
  for log_path in "${LOG_ROOT}"/*.log; do
    [[ -f "${log_path}" ]] || continue
    if grep -En "${FATAL_PATTERN}" "${log_path}"; then
      fatal=1
    fi
  done
  (( fatal == 0 )) || { warn "Fatal markers found"; overall=1; }
  return "${overall}"
}

wait_for_group_exit() {
  local pgid="$1"
  local deadline=$((SECONDS + STOP_TIMEOUT_SECONDS))
  while kill -0 -- "-${pgid}" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 2
  done
  ! kill -0 -- "-${pgid}" 2>/dev/null
}

stop_name() {
  local name="$1"
  local pid_path="${STATE_ROOT}/${name}.pid"
  local pid pgid
  if ! pid="$(read_pid "${pid_path}" 2>/dev/null)"; then
    log "${name}: no PID file"
    return 0
  fi
  if ! pid_is_alive "${pid}"; then
    log "${name}: already stopped"
    rm -f "${pid_path}"
    return 0
  fi
  owned_process "${pid}" || die "Refusing to signal unrecognized PID ${pid}"
  pgid="$(ps -o pgid= -p "${pid}" | tr -d '[:space:]')"
  [[ "${pgid}" == "${pid}" ]] || die "PID ${pid} is not its process-group leader"
  log "Stopping ${name} process group ${pgid}"
  kill -TERM -- "-${pgid}"
  if wait_for_group_exit "${pgid}"; then
    rm -f "${pid_path}"
    return 0
  fi
  if is_true "${FORCE_KILL}"; then
    warn "TERM timed out for ${name}; FORCE_KILL=1"
    kill -KILL -- "-${pgid}"
    wait_for_group_exit "${pgid}" || die "Process group ${pgid} did not exit"
    rm -f "${pid_path}"
    return 0
  fi
  die "${name} did not exit; inspect it before setting FORCE_KILL=1"
}

stop_action() {
  mkdir -p "${STATE_ROOT}"
  case "${NODE_ROLE}" in
    proxy) stop_name proxy ;;
    decode) stop_name attention; stop_name ffn ;;
    prefill) stop_name prefill ;;
  esac
  log "Stop complete; run npu-smi info on NPU nodes"
}

install_action() {
  validate_common_config
  [[ "${NODE_ROLE}" == "proxy" ]] && { log "Proxy needs no Mooncake wheel install"; return 0; }
  validate_wheel
  if is_true "${INSTALL_SYSTEM_PACKAGES}"; then
    require_command sudo
    sudo apt-get update
    sudo apt-get install -y libgoogle-glog0v6t64 libjsoncpp25 libjemalloc2 netcat-openbsd
  fi
  AFD_BUILD_ASCEND_OPS=0 "${PYTHON_BIN}" -m pip install \
    --no-build-isolation --no-deps --editable "${AFD_PLUGIN_ROOT}"
  "${PYTHON_BIN}" -m pip install --no-deps --force-reinstall "${MOONCAKE_WHEEL}"
  "${PYTHON_BIN}" -m pip show vllm vllm-ascend vllm-afd-plugin mooncake-transfer-engine
  log "Install complete; next run: bash $0 check ${CONFIG_FILE}"
}

check_action() {
  validate_common_config
  validate_local_network
  require_command curl
  require_command ss
  mkdir -p "${STATE_ROOT}" "${LOG_ROOT}" "${VALIDATION_ROOT}" "${OUTPUT_ROOT}"
  if [[ "${NODE_ROLE}" == "proxy" ]]; then
    require_file "${PROXY_SCRIPT}"
    require_file "${GOLDEN_VALIDATOR}"
    ensure_not_running
    check_start_ports
    log "Proxy preflight passed"
    return 0
  fi
  require_file "${CANN_ROOT}/set_env.sh"
  [[ "$(readlink -f "${CANN_ROOT}")" == *9.0.1* ]] || die "CANN_ROOT is not 9.0.1"
  require_file "${MODEL_PATH}/config.json"
  require_file "${RUNTIME_CHECK}"
  require_file "${ROUNDTRIP_TOOL}"
  validate_wheel
  check_npus
  ensure_not_running
  check_start_ports
  export DSV4_CANN_ROOT="${CANN_ROOT}" DSV4_RUNTIME_VENV="${VENV_ROOT}"
  export DSV4_VLLM_ROOT="${VLLM_ROOT}" DSV4_VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT}"
  bash "${RUNTIME_CHECK}" >"${STATE_ROOT}/runtime-check.log" 2>&1 \
    || { tail -n 100 "${STATE_ROOT}/runtime-check.log" >&2; die "Mooncake runtime check failed"; }
  if is_true "${RUN_LOCAL_ROUNDTRIP}"; then
    bash -c 'source "$1"; exec python "$2" --producer-device "$3" --consumer-device "$4"' \
      pd-roundtrip "${RUNTIME_CHECK}" "${ROUNDTRIP_TOOL}" \
      "${ROUNDTRIP_PRODUCER_DEVICE}" "${ROUNDTRIP_CONSUMER_DEVICE}" \
      >"${STATE_ROOT}/roundtrip.json" 2>"${STATE_ROOT}/roundtrip.stderr"
  fi
  log "Preflight passed; runtime=${STATE_ROOT}/runtime-check.log"
}

start_action() {
  local configured_roundtrip="${RUN_LOCAL_ROUNDTRIP}"
  RUN_LOCAL_ROUNDTRIP=0
  check_action
  RUN_LOCAL_ROUNDTRIP="${configured_roundtrip}"
  mkdir -p "${STATE_ROOT}" "${LOG_ROOT}"
  ensure_not_running
  case "${NODE_ROLE}" in
    prefill) require_file "${PREFILL_SCRIPT}"; start_prefill ;;
    decode) require_file "${ATTENTION_SCRIPT}"; require_file "${FFN_SCRIPT}"; start_decode ;;
    proxy) require_file "${PROXY_SCRIPT}"; start_proxy ;;
  esac
}

validate_action() {
  [[ "${NODE_ROLE}" == "proxy" ]] || die "validate must run with NODE_ROLE=proxy"
  status_action
  require_file "${GOLDEN_PATH}"
  require_file "${GOLDEN_VALIDATOR}"
  local run_dir endpoint cancel_rc
  run_dir="${VALIDATION_ROOT}/m9-f0-$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${run_dir}" "${STATE_ROOT}"
  endpoint="http://127.0.0.1:${PROXY_PORT}/v1/completions"
  curl -fsS --max-time 600 "${endpoint}" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL_NAME}\",\"prompt\":\"PD smoke request.\",\"temperature\":0,\"seed\":1024,\"max_tokens\":16,\"stream\":false,\"return_token_ids\":true}" \
    >"${run_dir}/smoke.json"
  read -r -a batch_sizes <<<"${VALIDATION_BATCH_SIZES}"
  "${PYTHON_BIN}" "${GOLDEN_VALIDATOR}" \
    --endpoint "${endpoint}" --model "${MODEL_NAME}" \
    --golden "${GOLDEN_PATH}" --rounds "${VALIDATION_ROUNDS}" \
    --batch-sizes "${batch_sizes[@]}" --output "${run_dir}/golden.json"
  if is_true "${RUN_CANCELLATION_TEST}"; then
    set +e
    curl -fsS --max-time 1 "${endpoint}" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${MODEL_NAME}\",\"prompt\":\"Write a detailed deterministic systems validation checklist.\",\"temperature\":0,\"seed\":1024,\"max_tokens\":512,\"stream\":false}" \
      >"${run_dir}/cancellation-response.json" 2>"${run_dir}/cancellation.stderr"
    cancel_rc=$?
    set -e
    printf '%s\n' "${cancel_rc}" >"${run_dir}/cancellation.exitcode"
    [[ "${cancel_rc}" == "28" ]] || die "Cancellation gate expected curl exit 28, got ${cancel_rc}"
    curl -fsS --max-time 10 "http://127.0.0.1:${PROXY_PORT}/healthcheck" \
      >"${run_dir}/health-after-cancellation.json"
  fi
  {
    printf 'status=proxy_validation_passed_decode_transfer_evidence_required\n'
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'golden=%s\n' "${run_dir}/golden.json"
  } >"${run_dir}/summary.env"
  printf '%s\n' "${run_dir}" >"${STATE_ROOT}/last-validation-dir"
  log "Proxy validation passed: ${run_dir}"
  log "Run collect on proxy, decode, and prefill; Decode artifact must contain KV transfer evidence"
}

copy_log_tail() {
  local source_path="$1"
  local output_path="$2"
  [[ -f "${source_path}" ]] || return 0
  tail -c "${ARTIFACT_LOG_TAIL_BYTES}" "${source_path}" >"${output_path}"
}

collect_action() {
  validate_role
  require_command tar
  require_command sha256sum
  mkdir -p "${STATE_ROOT}" "${OUTPUT_ROOT}"
  local temp_dir archive timestamp size_bytes status_rc name log_path validation_dir
  temp_dir="$(mktemp -d "${STATE_ROOT}/collect.XXXXXX")"
  COLLECT_TEMP_DIR="${temp_dir}"
  trap '[[ -z "${COLLECT_TEMP_DIR:-}" ]] || rm -rf -- "${COLLECT_TEMP_DIR}"' EXIT
  timestamp="$(date +%Y%m%d_%H%M%S)"
  archive="${OUTPUT_ROOT}/dsv4-m9-pd-${NODE_ROLE}-${timestamp}.tar.gz"
  {
    printf 'node_role=%s\n' "${NODE_ROLE}"
    printf 'collected_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'prefill_ip=%s\n' "${PREFILL_IP}"
    printf 'decode_ip=%s\n' "${DECODE_IP}"
    printf 'nic_name=%s\n' "${NIC_NAME}"
    printf 'cann_root=%s\n' "$(readlink -f "${CANN_ROOT}" 2>/dev/null || printf '%s' "${CANN_ROOT}")"
    printf 'vllm_commit=%s\n' "$(git_head "${VLLM_ROOT}" 2>/dev/null || true)"
    printf 'vllm_ascend_commit=%s\n' "$(git_head "${VLLM_ASCEND_ROOT}" 2>/dev/null || true)"
    printf 'afd_commit=%s\n' "$(git_head "${AFD_PLUGIN_ROOT}" 2>/dev/null || true)"
    printf 'mooncake_wheel_sha256=%s\n' "${MOONCAKE_WHEEL_SHA256}"
  } >"${temp_dir}/summary.env"
  {
    git -C "${VLLM_ROOT}" status --short 2>/dev/null || true
    git -C "${VLLM_ASCEND_ROOT}" status --short 2>/dev/null || true
    git -C "${AFD_PLUGIN_ROOT}" status --short 2>/dev/null || true
  } >"${temp_dir}/git-status.txt"
  "${PYTHON_BIN}" -m pip show torch torch-npu vllm vllm-ascend vllm-afd-plugin mooncake-transfer-engine \
    >"${temp_dir}/python-packages.txt" 2>&1 || true
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info >"${temp_dir}/npu-smi.txt" 2>&1 || true
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp >"${temp_dir}/ports.txt" 2>&1 || true
  fi
  set +e
  status_action >"${temp_dir}/status.txt" 2>&1
  status_rc=$?
  set -e
  printf '%s\n' "${status_rc}" >"${temp_dir}/status.exitcode"
  for name in prefill attention ffn proxy; do
    log_path="${LOG_ROOT}/${name}.log"
    copy_log_tail "${log_path}" "${temp_dir}/${name}.tail.log"
  done
  cp "${STATE_ROOT}/runtime-check.log" "${temp_dir}/" 2>/dev/null || true
  cp "${STATE_ROOT}/roundtrip.json" "${temp_dir}/" 2>/dev/null || true
  cp "${STATE_ROOT}/roundtrip.stderr" "${temp_dir}/" 2>/dev/null || true
  if [[ -f "${STATE_ROOT}/last-validation-dir" ]]; then
    read -r validation_dir <"${STATE_ROOT}/last-validation-dir"
    case "${validation_dir}" in
      "${VALIDATION_ROOT}"/*)
        for name in smoke.json golden.json cancellation.exitcode cancellation.stderr health-after-cancellation.json summary.env; do
          [[ -f "${validation_dir}/${name}" ]] && cp "${validation_dir}/${name}" "${temp_dir}/validation-${name}"
        done
        ;;
      *) warn "Ignoring validation path outside VALIDATION_ROOT: ${validation_dir}" ;;
    esac
  fi
  { grep -Eh 'KV cache transfer for request .* took .* remote_session_id' \
      "${LOG_ROOT}/attention.log" 2>/dev/null || true; } \
    | tail -n 50 >"${temp_dir}/kv-transfer-evidence.txt"
  { grep -Enh "${FATAL_PATTERN}" "${LOG_ROOT}"/*.log 2>/dev/null || true; } \
    | tail -n 200 >"${temp_dir}/fatal-markers.txt"
  find "${temp_dir}" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort >"${temp_dir}/manifest.txt"
  tar -czf "${archive}" -C "${temp_dir}" .
  size_bytes="$(stat -c '%s' "${archive}")"
  if (( size_bytes > ARTIFACT_MAX_BYTES )); then
    rm -f "${archive}"
    die "Artifact exceeded ${ARTIFACT_MAX_BYTES} bytes; lower ARTIFACT_LOG_TAIL_BYTES"
  fi
  sha256sum "${archive}" >"${archive}.sha256"
  rm -rf -- "${temp_dir}"
  COLLECT_TEMP_DIR=""
  trap - EXIT
  log "ARTIFACT=${archive}"
  log "ARTIFACT_SIZE_BYTES=${size_bytes}"
  log "SHA256_FILE=${archive}.sha256"
}

print_config_action() {
  validate_role
  printf 'NODE_ROLE=%s\n' "${NODE_ROLE}"
  printf 'PREFILL_IP=%s\n' "${PREFILL_IP:-}"
  printf 'DECODE_IP=%s\n' "${DECODE_IP:-}"
  printf 'NIC_NAME=%s\n' "${NIC_NAME:-}"
  printf 'AFD_PD_COMMIT=%s\n' "${AFD_PD_COMMIT:-}"
  printf 'CANN_ROOT=%s\n' "${CANN_ROOT}"
  printf 'VENV_ROOT=%s\n' "${VENV_ROOT}"
  printf 'MODEL_PATH=%s\n' "${MODEL_PATH}"
  printf 'STATE_ROOT=%s\n' "${STATE_ROOT}"
  printf 'LOG_ROOT=%s\n' "${LOG_ROOT}"
  printf 'OUTPUT_ROOT=%s\n' "${OUTPUT_ROOT}"
}

case "${ACTION}" in
  print-config) print_config_action ;;
  install) install_action ;;
  check) check_action ;;
  start) start_action ;;
  status) status_action ;;
  validate) validate_action ;;
  stop) stop_action ;;
  collect) collect_action ;;
  *) usage; die "Unknown action: ${ACTION}" ;;
esac
