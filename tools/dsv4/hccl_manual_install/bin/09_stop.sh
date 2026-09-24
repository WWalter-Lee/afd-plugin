#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

owned_process() {
  local pid="$1"
  local cmdline
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${cmdline}" == *"${SCRIPT_DIR}/run_role.sh"* || "${cmdline}" == *"vllm"* ]]
}

wait_for_exit() {
  local process_group="$1"
  local timeout="$2"
  local deadline=$((SECONDS + timeout))
  while kill -0 -- "-${process_group}" 2>/dev/null \
    && (( SECONDS < deadline )); do
    sleep 2
  done
  ! kill -0 -- "-${process_group}" 2>/dev/null
}

stop_role() {
  local role_name="$1"
  local pid_path="${STATE_ROOT}/${role_name}.pid"
  local pid
  if ! pid="$(read_pid_file "${pid_path}" 2>/dev/null)"; then
    log "${role_name}: no PID file"
    return 0
  fi
  if ! pid_is_alive "${pid}"; then
    log "${role_name}: PID ${pid} is already stopped"
    rm -f "${pid_path}"
    return 0
  fi
  owned_process "${pid}" \
    || die "Refusing to signal unrecognized PID ${pid} from ${pid_path}"
  process_group="$(ps -o pgid= -p "${pid}" | tr -d '[:space:]')"
  [[ "${process_group}" == "${pid}" ]] \
    || die "Refusing group stop: PID ${pid} is not its process-group leader"

  log "Stopping ${role_name} process group ${process_group}"
  kill -TERM -- "-${process_group}"
  if wait_for_exit "${process_group}" "${STOP_TIMEOUT_SECONDS}"; then
    rm -f "${pid_path}"
    log "${role_name}: stopped"
    return 0
  fi

  if is_true "${FORCE_KILL}"; then
    warn "${role_name}: TERM timed out; FORCE_KILL=1, sending KILL"
    kill -KILL -- "-${process_group}"
    wait_for_exit "${process_group}" 30 \
      || die "${role_name} process group ${process_group} did not exit"
    rm -f "${pid_path}"
    return 0
  fi

  die "${role_name} PID ${pid} did not exit; inspect it before setting FORCE_KILL=1"
}

# Protocol shutdown order is part of the validated lifecycle.
stop_role attention
stop_role ffn

log "Stop sequence completed. Verify NPU cleanup with: npu-smi info"
