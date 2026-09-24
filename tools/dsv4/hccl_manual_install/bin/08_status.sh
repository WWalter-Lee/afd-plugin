#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

overall_status=0
for role_name in attention ffn; do
  pid_path="${STATE_ROOT}/${role_name}.pid"
  if pid="$(read_pid_file "${pid_path}" 2>/dev/null)" && pid_is_alive "${pid}"; then
    cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
    log "${role_name}: RUNNING pid=${pid} command=${cmdline}"
  else
    warn "${role_name}: NOT RUNNING"
    overall_status=1
  fi
done

if curl -fsS --max-time 5 \
  "http://${API_HOST}:${ATTENTION_API_PORT}/health" >/dev/null 2>&1; then
  log "Attention health: OK"
else
  warn "Attention health: NOT READY"
  overall_status=1
fi

ffn_log="${LOG_ROOT}/ffn.log"
if [[ -f "${ffn_log}" ]]; then
  ffn_ready_count="$(
    { grep -o \
        'AFD FFN EngineCore started; workers run connector loop' \
        "${ffn_log}" 2>/dev/null || true; } | wc -l
  )"
  log "FFN connector loops: ${ffn_ready_count}/${FFN_RANKS}"
  if (( ffn_ready_count < FFN_RANKS )); then
    overall_status=1
  fi
else
  warn "FFN log is missing: ${ffn_log}"
  overall_status=1
fi

fatal_pattern='EngineCore encountered a fatal error|AFD NPU FFN worker loop failed|Communication_Error|507015|Traceback'
fatal_found=0
for log_path in "${LOG_ROOT}/attention.log" "${LOG_ROOT}/ffn.log"; do
  if [[ -f "${log_path}" ]] && grep -En "${fatal_pattern}" "${log_path}"; then
    fatal_found=1
  fi
done
if (( fatal_found )); then
  warn "Fatal markers found in role logs"
  overall_status=1
else
  log "Fatal log gate: clean"
fi

exit "${overall_status}"
