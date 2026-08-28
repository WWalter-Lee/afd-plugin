#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-help}"
CONFIG_FILE="${2:-${BI_CONFIG_FILE:-${SCRIPT_DIR}/config.env}}"
BASE_COMMIT="3da28f9414583d2d0b672a8f06d1fae142404bda"
PATCH_SHA256="cf97a0b6e509fbb128e847babbf8f01cc953f06cb3126936cc4111bbab60b897"
OPP_SHA256="9fc692978e9420336e3fea03a92c2a85df1b50a65a7df50173e3bf8bedaea70e"
WHEEL_SHA256="a5ae4cfbad39e47ba4233d1fa799b5d469960ff2f266eded4de3dc69ac0c0898"
OPP_URL="https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/cann-ops-batch_invariant-A3-1.0.0-linux.aarch64.run"
PATCH_FILE="${SCRIPT_DIR}/patches/vllm-ascend-dsv4-batch-invariant-3da28f9.patch"

usage() {
  cat <<'EOF'
Usage: bash bi.sh <action> [config.env]

Actions:
  init          Create a config from config.env.example.
  print-config  Print the effective paths, node label, IP and topology.
  download-opp  Download the official 105 MiB A3 custom OPP and verify SHA256.
  apply-patch   Apply and verify the vLLM-Ascend compatibility patch.
  install       Create the isolated venv and install the OPP/Python extension.
  check         Run source, CANN, custom-op, unit and NPU smoke checks.
  run-twice     Run two cold-start direct Prefill 10-prompt x 3-round checks.
  collect       Create a size-capped support archive for the latest result.
EOF
}

log() {
  printf '[dsv4-bi:%s] %s\n' "${NODE_LABEL:-unconfigured}" "$*"
}

die() {
  printf '[dsv4-bi:%s] ERROR: %s\n' "${NODE_LABEL:-unconfigured}" "$*" >&2
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

[[ -f "${CONFIG_FILE}" ]] || die "Missing config: ${CONFIG_FILE}"
set -a
# shellcheck disable=SC1090
source "${CONFIG_FILE}"
set +a

: "${CODE_ROOT:=/data/z00569729/code}"
: "${BASE_VENV:=${CODE_ROOT}/.venvs/afd-v023-vllm-cann}"
: "${BI_VENV:=${CODE_ROOT}/.venvs/afd-v023-vllm-cann-batch-invariant}"
: "${VLLM_ROOT:=${CODE_ROOT}/vllm-release-v0.23.0}"
: "${VLLM_ASCEND_ROOT:=${CODE_ROOT}/vllm-ascend-rfc-vllm-cann}"
: "${AFD_PLUGIN_ROOT:=${CODE_ROOT}/afd-plugin}"
: "${CANN_ROOT:=/usr/local/Ascend/cann-9.0.0}"
: "${ATB_ROOT:=}"
: "${MODEL_PATH:=/data/z00569729/models/DeepSeek-V4-Flash-w8a8-mtp}"
: "${BATCH_INVARIANT_OPP_RUN:=/data/z00569729/packages/cann-ops-batch_invariant-A3-1.0.0-linux.aarch64.run}"
: "${BATCH_INVARIANT_OPP_ROOT:=${CODE_ROOT}/.ascend/custom-opp/batch-invariant-a3-1.0.0}"
: "${BATCH_INVARIANT_WHEEL:=${SCRIPT_DIR}/packages/batch_invariant_ops-1.0.0-cp312-cp312-linux_aarch64.whl}"
: "${VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${API_PORT:=8930}"
: "${HCCL_IF_BASE_PORT:=50000}"
: "${DATA_PARALLEL_RPC_PORT:=29650}"
: "${MASTER_PORT:=29651}"
: "${PROMPT_SOURCE:=/data/z00569729/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json}"
: "${RESULT_ROOT:=/data/z00569729/validation/dsv4_batch_invariant_dual_a3}"
: "${STARTUP_TIMEOUT_SECONDS:=3600}"
: "${STOP_TIMEOUT_SECONDS:=300}"

PATCHED_FILES=(vllm_ascend/batch_invariant.py tests/ut/test_batch_invariant.py)
BASE_PYTHON="${BASE_VENV}/bin/python"
BI_PYTHON="${BI_VENV}/bin/python"
CUSTOM_OPP_ENV="${BATCH_INVARIANT_OPP_ROOT}/vendors/batch_invariant/bin/set_env.bash"
COMPARE_TOOL="${SCRIPT_DIR}/compare_batch_invariant_runs.py"
GOLDEN_TOOL="${AFD_PLUGIN_ROOT}/tools/dsv4/generate_golden.py"
CURRENT_PID=""

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

git_in() {
  local repo="$1"
  shift
  git -c safe.directory="${repo}" -C "${repo}" "$@"
}

validate_static_config() {
  [[ -n "${NODE_LABEL:-}" && "${NODE_LABEL}" != CHANGE_ME* ]] \
    || die "Set NODE_LABEL in ${CONFIG_FILE}"
  [[ -n "${LOCAL_IP:-}" && "${LOCAL_IP}" != CHANGE_ME* ]] \
    || die "Set LOCAL_IP in ${CONFIG_FILE}"
  require_file "${BASE_PYTHON}"
  require_dir "${VLLM_ROOT}"
  require_dir "${VLLM_ASCEND_ROOT}"
  require_dir "${AFD_PLUGIN_ROOT}"
  require_dir "${MODEL_PATH}"
  require_file "${CANN_ROOT}/set_env.sh"
  require_file "${PATCH_FILE}"
  require_file "${COMPARE_TOOL}"
  require_file "${GOLDEN_TOOL}"
  require_file "${PROMPT_SOURCE}"
  [[ "$(git_in "${VLLM_ROOT}" rev-parse HEAD)" == "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665" ]] \
    || die "vLLM must be releases/v0.23.0 at 0fc695fc..."
  [[ "$(git_in "${VLLM_ASCEND_ROOT}" rev-parse HEAD)" == "${BASE_COMMIT}" ]] \
    || die "vLLM-Ascend must be rfc/vllm_cann at ${BASE_COMMIT}"
  [[ -z "$(git_in "${VLLM_ROOT}" status --short)" ]] \
    || die "vLLM worktree is dirty"
}

patch_diff_sha256() {
  git_in "${VLLM_ASCEND_ROOT}" diff -- "${PATCHED_FILES[@]}" | sha256sum | awk '{print $1}'
}

validate_patch() {
  local status diff_sha
  status="$(git_in "${VLLM_ASCEND_ROOT}" status --short --untracked-files=no)"
  [[ "${status}" == $' M tests/ut/test_batch_invariant.py\n M vllm_ascend/batch_invariant.py' ]] \
    || die "vLLM-Ascend has changes outside the expected two-file patch: ${status:-clean}"
  diff_sha="$(patch_diff_sha256)"
  [[ "${diff_sha}" == "${PATCH_SHA256}" ]] \
    || die "vLLM-Ascend patch fingerprint mismatch: ${diff_sha}"
  git_in "${VLLM_ASCEND_ROOT}" diff --check -- "${PATCHED_FILES[@]}"
}

apply_patch_action() {
  validate_static_config
  if git_in "${VLLM_ASCEND_ROOT}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    validate_patch
    log "Patch is already applied and fingerprinted"
    return 0
  fi
  [[ -z "$(git_in "${VLLM_ASCEND_ROOT}" status --short --untracked-files=no)" ]] \
    || die "Refusing to apply patch to a dirty vLLM-Ascend worktree"
  git_in "${VLLM_ASCEND_ROOT}" apply --check "${PATCH_FILE}"
  git_in "${VLLM_ASCEND_ROOT}" apply "${PATCH_FILE}"
  validate_patch
  log "Applied vLLM-Ascend batch-invariant patch"
}

resolve_atb_root() {
  if [[ -n "${ATB_ROOT}" ]]; then
    return 0
  fi
  if [[ -f "${CANN_ROOT}/nnal/atb/set_env.sh" ]]; then
    ATB_ROOT="${CANN_ROOT}/nnal/atb"
  elif [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    ATB_ROOT=/usr/local/Ascend/nnal/atb
  else
    die "Set ATB_ROOT to a directory containing set_env.sh"
  fi
}

activate_bi_runtime() {
  resolve_atb_root
  export DSV4_CANN_ROOT="${CANN_ROOT}"
  export DSV4_ATB_ROOT="${ATB_ROOT}"
  export DSV4_VLLM_VENV="${BI_VENV}"
  # shellcheck disable=SC1091
  source "${AFD_PLUGIN_ROOT}/tools/dsv4/activate_runtime.sh"
  local transformer_env="${VLLM_ASCEND_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
  if [[ -f "${transformer_env}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${transformer_env}"
    set -u
  fi
  set +u
  # shellcheck disable=SC1090
  source "${CUSTOM_OPP_ENV}"
  set -u
  export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector
  export VLLM_BATCH_INVARIANT=1
  export HCCL_DETERMINISTIC=true
  export LCCL_DETERMINISTIC=1
  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  export ATB_LLM_LCOC_ENABLE=0
  unset VLLM_ASCEND_ENABLE_FLASHCOMM1
  case "${PATH}:${LD_LIBRARY_PATH:-}:${PYTHONPATH:-}:${ASCEND_HOME_PATH:-}" in
    *cann-9.1.0*) die "CANN 9.1.0 leaked into the CANN 9.0.0 runtime" ;;
  esac
}

download_opp_action() {
  validate_static_config
  command -v curl >/dev/null 2>&1 || die "curl is required"
  mkdir -p "$(dirname "${BATCH_INVARIANT_OPP_RUN}")"
  if [[ -f "${BATCH_INVARIANT_OPP_RUN}" ]] \
    && [[ "$(sha256_file "${BATCH_INVARIANT_OPP_RUN}")" == "${OPP_SHA256}" ]]; then
    log "Official OPP already exists with the expected SHA256"
    return 0
  fi
  curl -fL --retry 3 --connect-timeout 30 \
    -o "${BATCH_INVARIANT_OPP_RUN}.part" "${OPP_URL}"
  [[ "$(sha256_file "${BATCH_INVARIANT_OPP_RUN}.part")" == "${OPP_SHA256}" ]] \
    || die "Downloaded OPP SHA256 mismatch"
  mv "${BATCH_INVARIANT_OPP_RUN}.part" "${BATCH_INVARIANT_OPP_RUN}"
  log "Downloaded and verified ${BATCH_INVARIANT_OPP_RUN}"
}

install_opp() {
  require_file "${BATCH_INVARIANT_OPP_RUN}"
  [[ "$(sha256_file "${BATCH_INVARIANT_OPP_RUN}")" == "${OPP_SHA256}" ]] \
    || die "Official OPP SHA256 mismatch"
  if [[ ! -f "${CUSTOM_OPP_ENV}" ]]; then
    mkdir -p "${BATCH_INVARIANT_OPP_ROOT}" "${RESULT_ROOT}/installer-tmp"
    TMPDIR="${RESULT_ROOT}/installer-tmp" \
      bash "${BATCH_INVARIANT_OPP_RUN}" --quiet \
      --install-path="${BATCH_INVARIANT_OPP_ROOT}"
  fi
  require_file "${CUSTOM_OPP_ENV}"

  local opp_parent opp_link expected_opp config_dir delivered_name expected_name
  opp_parent="$(dirname "${BATCH_INVARIANT_OPP_ROOT}")"
  opp_link="${opp_parent}/opp"
  expected_opp="$(readlink -f "${CANN_ROOT}/opp")"
  if [[ ! -e "${opp_link}" && ! -L "${opp_link}" ]]; then
    ln -s "${expected_opp}" "${opp_link}"
  fi
  [[ "$(readlink -f "${opp_link}")" == "${expected_opp}" ]] \
    || die "${opp_link} must point to the fixed CANN 9.0.0 opp directory"

  config_dir="${BATCH_INVARIANT_OPP_ROOT}/vendors/batch_invariant/op_impl/ai_core/tbe/kernel/config/ascend910_93"
  delivered_name="${config_dir}/mat_mul_v3_batch_invariant.json"
  expected_name="${config_dir}/mat_mul_v_3_batch_invariant.json"
  require_file "${delivered_name}"
  if [[ ! -e "${expected_name}" && ! -L "${expected_name}" ]]; then
    ln -s "$(basename "${delivered_name}")" "${expected_name}"
  fi
  require_file "${expected_name}"
}

install_venv() {
  require_file "${BATCH_INVARIANT_WHEEL}"
  [[ "$(sha256_file "${BATCH_INVARIANT_WHEEL}")" == "${WHEEL_SHA256}" ]] \
    || die "batch_invariant_ops wheel SHA256 mismatch"
  if [[ ! -x "${BI_PYTHON}" ]]; then
    "${BASE_PYTHON}" -m venv "${BI_VENV}"
  fi
  local base_site bi_site pth_file
  base_site="$("${BASE_PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  bi_site="$("${BI_PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  pth_file="${bi_site}/dsv4_v023_batch_invariant.pth"
  printf '%s\n' \
    "${VLLM_ROOT}" \
    "${VLLM_ASCEND_ROOT}" \
    "${AFD_PLUGIN_ROOT}" \
    "${base_site}" >"${pth_file}"
  "${BI_PYTHON}" -m pip install --no-deps --force-reinstall \
    "${BATCH_INVARIANT_WHEEL}"
}

install_action() {
  validate_static_config
  validate_patch
  install_opp
  install_venv
  log "Install complete; run check on this node"
}

check_action() {
  validate_static_config
  validate_patch
  require_file "${BI_PYTHON}"
  require_file "${CUSTOM_OPP_ENV}"
  activate_bi_runtime
  export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES%%,*}"
  export SOC_VERSION=ascend910_9362
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 "${BI_PYTHON}" -m pytest --noconftest -q \
    "${VLLM_ASCEND_ROOT}/tests/ut/test_batch_invariant.py"
  "${BI_PYTHON}" - <<'PY'
import torch

import batch_invariant_ops  # noqa: F401
from vllm_ascend.batch_invariant import (
    HAS_ASCENDC_BATCH_INVARIANT,
    enable_batch_invariant_mode,
    reduce_sum,
)

assert HAS_ASCENDC_BATCH_INVARIANT, "batch-invariant backend unavailable"
x = torch.arange(24, dtype=torch.float32, device="npu").reshape(2, 3, 4)
expected = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).sum(dim=1)
enable_batch_invariant_mode()
actual = reduce_sum(x, dim=1)
actual_keepdim = reduce_sum(x, dim=1, keepdim=True)
repeated = torch.repeat_interleave(
    torch.tensor([10, 20, 30], device="npu"),
    torch.tensor([1, 2, 1], device="npu"),
)
torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
torch.testing.assert_close(actual_keepdim.cpu(), expected.unsqueeze(1), rtol=0, atol=0)
torch.testing.assert_close(
    repeated.cpu(), torch.tensor([10, 20, 20, 30]), rtol=0, atol=0
)
print("HAS_ASCENDC_BATCH_INVARIANT=True reduce/repeat smoke passed=True")
PY
  log "All patch, custom-op and NPU checks passed"
}

pid_group_alive() {
  [[ -n "${1:-}" ]] && kill -0 -- "-$1" 2>/dev/null
}

stop_current_service() {
  [[ -n "${CURRENT_PID}" ]] || return 0
  if pid_group_alive "${CURRENT_PID}"; then
    kill -TERM -- "-${CURRENT_PID}" 2>/dev/null || true
    local deadline=$((SECONDS + STOP_TIMEOUT_SECONDS))
    while pid_group_alive "${CURRENT_PID}" && (( SECONDS < deadline )); do
      sleep 2
    done
  fi
  if pid_group_alive "${CURRENT_PID}"; then
    log "TERM timed out; stopping the owned process group with KILL"
    kill -KILL -- "-${CURRENT_PID}" 2>/dev/null || true
  fi
  CURRENT_PID=""
}

wait_http() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    pid_group_alive "${CURRENT_PID}" || return 1
    curl -fsS --max-time 5 "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1 \
      && return 0
    sleep 5
  done
  return 1
}

start_service() {
  local log_path="$1"
  activate_bi_runtime
  export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
  export HCCL_IF_IP="${LOCAL_IP}"
  export VLLM_HOST_IP="${LOCAL_IP}"
  export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT}"
  export GLOO_SOCKET_IFNAME="${NIC_NAME}"
  export HCCL_SOCKET_IFNAME="${NIC_NAME}"
  export OMP_PROC_BIND=false
  export OMP_NUM_THREADS=10
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export HCCL_BUFFSIZE=2048
  export HCCL_OP_EXPANSION_MODE=AIV
  export TASK_QUEUE_ENABLE=1
  export SOC_VERSION=ascend910_9362
  export VLLM_ENGINE_READY_TIMEOUT_S=18000

  nohup setsid "${BI_PYTHON}" "${BASE_VENV}/bin/vllm" serve "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${API_PORT}" \
    --api-server-count 1 \
    --served-model-name dsv4-v023-native-dp2tp4-bi \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 16 \
    --data-parallel-size 2 \
    --data-parallel-rpc-port "${DATA_PARALLEL_RPC_PORT}" \
    --master-port "${MASTER_PORT}" \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --all2all-backend flashinfer_all2allv \
    --enforce-eager \
    --seed 1024 \
    --gpu-memory-utilization 0.90 \
    --tokenizer-mode deepseek_v4 \
    --no-enable-prefix-caching \
    --safetensors-load-strategy lazy \
    --quantization ascend \
    --block-size 128 >"${log_path}" 2>&1 &
  CURRENT_PID=$!
  if ! wait_http; then
    tail -n 200 "${log_path}" >&2 || true
    return 1
  fi
  local enable_count
  enable_count="$(grep -c 'Enabling batch-invariant mode' "${log_path}" || true)"
  (( enable_count >= 8 )) \
    || die "Expected at least 8 batch-invariant enable markers, got ${enable_count}"
  if grep -q 'backend unavailable' "${log_path}"; then
    die "Startup reported batch-invariant backend unavailable"
  fi
}

run_one_start() {
  local start_index="$1"
  local run_dir="$2"
  local log_path="${run_dir}/start${start_index}-service.log"
  local output_path="${run_dir}/start${start_index}-golden_results.json"
  local request_rc=0
  start_service "${log_path}"
  set +e
  "${BI_PYTHON}" "${GOLDEN_TOOL}" \
    --endpoint "http://127.0.0.1:${API_PORT}/v1/completions" \
    --model dsv4-v023-native-dp2tp4-bi \
    --prompt-source "${PROMPT_SOURCE}" \
    --output "${output_path}" \
    --rounds 3 \
    --metadata "node_label=${NODE_LABEL}" \
    --metadata "cold_start=${start_index}" \
    --metadata "mode=direct_prefill_batch_invariant"
  request_rc=$?
  set -e
  stop_current_service
  return "${request_rc}"
}

write_snapshot() {
  local output="$1"
  {
    printf 'node_label=%s\n' "${NODE_LABEL}"
    printf 'local_ip=%s\n' "${LOCAL_IP}"
    printf 'nic=%s\n' "${NIC_NAME}"
    printf 'cann_root=%s\n' "$(readlink -f "${CANN_ROOT}")"
    printf 'base_venv=%s\n' "${BASE_VENV}"
    printf 'bi_venv=%s\n' "${BI_VENV}"
    printf 'vllm_commit=%s\n' "$(git_in "${VLLM_ROOT}" rev-parse HEAD)"
    printf 'vllm_ascend_commit=%s\n' "$(git_in "${VLLM_ASCEND_ROOT}" rev-parse HEAD)"
    printf 'vllm_ascend_patch_sha256=%s\n' "$(patch_diff_sha256)"
    printf 'opp_sha256=%s\n' "$(sha256_file "${BATCH_INVARIANT_OPP_RUN}")"
    printf 'wheel_sha256=%s\n' "$(sha256_file "${BATCH_INVARIANT_WHEEL}")"
    printf 'driver=\n'
    npu-smi info 2>&1 || true
  } >"${output}"
}

collect_run_dir() {
  local run_dir="$1"
  local support_dir="${run_dir}/support"
  local archive="${run_dir}/${NODE_LABEL}-batch-invariant-support.tar.gz"
  mkdir -p "${support_dir}"
  cp "${run_dir}"/*-golden_results.json "${support_dir}/" 2>/dev/null || true
  cp "${run_dir}/cross-start-summary.json" "${support_dir}/" 2>/dev/null || true
  cp "${run_dir}/environment.txt" "${support_dir}/" 2>/dev/null || true
  cp "${run_dir}/npu-smi-after.txt" "${support_dir}/" 2>/dev/null || true
  local service_log
  for service_log in "${run_dir}"/*-service.log; do
    [[ -f "${service_log}" ]] || continue
    tail -c 262144 "${service_log}" \
      >"${support_dir}/$(basename "${service_log}").tail"
  done
  tar -czf "${archive}" -C "${support_dir}" .
  local size
  size="$(stat -c %s "${archive}")"
  (( size <= 2097152 )) || die "Support archive exceeds 2 MiB: ${archive}"
  log "Support archive: ${archive} (${size} bytes)"
}

run_twice_action() {
  validate_static_config
  validate_patch
  require_file "${BI_PYTHON}"
  require_file "${CUSTOM_OPP_ENV}"
  command -v curl >/dev/null 2>&1 || die "curl is required"
  command -v npu-smi >/dev/null 2>&1 || die "npu-smi is required"
  mkdir -p "${RESULT_ROOT}"
  local run_dir first_rc=0 second_rc=0 compare_rc=0
  run_dir="${RESULT_ROOT}/${NODE_LABEL}-$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${run_dir}"
  write_snapshot "${run_dir}/environment.txt"
  trap stop_current_service EXIT INT TERM
  run_one_start 1 "${run_dir}" || first_rc=$?
  run_one_start 2 "${run_dir}" || second_rc=$?
  if [[ -f "${run_dir}/start1-golden_results.json" \
    && -f "${run_dir}/start2-golden_results.json" ]]; then
    set +e
    "${BI_PYTHON}" "${COMPARE_TOOL}" \
      "${run_dir}/start1-golden_results.json" \
      "${run_dir}/start2-golden_results.json" \
      --output "${run_dir}/cross-start-summary.json"
    compare_rc=$?
    set -e
  else
    compare_rc=1
  fi
  npu-smi info >"${run_dir}/npu-smi-after.txt" 2>&1 || true
  collect_run_dir "${run_dir}"
  trap - EXIT INT TERM
  (( first_rc == 0 && second_rc == 0 && compare_rc == 0 )) \
    || die "Two-start validation failed; inspect ${run_dir}"
  log "Two cold starts are stable and cross-start exact; result=${run_dir}"
}

collect_action() {
  validate_static_config
  local run_dir="${3:-}"
  if [[ -z "${run_dir}" ]]; then
    run_dir="$(find "${RESULT_ROOT}" -mindepth 1 -maxdepth 1 -type d \
      -name "${NODE_LABEL}-*" | sort | tail -n 1)"
  fi
  [[ -n "${run_dir}" && -d "${run_dir}" ]] || die "No result directory found"
  collect_run_dir "${run_dir}"
}

print_config_action() {
  printf 'NODE_LABEL=%s\n' "${NODE_LABEL:-}"
  printf 'LOCAL_IP=%s\n' "${LOCAL_IP:-}"
  printf 'NIC_NAME=%s\n' "${NIC_NAME:-}"
  printf 'CANN_ROOT=%s\n' "${CANN_ROOT}"
  printf 'BASE_VENV=%s\n' "${BASE_VENV}"
  printf 'BI_VENV=%s\n' "${BI_VENV}"
  printf 'VLLM_ASCEND_ROOT=%s\n' "${VLLM_ASCEND_ROOT}"
  printf 'BATCH_INVARIANT_OPP_RUN=%s\n' "${BATCH_INVARIANT_OPP_RUN}"
  printf 'BATCH_INVARIANT_OPP_ROOT=%s\n' "${BATCH_INVARIANT_OPP_ROOT}"
  printf 'PROMPT_SOURCE=%s\n' "${PROMPT_SOURCE}"
  printf 'TOPOLOGY=DP2/TP4/EP8 direct Prefill, NPU %s\n' "${VISIBLE_DEVICES}"
}

case "${ACTION}" in
  print-config) print_config_action ;;
  download-opp) download_opp_action ;;
  apply-patch) apply_patch_action ;;
  install) install_action ;;
  check) check_action ;;
  run-twice) run_twice_action ;;
  collect) collect_action "$@" ;;
  *) usage; die "Unknown action: ${ACTION}" ;;
esac
