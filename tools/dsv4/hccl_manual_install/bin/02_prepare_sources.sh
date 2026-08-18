#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

prepare_target() {
  local target="$1"
  local label="$2"
  if ! dir_is_empty "${target}"; then
    if is_true "${REUSE_SOURCES}"; then
      log "Reusing ${label} source: ${target}"
      return 1
    fi
    die "${label} source directory is not empty: ${target}"
  fi
  ensure_dir "${target}"
  return 0
}

extract_source() {
  local archive="$1"
  local target="$2"
  local label="$3"
  require_file "${archive}"
  if prepare_target "${target}" "${label}"; then
    log "Extracting ${label}: ${archive}"
    tar -xzf "${archive}" -C "${target}"
  fi
}

clone_source() {
  local url="$1"
  local ref="$2"
  local target="$3"
  local label="$4"
  if prepare_target "${target}" "${label}"; then
    log "Cloning ${label}: ${url}"
    git clone "${url}" "${target}"
    git -C "${target}" checkout --detach "${ref}"
  fi
}

ensure_dir "${CODE_ROOT}"

if is_true "${USE_BUNDLED_SOURCES}"; then
  extract_source \
    "${BUNDLE_ROOT}/sources/vllm-release-v0.23.0.tar.gz" \
    "${VLLM_ROOT}" \
    "vLLM"
  extract_source \
    "${BUNDLE_ROOT}/sources/vllm-ascend-rfc-vllm-cann.tar.gz" \
    "${VLLM_ASCEND_ROOT}" \
    "vLLM-Ascend"
  extract_source \
    "${BUNDLE_ROOT}/sources/afd-plugin-mtp-m1-snapshot.tar.gz" \
    "${AFD_PLUGIN_ROOT}" \
    "afd-plugin"
else
  is_true "${OFFLINE}" \
    && die "USE_BUNDLED_SOURCES=0 is incompatible with OFFLINE=1"
  clone_source "${VLLM_GIT_URL}" "${VLLM_COMMIT}" "${VLLM_ROOT}" "vLLM"
  clone_source \
    "${VLLM_ASCEND_GIT_URL}" \
    "${VLLM_ASCEND_COMMIT}" \
    "${VLLM_ASCEND_ROOT}" \
    "vLLM-Ascend"
  git -C "${VLLM_ASCEND_ROOT}" submodule update --init --recursive
  [[ -n "${AFD_PLUGIN_REF}" ]] \
    || die "AFD_PLUGIN_REF is required when bundled sources are disabled"
  clone_source \
    "${AFD_PLUGIN_GIT_URL}" \
    "${AFD_PLUGIN_REF}" \
    "${AFD_PLUGIN_ROOT}" \
    "afd-plugin"
fi

require_file "${VLLM_ROOT}/setup.py"
require_file "${VLLM_ASCEND_ROOT}/setup.py"
require_file "${VLLM_ASCEND_ROOT}/csrc/third_party/catlass/CMakeLists.txt"
require_file "${AFD_PLUGIN_ROOT}/pyproject.toml"
require_file "${AFD_PLUGIN_ROOT}/afd_plugin/connectors/npu/p2p_hccl.py"
require_file \
  "${AFD_PLUGIN_ROOT}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_attention.sh"

printf '%s\n' "${VLLM_COMMIT}" >"${VLLM_ROOT}/.bundle-source-version"
printf '%s\n' "${VLLM_ASCEND_COMMIT}" >"${VLLM_ASCEND_ROOT}/.bundle-source-version"
printf '%s\n' "${AFD_SNAPSHOT_ID}" >"${AFD_PLUGIN_ROOT}/.bundle-source-version"

log "Sources are ready under ${CODE_ROOT}"
