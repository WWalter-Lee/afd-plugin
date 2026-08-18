#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

require_file "${VENV_ROOT}/bin/python"
require_file "${VLLM_ROOT}/requirements/common.txt"
require_file "${VLLM_ASCEND_ROOT}/requirements.txt"

python_bin="${VENV_ROOT}/bin/python"
pip_args=()

if is_true "${OFFLINE}"; then
  require_dir "${WHEELHOUSE}"
  pip_args+=(--no-index --find-links "${WHEELHOUSE}")
else
  if [[ -n "${PIP_INDEX_URL}" ]]; then
    pip_args+=(--index-url "${PIP_INDEX_URL}")
  fi
  if [[ -n "${PIP_EXTRA_INDEX_URL}" ]]; then
    pip_args+=(--extra-index-url "${PIP_EXTRA_INDEX_URL}")
  fi
  if [[ -n "${PIP_TRUSTED_HOST}" ]]; then
    pip_args+=(--trusted-host "${PIP_TRUSTED_HOST}")
  fi
fi

log "Installing Python build dependencies"
"${python_bin}" -m pip install "${pip_args[@]}" --upgrade \
  pip "setuptools>=64" "setuptools-scm>=8" wheel \
  "cmake>=3.26" ninja pybind11

log "Installing torch and Ascend Python runtime"
"${python_bin}" -m pip install "${pip_args[@]}" \
  torch==2.10.0 \
  torchvision==0.25.0 \
  torchaudio==2.10.0 \
  torch-npu==2.10.0.post2 \
  triton-ascend==3.2.1

log "Installing vLLM common requirements"
"${python_bin}" -m pip install "${pip_args[@]}" \
  -r "${VLLM_ROOT}/requirements/common.txt"

log "Installing vLLM-Ascend requirements"
"${python_bin}" -m pip install "${pip_args[@]}" \
  -r "${VLLM_ASCEND_ROOT}/requirements.txt"

# The target branch metadata still says torch-npu 2.10.0 and triton-ascend
# metadata pins numpy 1.26.4. The validated runtime intentionally restores
# these exact final versions after dependency resolution.
log "Restoring validated runtime pins"
"${python_bin}" -m pip install "${pip_args[@]}" \
  --upgrade --force-reinstall --no-deps \
  torch-npu==2.10.0.post2 \
  transformers==5.5.4 \
  numpy==2.2.6

ensure_dir "${STATE_ROOT}"
"${python_bin}" -m pip list --format=freeze \
  >"${STATE_ROOT}/python-packages-after-deps.txt"
log "Python dependencies installed"
