#!/usr/bin/env bash
# Source this file before building or running the pinned DSV4 AFD stack.

DSV4_CANN_ROOT="${DSV4_CANN_ROOT:-/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1}"
DSV4_VLLM_VENV="${DSV4_VLLM_VENV:-/mnt/workspace/code/.venvs/afd-v026}"

if [[ ! -f "${DSV4_CANN_ROOT}/set_env.sh" ]]; then
  echo "Missing CANN set_env.sh: ${DSV4_CANN_ROOT}/set_env.sh" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ ! -x "${DSV4_VLLM_VENV}/bin/python" ]]; then
  echo "Missing DSV4 virtual environment: ${DSV4_VLLM_VENV}" >&2
  return 2 2>/dev/null || exit 2
fi

unset ASCEND_AICPU_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_TOOLKIT_HOME
unset ASCEND_CUSTOM_OPP_PATH ATB_HOME_PATH TOOLCHAIN_HOME VIRTUAL_ENV
export CMAKE_PREFIX_PATH=
export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib
export PYTHONPATH=
export PATH=/opt/buildtools/python-3.12.9/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

source "${DSV4_CANN_ROOT}/set_env.sh"
if [[ -f "${DSV4_CANN_ROOT}/nnal/atb/set_env.sh" ]]; then
  source "${DSV4_CANN_ROOT}/nnal/atb/set_env.sh"
fi

export VIRTUAL_ENV="${DSV4_VLLM_VENV}"
export PATH="${DSV4_VLLM_VENV}/bin:${PATH}"
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd

case "${PATH}:${LD_LIBRARY_PATH:-}:${PYTHONPATH:-}:${ASCEND_HOME_PATH:-}" in
  *cann-9.1.0*)
    echo "CANN 9.1.0 leaked into the pinned DSV4 runtime" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

unset DSV4_CANN_ROOT DSV4_VLLM_VENV
