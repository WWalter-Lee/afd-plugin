#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/tools/dsv4/activate_v023_vllm_cann_runtime.sh"

MOONCAKE_VENV_SITE="$("${VIRTUAL_ENV}/bin/python" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ ! -d "$MOONCAKE_VENV_SITE" ]]; then
  echo "Target venv site-packages does not exist: $MOONCAKE_VENV_SITE" >&2
  exit 2
fi
case ":${PYTHONPATH:-}:" in
  *":${MOONCAKE_VENV_SITE}:"*) ;;
  *) export PYTHONPATH="${MOONCAKE_VENV_SITE}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

MOONCAKE_JEMALLOC="${MOONCAKE_JEMALLOC:-}"
if [[ -n "$MOONCAKE_JEMALLOC" && ! -f "$MOONCAKE_JEMALLOC" ]]; then
  echo "Configured MOONCAKE_JEMALLOC does not exist: $MOONCAKE_JEMALLOC" >&2
  exit 2
fi
if [[ -z "$MOONCAKE_JEMALLOC" ]]; then
  for candidate in \
    /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 \
    /usr/lib64/libjemalloc.so.2 \
    /usr/lib/libjemalloc.so.2 \
    /usr/local/lib64/libjemalloc.so.2 \
    /usr/local/lib/libjemalloc.so.2; do
    if [[ -f "$candidate" ]]; then
      MOONCAKE_JEMALLOC="$candidate"
      break
    fi
  done
fi
if [[ -z "$MOONCAKE_JEMALLOC" ]] && command -v ldconfig >/dev/null 2>&1; then
  MOONCAKE_JEMALLOC="$(
    ldconfig -p 2>/dev/null \
      | awk '$1 == "libjemalloc.so.2" {print $NF; exit}'
  )"
fi
if [[ -z "$MOONCAKE_JEMALLOC" || ! -f "$MOONCAKE_JEMALLOC" ]]; then
  echo "Mooncake PD requires libjemalloc.so.2; set MOONCAKE_JEMALLOC to its absolute path" >&2
  exit 2
fi
case ":${LD_PRELOAD:-}:" in
  *":${MOONCAKE_JEMALLOC}:"*) ;;
  *) export LD_PRELOAD="${MOONCAKE_JEMALLOC}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
esac

TORCH_NPU_DIR="$("${VIRTUAL_ENV}/bin/python" -c \
  'import importlib.util; s = importlib.util.find_spec("torch_npu"); print(next(iter(s.submodule_search_locations or []), "") if s else "")')"
ATB_PLUGIN="${TORCH_NPU_DIR}/lib/libop_plugin_atb.so"
if [[ -z "$ATB_PLUGIN" || ! -f "$ATB_PLUGIN" ]]; then
  echo "torch_npu ATB plugin was not found in the target Python runtime" >&2
  exit 2
fi
ATB_LDD_OUTPUT="$(ldd "$ATB_PLUGIN" 2>&1)" \
  || { printf '%s\n' "$ATB_LDD_OUTPUT" >&2; exit 2; }
if grep -q 'not found' <<<"$ATB_LDD_OUTPUT"; then
  grep 'not found' <<<"$ATB_LDD_OUTPUT" >&2
  echo "NNAL/ATB runtime check failed: $ATB_PLUGIN" >&2
  exit 2
fi
grep -q 'libatb.so =>' <<<"$ATB_LDD_OUTPUT" \
  || { echo "libop_plugin_atb.so does not resolve libatb.so" >&2; exit 2; }

MOONCAKE_DIR=$(
  python -c 'import importlib.util; spec = importlib.util.find_spec("mooncake"); print(next(iter(spec.submodule_search_locations or []), "") if spec else "")'
)
MOONCAKE_ENGINE=""
if [[ -n "$MOONCAKE_DIR" && -d "$MOONCAKE_DIR" ]]; then
  MOONCAKE_ENGINE="$(find "$MOONCAKE_DIR" -maxdepth 1 -name 'engine*.so' -print -quit)"
fi
if [[ -z "$MOONCAKE_ENGINE" || ! -f "$MOONCAKE_ENGINE" ]]; then
  echo "Mooncake Python extension was not found in the target venv" >&2
  exit 2
fi

MOONCAKE_LIBRARY_DIR="${MOONCAKE_LIBRARY_DIR:-}"
if [[ -n "$MOONCAKE_LIBRARY_DIR" ]]; then
  if [[ ! -f "${MOONCAKE_LIBRARY_DIR}/libtransfer_engine.so" \
    || ! -f "${MOONCAKE_LIBRARY_DIR}/ascend_transport.so" ]]; then
    echo "MOONCAKE_LIBRARY_DIR must contain libtransfer_engine.so and ascend_transport.so: ${MOONCAKE_LIBRARY_DIR}" >&2
    exit 2
  fi
else
  for candidate in "$MOONCAKE_DIR" /usr/local/lib /usr/local/lib64; do
    if [[ -f "${candidate}/libtransfer_engine.so" \
      && -f "${candidate}/ascend_transport.so" ]]; then
      MOONCAKE_LIBRARY_DIR="$candidate"
      break
    fi
  done
fi
if [[ -z "$MOONCAKE_LIBRARY_DIR" ]]; then
  echo "Mooncake runtime libraries were not found together; set MOONCAKE_LIBRARY_DIR" >&2
  exit 2
fi
case ":${LD_LIBRARY_PATH:-}:" in
  *":${MOONCAKE_LIBRARY_DIR}:"*) ;;
  *) export LD_LIBRARY_PATH="${MOONCAKE_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac

RUNTIME_INVALID=0
case "$MOONCAKE_ENGINE" in
  "$VIRTUAL_ENV"/*) ;;
  *)
    echo "Mooncake extension is outside the target venv: $MOONCAKE_ENGINE" >&2
    RUNTIME_INVALID=1
    ;;
esac

LDD_OUTPUT="$(ldd "$MOONCAKE_ENGINE")"
if grep -q 'not found' <<<"$LDD_OUTPUT"; then
  grep 'not found' <<<"$LDD_OUTPUT" >&2
  RUNTIME_INVALID=1
fi
if [[ -n "${DSV4_CANN_VERSION:-}" ]]; then
  while IFS= read -r linked_cann; do
    if [[ "$linked_cann" != "cann-${DSV4_CANN_VERSION}" ]]; then
      echo "CANN dependency mismatch: expected ${DSV4_CANN_VERSION}, found ${linked_cann}" >&2
      RUNTIME_INVALID=1
    fi
  done < <(grep -oE 'cann-[0-9]+(\.[0-9]+){1,2}' <<<"$LDD_OUTPUT" | sort -u)
fi
if [[ "$RUNTIME_INVALID" == "1" ]]; then
  exit 2
fi

python "${ROOT_DIR}/tools/dsv4/check_mooncake_contract.py"
echo "NNAL/ATB runtime check passed: plugin=$ATB_PLUGIN root=${DSV4_ATB_ROOT}"
echo "Mooncake runtime check passed: engine=$MOONCAKE_ENGINE libraries=$MOONCAKE_LIBRARY_DIR"

unset candidate linked_cann TORCH_NPU_DIR ATB_PLUGIN ATB_LDD_OUTPUT MOONCAKE_JEMALLOC
unset MOONCAKE_VENV_SITE MOONCAKE_DIR
unset MOONCAKE_ENGINE MOONCAKE_LIBRARY_DIR LDD_OUTPUT RUNTIME_INVALID
