#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/tools/dsv4/activate_v023_vllm_cann_runtime.sh"

MOONCAKE_JEMALLOC="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
if [[ ! -f "$MOONCAKE_JEMALLOC" ]]; then
  echo "Mooncake PD requires jemalloc: $MOONCAKE_JEMALLOC" >&2
  exit 2
fi
case ":${LD_PRELOAD:-}:" in
  *":${MOONCAKE_JEMALLOC}:"*) ;;
  *) export LD_PRELOAD="${MOONCAKE_JEMALLOC}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
esac

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
if grep -q 'cann-9.1.0' <<<"$LDD_OUTPUT"; then
  echo "CANN 9.1.0 leaked into Mooncake runtime dependencies" >&2
  RUNTIME_INVALID=1
fi
if [[ "$RUNTIME_INVALID" == "1" ]]; then
  exit 2
fi

python "${ROOT_DIR}/tools/dsv4/check_mooncake_contract.py"
echo "Mooncake runtime check passed: $MOONCAKE_ENGINE"

unset MOONCAKE_JEMALLOC MOONCAKE_DIR MOONCAKE_ENGINE LDD_OUTPUT RUNTIME_INVALID
