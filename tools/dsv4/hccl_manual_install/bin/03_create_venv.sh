#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

if [[ -x "${VENV_ROOT}/bin/python" ]]; then
  if is_true "${REUSE_VENV}"; then
    log "Reusing venv: ${VENV_ROOT}"
    "${VENV_ROOT}/bin/python" --version
    exit 0
  fi
  die "Venv already exists: ${VENV_ROOT}. Set REUSE_VENV=1 only after verifying it."
fi

if ! dir_is_empty "${VENV_ROOT}"; then
  die "VENV_ROOT exists and is not an empty venv target: ${VENV_ROOT}"
fi

ensure_dir "$(dirname "${VENV_ROOT}")"
log "Creating venv with ${PYTHON_BIN}: ${VENV_ROOT}"
"${PYTHON_BIN}" -m venv "${VENV_ROOT}"
"${VENV_ROOT}/bin/python" --version
