#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/tools/dsv4/activate_runtime.sh"

export CCACHE_DIR="${CCACHE_DIR:-/mnt/workspace/.cache/ccache}"
export TMPDIR="${TMPDIR:-/mnt/workspace/.cache/tmp}"
export SOC_VERSION="${SOC_VERSION:-ascend910_9362}"
mkdir -p "${CCACHE_DIR}" "${TMPDIR}"

AFD_BUILD_ASCEND_OPS=1 python -m pip install \
  --no-build-isolation \
  --no-deps \
  --editable "${ROOT_DIR}"

bash "${ROOT_DIR}/tools/dsv4/check_runtime.sh"
