#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export AFD_CONNECTOR=P2pHcclAFDConnector
exec bash "${ROOT_DIR}/recipe/npu/CAMP2pAFDConnector/deepseek_v4/afd_attention.sh"
