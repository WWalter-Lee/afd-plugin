#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

curl -fsS --max-time 10 \
  "http://${API_HOST}:${ATTENTION_API_PORT}/health" >/dev/null \
  || die "Attention API is not healthy"

ensure_dir "${STATE_ROOT}"
response_file="${STATE_ROOT}/smoke-response-$(date +%Y%m%d_%H%M%S).json"
curl -fsS --max-time 300 \
  "http://${API_HOST}:${ATTENTION_API_PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4-afd",
    "prompt": "Please explain why deterministic validation matters.",
    "max_tokens": 32,
    "temperature": 0
  }' \
  --output "${response_file}"

grep -q '"choices"' "${response_file}" \
  || die "Smoke response has no choices: ${response_file}"
log "Smoke request passed: ${response_file}"
