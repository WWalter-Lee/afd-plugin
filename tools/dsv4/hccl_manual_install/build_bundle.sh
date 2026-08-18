#!/usr/bin/env bash
set -euo pipefail

BUNDLE_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AFD_REPO_ROOT="$(cd "${BUNDLE_SOURCE_DIR}/../../.." && pwd)"
WORKSPACE_CODE_ROOT="$(dirname "${AFD_REPO_ROOT}")"
VLLM_SOURCE_ROOT="${VLLM_SOURCE_ROOT:-${WORKSPACE_CODE_ROOT}/vllm-release-v0.23.0}"
VLLM_ASCEND_SOURCE_ROOT="${VLLM_ASCEND_SOURCE_ROOT:-${WORKSPACE_CODE_ROOT}/vllm-ascend-rfc-vllm-cann}"
OUTPUT_DIR="${1:-/mnt/workspace/artifacts}"

VLLM_COMMIT=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
VLLM_ASCEND_COMMIT=3da28f9414583d2d0b672a8f06d1fae142404bda
AFD_RELEASE_REF="${AFD_RELEASE_REF:-dsv4-afd-v023-hccl-mtp-m1-v1}"
AFD_BASE_COMMIT="$(git -C "${AFD_REPO_ROOT}" rev-parse "${AFD_RELEASE_REF}^{commit}" 2>/dev/null)" \
  || { echo "afd-plugin release ref does not exist: ${AFD_RELEASE_REF}" >&2; exit 2; }
timestamp="$(date +%Y%m%d_%H%M%S)"
package_name="dsv4-afd-hccl-manual-install-${timestamp}"

[[ "$(git -C "${VLLM_SOURCE_ROOT}" rev-parse HEAD)" == "${VLLM_COMMIT}" ]] \
  || { echo "vLLM source commit mismatch" >&2; exit 2; }
[[ "$(git -C "${VLLM_ASCEND_SOURCE_ROOT}" rev-parse HEAD)" == "${VLLM_ASCEND_COMMIT}" ]] \
  || { echo "vLLM-Ascend source commit mismatch" >&2; exit 2; }
[[ -z "$(git -C "${VLLM_SOURCE_ROOT}" status --short)" ]] \
  || { echo "vLLM source worktree is dirty" >&2; exit 2; }
[[ -z "$(git -C "${VLLM_ASCEND_SOURCE_ROOT}" status --short)" ]] \
  || { echo "vLLM-Ascend source worktree is dirty" >&2; exit 2; }
[[ "$(git -C "${AFD_REPO_ROOT}" rev-parse HEAD)" == "${AFD_BASE_COMMIT}" ]] \
  || { echo "afd-plugin HEAD does not match ${AFD_RELEASE_REF}" >&2; exit 2; }
[[ -z "$(git -C "${AFD_REPO_ROOT}" status --short)" ]] \
  || { echo "afd-plugin source worktree is dirty" >&2; exit 2; }

temp_root="$(mktemp -d)"
cleanup() {
  rm -rf "${temp_root}"
}
trap cleanup EXIT

payload_root="${temp_root}/${package_name}"
mkdir -p "${payload_root}/manifest" "${payload_root}/sources"
cp -a "${BUNDLE_SOURCE_DIR}/." "${payload_root}/"
cp "${BUNDLE_SOURCE_DIR}/config.env.example" "${payload_root}/config.env"

cat >"${payload_root}/manifest/versions.env" <<EOF
VLLM_COMMIT="${VLLM_COMMIT}"
VLLM_ASCEND_COMMIT="${VLLM_ASCEND_COMMIT}"
AFD_BASE_COMMIT="${AFD_BASE_COMMIT}"
AFD_SNAPSHOT_ID="${AFD_RELEASE_REF}"
EOF

git -C "${VLLM_SOURCE_ROOT}" archive \
  --format=tar.gz \
  --output="${payload_root}/sources/vllm-release-v0.23.0.tar.gz" \
  "${VLLM_COMMIT}"

ascend_stage="${temp_root}/ascend-source"
mkdir -p "${ascend_stage}"
git -C "${VLLM_ASCEND_SOURCE_ROOT}" archive "${VLLM_ASCEND_COMMIT}" \
  | tar -xf - -C "${ascend_stage}"
mkdir -p "${ascend_stage}/csrc/third_party/catlass"
git -C "${VLLM_ASCEND_SOURCE_ROOT}/csrc/third_party/catlass" archive HEAD \
  | tar -xf - -C "${ascend_stage}/csrc/third_party/catlass"
mkdir -p "${ascend_stage}/csrc/third_party/catlass/3rdparty/googletest"
git -C "${VLLM_ASCEND_SOURCE_ROOT}/csrc/third_party/catlass/3rdparty/googletest" \
  archive HEAD \
  | tar -xf - -C \
    "${ascend_stage}/csrc/third_party/catlass/3rdparty/googletest"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -czf "${payload_root}/sources/vllm-ascend-rfc-vllm-cann.tar.gz" \
  -C "${ascend_stage}" .

afd_stage="${temp_root}/afd-source"
mkdir -p "${afd_stage}"
git -C "${AFD_REPO_ROOT}" archive "${AFD_RELEASE_REF}" \
  | tar -xf - -C "${afd_stage}"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -czf "${payload_root}/sources/afd-plugin-mtp-m1-snapshot.tar.gz" \
  -C "${afd_stage}" .

if [[ -n "${INCLUDE_WHEELHOUSE:-}" ]]; then
  [[ -d "${INCLUDE_WHEELHOUSE}" ]] \
    || { echo "INCLUDE_WHEELHOUSE is not a directory" >&2; exit 2; }
  cp -a "${INCLUDE_WHEELHOUSE}" "${payload_root}/wheelhouse"
fi

(
  cd "${payload_root}"
  find . -type f \
    ! -path './manifest/SHA256SUMS' \
    ! -path './config.env' \
    -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum \
    >manifest/SHA256SUMS
)

mkdir -p "${OUTPUT_DIR}"
archive_path="${OUTPUT_DIR}/${package_name}.tar.gz"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -czf "${archive_path}" \
  -C "${temp_root}" "${package_name}"
(
  cd "${OUTPUT_DIR}"
  sha256sum "$(basename "${archive_path}")" \
    >"$(basename "${archive_path}").sha256"
)

printf '%s\n' "${archive_path}"
printf '%s\n' "${archive_path}.sha256"
