#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/tools/dsv4/activate_runtime.sh"
set -u

EXPECTED_VLLM_COMMIT=568afb3a13806beb53bb2e6bd518269357b237c0
EXPECTED_ASCEND_COMMIT=80d8c194f7584b17fe08065ea99a130916f6b0e7
VLLM_ROOT=/mnt/workspace/code/vllm-afd-v0.26.0
ASCEND_ROOT=/mnt/workspace/code/vllm-ascend-afd-80d8c194f

[[ "$(git -C "${VLLM_ROOT}" rev-parse HEAD)" == "${EXPECTED_VLLM_COMMIT}" ]]
[[ "$(git -C "${ASCEND_ROOT}" rev-parse HEAD)" == "${EXPECTED_ASCEND_COMMIT}" ]]

python - <<'PY'
from importlib.metadata import version
import os

import torch
import torch_npu
import vllm
import vllm_ascend  # noqa: F401

from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded

assert "cann-9.1.0" not in repr(dict(os.environ))
assert torch.npu.is_available()
assert torch.npu.device_count() == 16
assert vllm.__version__.startswith("0.26.0")
assert version("torch-npu") == "2.10.0.post2"
assert version("vllm-ascend").endswith("g80d8c194f")
ensure_afd_ascend_ops_loaded()
print("DSV4_AFD_RUNTIME_OK")
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("vllm", vllm.__version__)
print("vllm_ascend", version("vllm-ascend"))
PY
