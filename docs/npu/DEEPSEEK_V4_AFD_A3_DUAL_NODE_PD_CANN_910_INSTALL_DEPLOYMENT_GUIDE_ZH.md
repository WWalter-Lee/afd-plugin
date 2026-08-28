# DeepSeek-V4 AFD A3 双机 PD 分离安装部署指南（CANN 9.1.0）

## 0. 文档状态和适用范围

本文给出 CANN 9.1.0 环境中，两台 Atlas A3 部署 DeepSeek-V4 Mooncake PD +
Attention/FFN 分离服务的手工安装、启停和验收流程。

固定拓扑为：

- A3-P：NPU 0-7，Prefill DP2 x TP4；
- A3-D：NPU 0-7 运行 Decode Attention DP8 x TP1，NPU 8-15 运行 Decode
  FFN DP8 x TP1；
- Prefill 和 Decode Attention 之间使用 Mooncake Ascend Direct 传输 KV cache；
- Decode Attention 和 FFN 之间使用 afd-plugin 的 `P2pHcclAFDConnector`；
- 客户端统一访问 PD Proxy。

本文继续使用上游 vLLM-Ascend 的实际 Git 分支 `rfc/vllm_cann`。文档、命令和
目录中不使用其他 vLLM-Ascend fork。用户口头所说的 `rfc/vllm-cann` 对应仓库中
的实际分支名是 `rfc/vllm_cann`。

截至 2026-08-27，下面的 vLLM 0.23 / vLLM-Ascend / afd-plugin 组合已在 CANN
9.0.1 环境形成安装和 Mooncake 本机门禁；CANN 9.1.0 下尚未完成真实双机实模
F0。因此本文是 CANN 9.1.0 的安装与强制验收流程，只有完成第 12 节全部门禁后，
才能声明 CANN 9.1.0 双机 PD 基线通过。

本文不执行旧交付包的 `install_all.sh`，也不使用与本基线不兼容的 AFD UBatch
补丁或部署脚本。

## 1. 固定版本基线

两台 A3 必须使用相同版本、源码提交和二进制产物。

| 组件 | 固定值 |
| --- | --- |
| 架构 | Linux aarch64，Python 3.12；参考环境为 Python 3.12.9 |
| 硬件 | Atlas A3；参考 SoC 为 `ascend910_9362` |
| CANN / NNAL | 9.1.0；NNAL/ATB 必须与 CANN 和实际 A3 SKU 匹配 |
| torch | 2.10.0 |
| torch-npu | 2.10.0.post2 |
| torchvision / torchaudio | 0.25.0 / 2.10.0 |
| triton-ascend | 3.2.1 |
| transformers | 5.5.4 |
| NumPy | 2.2.6 |
| vLLM | `vllm-project/vllm`，分支 `releases/v0.23.0`，commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `vllm-project/vllm-ascend`，分支 `rfc/vllm_cann`，commit `3da28f9414583d2d0b672a8f06d1fae142404bda` |
| afd-plugin | `feat/dsv4-afd-mooncake-pd`，commit `49bb4a1dda5f7a59dcfbb45ea36d3ad1b2b89193` |
| Mooncake | 上游 v0.3.9；正式基线必须使用 CANN 9.1.0 环境构建的 Ascend Direct wheel |

不要使用浮动分支头代替固定 commit。分支用于确认代码来源，40 位 commit 用于
保证两端安装可复现。

## 2. 双机拓扑和端口

```text
client
  |
  v
PD Proxy :9000
  |------------------------------------|
  v                                    v
A3-P                                   A3-D
Prefill                                Decode Attention
NPU 0-7                                NPU 0-7
DP2 x TP4                              DP8 x TP1
:8100                                  :8910
  |                                    |
  |        Mooncake KV transfer        | local HCCL P2P
  |----------------------------------->|
                                       v
                                       Decode FFN
                                       NPU 8-15
                                       DP8 x TP1 / EP8
                                       no HTTP endpoint
```

| 用途 | A3-P | A3-D | 说明 |
| --- | --- | --- | --- |
| 业务 HTTP | 8100 | 8910 | Proxy 的 Prefill/Decode 后端 |
| FFN 进程参数 | 无 | 8911 | 不是 HTTP health 端口 |
| PD Proxy | 9000 | 可选 | 推荐运行在 A3-P |
| Mooncake KV | 30000 | 30100 | producer / consumer |
| AFD | 无 | 29761 | Attention/FFN 本机连接 |
| HCCL base | 50000 | 51000、52000 | Prefill、Attention、FFN |
| Mooncake RPC | 15000-17000 | 15000-17000 | 两机双向放通 |
| Ascend Transport OOB | 10000-10015 | 10000-10015 | 物理 NPU ID 对应端口，冲突时可能顺延 |

Decode FFN 不提供 HTTP 服务。不要探测 8911，也不要把它加入 Proxy。

## 3. 目录规划

```bash
export DEPLOY_ROOT=/data/dsv4
export CODE_ROOT="${DEPLOY_ROOT}/code"
export VENV_ROOT="${CODE_ROOT}/.venvs/afd-v023-vllm-cann"
export PACKAGE_ROOT="${DEPLOY_ROOT}/packages"
export MODEL_PATH="${DEPLOY_ROOT}/models/DeepSeek-V4-Flash-w8a8-mtp"
export RUN_ROOT="${DEPLOY_ROOT}/run/dsv4-afd-mooncake-pd"

mkdir -p "${CODE_ROOT}" "${CODE_ROOT}/.venvs" "${PACKAGE_ROOT}" \
  "${DEPLOY_ROOT}/models" "${RUN_ROOT}"
```

模型约 280 GiB。两端模型目录必须完整且一致，至少包含 70 个权重分片、index、
config 和 tokenizer 文件。

## 4. 系统、Driver 和 CANN 9.1.0

### 4.1 系统依赖

Ubuntu/Debian 示例：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build git git-lfs curl jq pkg-config patchelf \
  python3.12 python3.12-dev python3.12-venv \
  iproute2 netcat-openbsd libjemalloc2 \
  libgoogle-glog-dev libgflags-dev libjsoncpp-dev libyaml-cpp-dev \
  libibverbs-dev libnuma-dev libunwind-dev libboost-all-dev \
  libssl-dev libcurl4-openssl-dev libhiredis-dev \
  libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
  mpich libmpich-dev
```

openEuler/RHEL 系安装提供相同命令和共享库的 RPM，至少确认：

```bash
command -v gcc g++ cmake git curl ip ss
ldconfig -p | grep -E 'libjemalloc|libglog|libjsoncpp|libyaml-cpp'
```

Mooncake Ascend Direct 的 MPI 环境应统一，不要混装互相冲突的 MPICH 和 OpenMPI。

### 4.2 CANN 9.1.0 门禁

先安装目标 A3 对应的 Driver/Firmware，再安装同一套 CANN 9.1.0 Toolkit、
Kernel/Ops、HCCL 和匹配的 NNAL/ATB。安装命令和介质文件名以目标环境的官方或
内部发布包为准。

```bash
export CANN_ROOT=/opt/Ascend/cann-9.1.0
export CANN_VERSION=9.1.0
export ATB_ROOT=/opt/Ascend/nnal/atb-9.1.0

test -r "${CANN_ROOT}/set_env.sh"
test -r "${ATB_ROOT}/set_env.sh"
source "${CANN_ROOT}/set_env.sh"
source "${ATB_ROOT}/set_env.sh"

"${CANN_ROOT}/query_pkg_version.sh" | sed -n '1,40p'
npu-smi info
test -r /etc/hccn.conf
```

两端 CANN 核心包必须均为 9.1.0，并各能看到 16 张健康 NPU。参考构建值为
`SOC_VERSION=ascend910_9362`；其他 A3 SKU 必须使用实际 SoC。

不要在同一个 shell 中先后 source CANN 9.0.x 和 9.1.0。容器还必须使用 host
network、映射全部 NPU/Driver 设备和 `/etc/hccn.conf`，并提供至少 512 GiB
`/dev/shm`。

## 5. 获取固定源码和交付 overlay

### 5.1 下载三个源码仓库

两端执行。目录必须是新目录，已有本地改动时不要覆盖：

```bash
export VLLM_BRANCH=releases/v0.23.0
export VLLM_COMMIT=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
export VLLM_ASCEND_BRANCH=rfc/vllm_cann
export VLLM_ASCEND_COMMIT=3da28f9414583d2d0b672a8f06d1fae142404bda
export AFD_PD_BRANCH=feat/dsv4-afd-mooncake-pd
export AFD_PD_COMMIT=49bb4a1dda5f7a59dcfbb45ea36d3ad1b2b89193

git clone --recursive --single-branch --branch "${VLLM_BRANCH}" \
  https://github.com/vllm-project/vllm.git \
  "${CODE_ROOT}/vllm-release-v0.23.0"

git clone --recursive --single-branch --branch "${VLLM_ASCEND_BRANCH}" \
  https://github.com/vllm-project/vllm-ascend.git \
  "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann"

git clone --recursive --single-branch --branch "${AFD_PD_BRANCH}" \
  https://github.com/wenhow/afd-plugin.git \
  "${CODE_ROOT}/afd-plugin"
```

立即检查分支和提交：

```bash
test "$(git -C "${CODE_ROOT}/vllm-release-v0.23.0" branch --show-current)" = \
  "${VLLM_BRANCH}"
test "$(git -C "${CODE_ROOT}/vllm-release-v0.23.0" rev-parse HEAD)" = \
  "${VLLM_COMMIT}"

test "$(git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" branch --show-current)" = \
  "${VLLM_ASCEND_BRANCH}"
test "$(git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" rev-parse HEAD)" = \
  "${VLLM_ASCEND_COMMIT}"

test "$(git -C "${CODE_ROOT}/afd-plugin" branch --show-current)" = \
  "${AFD_PD_BRANCH}"
test "$(git -C "${CODE_ROOT}/afd-plugin" rev-parse HEAD)" = \
  "${AFD_PD_COMMIT}"
```

如果将来远端分支头移动，上述 commit 检查会失败。此时应重新做兼容和双机回归，
不能只修改文档里的期望 commit。

### 5.2 应随指南交付的 overlay

当前 Mooncake PD 管理脚本以小型 overlay 交付：

```text
afd-plugin-dsv4-mooncake-pd-update10-20260825.tar.gz
SHA256: ce1274f258abe7ea462766d2c004798fe99ff990b1de91e887301ab7710bd86a
```

当前工作区文件位于：

```text
/mnt/workspace/afd-plugin-dsv4-mooncake-pd-update10-20260825.tar.gz
/mnt/workspace/afd-plugin-dsv4-mooncake-pd-update10-20260825.tar.gz.sha256
```

在两端放到 `${PACKAGE_ROOT}` 后执行：

```bash
cd "${PACKAGE_ROOT}"
sha256sum -c afd-plugin-dsv4-mooncake-pd-update10-20260825.tar.gz.sha256
tar -xzf afd-plugin-dsv4-mooncake-pd-update10-20260825.tar.gz \
  -C "${CODE_ROOT}"
```

overlay 只修改文档、运行检查和管理工具，不修改 recipe、connector 或模型运行
代码。`pd.sh` 会拒绝 overlay 白名单之外的 afd-plugin 工作树改动。

## 6. 创建 Python 3.12 环境并安装运行栈

### 6.1 创建 venv

在只 source CANN 9.1.0 和匹配 ATB 的干净 shell 中执行：

```bash
command -v python3.12
python3.12 --version
if ldd "$(command -v python3.12)" | grep -q 'not found'; then
  echo 'Python 3.12 has unresolved shared libraries' >&2
  exit 1
fi

python3.12 -m venv "${VENV_ROOT}"
source "${VENV_ROOT}/bin/activate"

python -m pip install --upgrade \
  pip 'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8' \
  'setuptools-rust>=1.9.0' 'packaging>=24.2' wheel jinja2 \
  'cmake>=3.26.1' ninja pybind11

test "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = \
  3.12
```

### 6.2 安装 torch、torch-npu 和 Python 依赖

华为源可以替换为包含相同 aarch64 wheel 的内部镜像：

```bash
python -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  torch-npu==2.10.0.post2 triton-ascend==3.2.1

python -m pip install \
  -r "${CODE_ROOT}/vllm-release-v0.23.0/requirements/common.txt"

python -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  -r "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann/requirements.txt"

python -m pip install --upgrade --force-reinstall --no-deps \
  torch-npu==2.10.0.post2 transformers==5.5.4 numpy==2.2.6
```

该固定栈存在两个已知 metadata 偏差：vLLM-Ascend requirements 写的是
`torch-npu==2.10.0`，triton-ascend 3.2.1 metadata 要求 NumPy 1.26.4；固定运行
基线分别使用 2.10.0.post2 和 2.2.6。因此 `pip check` 可能报告这两项，最终以
本节固定版本和第 9 节实际运行门禁为准。

### 6.3 安装 vLLM、vLLM-Ascend 和 afd-plugin

安装顺序固定为 vLLM、vLLM-Ascend、afd-plugin：

```bash
cd "${CODE_ROOT}/vllm-release-v0.23.0"
VLLM_TARGET_DEVICE=empty \
  "${VENV_ROOT}/bin/python" -m pip install \
  --no-build-isolation --no-deps --editable .

export SOC_VERSION=ascend910_9362
export ASCEND_HOME_PATH="${CANN_ROOT}"
cd "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann"
"${VENV_ROOT}/bin/python" -m pip install \
  -v --no-build-isolation --no-deps --editable .

cd "${CODE_ROOT}/afd-plugin"
AFD_BUILD_ASCEND_OPS=0 \
  "${VENV_ROOT}/bin/python" -m pip install \
  -v --no-build-isolation --no-deps --editable .
```

`P2pHcclAFDConnector` 不需要 afd-plugin 的 CAMP2P 自定义算子，因此固定
`AFD_BUILD_ASCEND_OPS=0`。构建 vLLM 和 vLLM-Ascend 时必须保留
`--no-build-isolation`，避免构建隔离拉入 torch 2.11.0。

```bash
export VLLM_ASCEND_OPS_ENV="${CODE_ROOT}/vllm-ascend-rfc-vllm-cann/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
test -f "${VLLM_ASCEND_OPS_ENV}"
```

## 7. 构建和安装 Mooncake 0.3.9

### 7.1 CANN 9.1.0 产物要求

推荐在 A3-P 的 CANN 9.1.0 环境构建一次 Ascend Direct wheel，完成本机门禁后，
把完全相同的 wheel 分发到 A3-D。

已有的
`mooncake_transfer_engine-0.3.9-cp312-cp312-manylinux_2_39_aarch64.whl`
SHA256 为 `0f9964801b24fd683d6016e1196cc0606fc87b0285b45d89c433650b9477ca12`，
但它是在 CANN 9.0.1 上构建的，只能作为迁移候选，不能直接写入 CANN 9.1.0
正式基线。

### 7.2 从源码构建参考流程

```bash
git clone --recursive https://github.com/kvcache-ai/Mooncake.git \
  "${CODE_ROOT}/Mooncake"
git -C "${CODE_ROOT}/Mooncake" checkout --detach \
  a00f75742469d6cc408bc5393b73c501e92dc74a
git -C "${CODE_ROOT}/Mooncake" submodule update --init --recursive

git clone https://github.com/alibaba/yalantinglibs.git \
  "${CODE_ROOT}/yalantinglibs"
git -C "${CODE_ROOT}/yalantinglibs" checkout 0.5.5

cmake -S "${CODE_ROOT}/yalantinglibs" \
  -B "${CODE_ROOT}/yalantinglibs/build" \
  -DCMAKE_INSTALL_PREFIX="${CODE_ROOT}/Mooncake/thirdparties/install" \
  -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build "${CODE_ROOT}/yalantinglibs/build" -j"$(nproc)"
cmake --install "${CODE_ROOT}/yalantinglibs/build"
```

Mooncake v0.3.9 需要补充 `mooncake_common` 的 yalantinglibs 链接：

```bash
cd "${CODE_ROOT}/Mooncake"
git apply <<'PATCH'
diff --git a/mooncake-common/src/CMakeLists.txt b/mooncake-common/src/CMakeLists.txt
index 63bb25d..b5dc0de 100644
--- a/mooncake-common/src/CMakeLists.txt
+++ b/mooncake-common/src/CMakeLists.txt
@@ -10,4 +10,5 @@
 target_link_libraries(mooncake_common PUBLIC
     yaml-cpp
     jsoncpp
+    yalantinglibs::yalantinglibs
 )
PATCH
git diff --check
```

在目标 venv 和 CANN 9.1.0 环境编译：

```bash
cd "${CODE_ROOT}/Mooncake"
export ASCEND_HOME_PATH="${CANN_ROOT}"
export CMAKE_PREFIX_PATH="${CODE_ROOT}/Mooncake/thirdparties/install"
export Python3_EXECUTABLE="${VENV_ROOT}/bin/python"

cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
  -DPython3_EXECUTABLE="${Python3_EXECUTABLE}" \
  -DUSE_ASCEND_DIRECT=ON \
  -DUSE_ASCEND=OFF \
  -DWITH_STORE=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF
cmake --build build -j"$(nproc)"

PYTHON_VERSION=3.12 BUILD_WITH_EP=0 \
  bash scripts/build_wheel.sh 3.12 dist
cp mooncake-wheel/dist/mooncake_transfer_engine-*.whl "${PACKAGE_ROOT}/"
sha256sum "${PACKAGE_ROOT}"/mooncake_transfer_engine-*.whl
```

记录新的 wheel SHA256，并在 A3-D 使用同一文件。构建日志中若出现 CANN 9.0.x
路径，删除该次构建目录后在干净的 CANN 9.1.0 shell 重建。

## 8. 配置统一 PD 管理脚本

### 8.1 创建三个角色配置

overlay 应提供：

```text
${CODE_ROOT}/afd-plugin/tools/dsv4/mooncake_pd_manual/pd.sh
${CODE_ROOT}/afd-plugin/tools/dsv4/mooncake_pd_manual/config.env.example
```

```bash
export PD_TOOL="${CODE_ROOT}/afd-plugin/tools/dsv4/mooncake_pd_manual/pd.sh"
mkdir -p "${DEPLOY_ROOT}/config"

bash "${PD_TOOL}" init "${DEPLOY_ROOT}/config/pd-prefill.env"
bash "${PD_TOOL}" init "${DEPLOY_ROOT}/config/pd-decode.env"
bash "${PD_TOOL}" init "${DEPLOY_ROOT}/config/pd-proxy.env"
```

三个配置都必须修改以下公共项：

```bash
CODE_ROOT="/data/dsv4/code"
VENV_ROOT="${CODE_ROOT}/.venvs/afd-v023-vllm-cann"
VLLM_ROOT="${CODE_ROOT}/vllm-release-v0.23.0"
VLLM_ASCEND_ROOT="${CODE_ROOT}/vllm-ascend-rfc-vllm-cann"
AFD_PLUGIN_ROOT="${CODE_ROOT}/afd-plugin"

CANN_ROOT="/opt/Ascend/cann-9.1.0"
CANN_VERSION="9.1.0"
ATB_ROOT="/opt/Ascend/nnal/atb-9.1.0"

VLLM_COMMIT="0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"
VLLM_ASCEND_COMMIT="3da28f9414583d2d0b672a8f06d1fae142404bda"
AFD_PD_COMMIT="49bb4a1dda5f7a59dcfbb45ea36d3ad1b2b89193"

MODEL_PATH="/data/dsv4/models/DeepSeek-V4-Flash-w8a8-mtp"
GOLDEN_PATH="/data/dsv4/validation/CHANGE_ME_GOLDEN_RESULTS.json"
PREFILL_IP="CHANGE_ME_A3_P_IP"
DECODE_IP="CHANGE_ME_A3_D_IP"
NIC_NAME="eth0"

MOONCAKE_INSTALL_MODE="wheel"
MOONCAKE_VERSION="0.3.9"
MOONCAKE_WHEEL="/data/dsv4/packages/CHANGE_ME_CANN910_WHEEL.whl"
MOONCAKE_WHEEL_SHA256="CHANGE_ME_CANN910_WHEEL_64_HEX_SHA256"

RUN_ROOT="/data/dsv4/run/dsv4-afd-mooncake-pd"
RUN_LOCAL_ROUNDTRIP="1"
ALLOW_NPU_PROCESSES="0"
```

角色项分别设置：

```text
pd-prefill.env: NODE_ROLE="prefill"
pd-decode.env:  NODE_ROLE="decode"
pd-proxy.env:   NODE_ROLE="proxy"
```

不要沿用 `config.env.example` 中的 CANN 9.0.1 默认值或旧 Mooncake wheel SHA。
`GOLDEN_PATH` 必须指向同一 vLLM 0.23 目标栈、同一模型与采样参数生成的非 AFD
token-ID 基线，不要拿其他软件栈的 token 文件代替。三个配置的 IP、源码路径、
commit、Mooncake wheel、golden 文件和网络端口必须一致；执行前替换全部
`CHANGE_ME_*` 值。

### 8.2 分支和配置门禁

```bash
test "$(git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" branch --show-current)" = \
  rfc/vllm_cann
test "$(git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" rev-parse HEAD)" = \
  3da28f9414583d2d0b672a8f06d1fae142404bda

bash "${PD_TOOL}" print-config "${DEPLOY_ROOT}/config/pd-prefill.env"
bash "${PD_TOOL}" print-config "${DEPLOY_ROOT}/config/pd-decode.env"
```

## 9. 安装和本机预检

### 9.1 安装 Mooncake 和 afd-plugin

A3-P：

```bash
bash "${PD_TOOL}" install "${DEPLOY_ROOT}/config/pd-prefill.env"
bash "${PD_TOOL}" check "${DEPLOY_ROOT}/config/pd-prefill.env"
```

A3-D：

```bash
bash "${PD_TOOL}" install "${DEPLOY_ROOT}/config/pd-decode.env"
bash "${PD_TOOL}" check "${DEPLOY_ROOT}/config/pd-decode.env"
```

`check` 必须完成 CANN 9.1.0、ATB/Mooncake 动态库、Python metadata、16 张 NPU、
网卡/IP/端口、2 MiB 两轮 NPU round-trip 和 Mooncake 共享库指纹检查。A3-P 和
A3-D 的 `mooncake-libraries.sha256` 必须相同。

### 9.2 Python 和 connector 门禁

```bash
cd "${CODE_ROOT}/afd-plugin"
export DSV4_CANN_ROOT="${CANN_ROOT}"
export DSV4_ATB_ROOT="${ATB_ROOT}"
export DSV4_RUNTIME_VENV="${VENV_ROOT}"
export DSV4_VLLM_ROOT="${CODE_ROOT}/vllm-release-v0.23.0"
export DSV4_VLLM_ASCEND_ROOT="${CODE_ROOT}/vllm-ascend-rfc-vllm-cann"
source tools/dsv4/activate_v023_vllm_cann_runtime.sh
```

```bash
python - <<'PY'
from importlib.metadata import version

import torch
import torch_npu
import vllm
import vllm_ascend  # noqa: F401
from afd_plugin.connectors.npu.p2p_hccl import P2pHcclAFDConnector

assert torch.__version__.startswith('2.10.0')
assert version('torch-npu') == '2.10.0.post2'
assert vllm.__version__.startswith('0.23.0')
assert version('vllm-ascend').endswith('g3da28f941')
assert version('transformers') == '5.5.4'
assert version('numpy') == '2.2.6'
assert P2pHcclAFDConnector.__name__ == 'P2pHcclAFDConnector'
assert torch.npu.is_available()
assert torch.npu.device_count() == 16
print('DSV4_AFD_CANN910_RUNTIME_OK')
PY

vllm serve --help=all > "${RUN_ROOT}/vllm-help.txt" 2>&1
for flag in \
  --kv-transfer-config --data-parallel-size --tensor-parallel-size \
  --enable-expert-parallel --additional-config; do
  grep -Fq -- "${flag}" "${RUN_ROOT}/vllm-help.txt" || exit 1
done
```

任一门禁失败都不要继续启动。

## 10. 网络和启动前检查

两端分别填写实际网卡和 IP：

```bash
# A3-P
export NIC_NAME=eth0
export LOCAL_IP=192.168.10.11
export PEER_IP=192.168.10.12

# A3-D
export NIC_NAME=eth0
export LOCAL_IP=192.168.10.12
export PEER_IP=192.168.10.11
```

```bash
ip -4 addr show dev "${NIC_NAME}"
ping -c 3 "${PEER_IP}"
npu-smi info
df -h /dev/shm
ss -ltnp | grep -E ':(8100|8910|8911|9000|29761|30000|30100|50000|51000|52000)([[:space:]]|$)' || true
```

启动前上述端口应未被占用。防火墙还必须允许两台机器双向访问 15000-17000、
10000-10015 和 HCCL 使用的端口范围。

## 11. 启动、Readiness 和 Proxy

启动顺序固定为 Prefill、Decode、Proxy。

A3-P：

```bash
bash "${PD_TOOL}" start "${DEPLOY_ROOT}/config/pd-prefill.env"
bash "${PD_TOOL}" status "${DEPLOY_ROOT}/config/pd-prefill.env"
```

A3-D：

```bash
bash "${PD_TOOL}" start "${DEPLOY_ROOT}/config/pd-decode.env"
bash "${PD_TOOL}" status "${DEPLOY_ROOT}/config/pd-decode.env"
```

Decode 启动器会紧邻拉起 FFN 和 Attention。不要等待 FFN HTTP ready；FFN 没有
HTTP 服务。Decode 状态必须确认 8 个 FFN connector loop 全部 ready，Attention
的 8910 `/health` 通过。

Proxy 所在节点：

```bash
bash "${PD_TOOL}" check "${DEPLOY_ROOT}/config/pd-proxy.env"
bash "${PD_TOOL}" start "${DEPLOY_ROOT}/config/pd-proxy.env"
bash "${PD_TOOL}" status "${DEPLOY_ROOT}/config/pd-proxy.env"
curl -fsS --max-time 10 http://127.0.0.1:9000/healthcheck
```

## 12. 跨机 PD 功能验收

在 Proxy 节点执行：

```bash
bash "${PD_TOOL}" validate "${DEPLOY_ROOT}/config/pd-proxy.env"
```

该动作应完成 Proxy smoke、固定 10 条 prompt x 3 轮、30/30 token IDs exact、
batch 1/8/32 和请求取消恢复。

然后在 A3-D 检查真实跨机 KV transfer：

```bash
grep -En 'KV cache transfer for request .* took .* remote_session_id' \
  "${RUN_ROOT}/logs/decode"/*.log
```

`remote_session_id` 必须包含 A3-P 的可达业务 IP，不能是 `127.0.0.1` 或
`0.0.0.0`。没有该证据时，Proxy 请求成功也不能证明跨机 PD 生效。

CANN 9.1.0 正式验收要求：

1. 两端 CANN、Python 包、三个源码 commit、分支和 Mooncake wheel SHA 一致；
2. 两端 Mooncake 本机 NPU round-trip 均通过；
3. Prefill、Attention、8 个 FFN loop 和 Proxy 全部 ready；
4. smoke、30/30 golden、batch 1/8/32 和取消恢复通过；
5. 正常停止后完整二次启动，并再次产生真实跨机 KV transfer；
6. 两端无 fatal 日志、遗留端口、遗留推理进程和 NPU 占用。

## 13. 停止、二次启动和输出件

停止顺序固定为 Proxy、Decode、Prefill：

```bash
bash "${PD_TOOL}" stop "${DEPLOY_ROOT}/config/pd-proxy.env"
bash "${PD_TOOL}" stop "${DEPLOY_ROOT}/config/pd-decode.env"
bash "${PD_TOOL}" stop "${DEPLOY_ROOT}/config/pd-prefill.env"
```

不要使用全局 `pkill`。停止后检查：

```bash
npu-smi info
ss -ltnp | grep -E ':(8100|8910|8911|9000|29761|30000|30100)([[:space:]]|$)' || true
```

重新执行第 11、12 节，至少完成一次 smoke、一次真实 KV transfer 和正常停止。

三个角色分别收集小输出件：

```bash
bash "${PD_TOOL}" collect "${DEPLOY_ROOT}/config/pd-prefill.env"
bash "${PD_TOOL}" collect "${DEPLOY_ROOT}/config/pd-decode.env"
bash "${PD_TOOL}" collect "${DEPLOY_ROOT}/config/pd-proxy.env"
```

输出件不包含模型、wheel、完整日志、profiler 或凭据。

## 14. 常见故障

| 现象 | 优先检查 |
| --- | --- |
| vLLM-Ascend commit/branch 不一致 | 是否使用上游 `vllm-project/vllm-ascend` 的 `rfc/vllm_cann` 和固定 commit |
| import 出现 ABI 或 undefined symbol | torch/torch-npu/CANN 是否成套，构建日志是否混入 CANN 9.0.x |
| `libatb.so` not found | 是否安装并 source 与 CANN 9.1.0 匹配的 NNAL/ATB |
| Mooncake `ldd` 出现 not found | wheel 构建机和部署机 ABI 是否一致，是否使用 CANN 9.1.0 重建 |
| 本机 Mooncake round-trip 失败 | `/etc/hccn.conf`、2 MiB 对齐、NPU 物理 ID 和端口 |
| FFN 一直等待 | FFN/Attention 是否紧邻启动，AFD 端口和 rank 数是否一致 |
| 8911 health 失败 | 正常现象；8911 不是 FFN HTTP 服务 |
| Proxy 正常但无 KV transfer 日志 | 请求是否真正经过 Prefill，Mooncake consumer 是否只挂在 Attention |
| 重启端口或显存未释放 | 是否按 `pd.sh stop` 正常停止并等待 process group 退出 |

## 15. 最终交付检查表

- 两端 Driver/Firmware、CANN 9.1.0、NNAL/ATB 和 SoC 匹配；
- 环境变量和动态库解析中没有 CANN 9.0.x；
- vLLM 使用 `releases/v0.23.0` 和固定 commit；
- vLLM-Ascend 使用上游 `rfc/vllm_cann` 和 commit `3da28f9414...`；
- afd-plugin 使用 `feat/dsv4-afd-mooncake-pd` 和固定 commit；
- 两端 Python 包、Mooncake wheel SHA 和 `.so` 指纹完全一致；
- 两端 16 张 NPU、模型、`/dev/shm`、网卡和端口检查通过；
- 两端 Mooncake 2 MiB 两轮本机 round-trip 通过；
- Prefill、Attention、8 个 FFN loop 和 Proxy 全部 ready；
- Proxy 请求产生真实跨机 KV transfer；
- golden、batch、取消恢复、二次启动和正常清理全部通过；
- 已归档三个 `collect` 输出件及其 SHA256。
