# DeepSeek-V4 AFD HCCL P2P 安装部署指南（含双机 Mooncake PD）

## 0. 新环境快速入口

### 0.1 `dsv4-afd-hccl-install-delivery-*.zip` 是否必需

不必需。它只是曾用于交付**单机 HCCL MTP M1** 指导书和 slim 安装脚本的外层
容器，不是运行依赖，也不是另一个压缩包内部必须存在的嵌套文件。如果已经拿到
本指导书或 afd-plugin 源码，不需要再寻找该 ZIP。双机 Mooncake PD 部署也不要
使用其中的 MTP M1 补丁树替代 M9 PD 源码。

如果确实收到了该 ZIP，可以按第 16 节使用；没有 ZIP 时，按第 4.1 节从 Git
下载固定源码即可。两条路径最终使用的运行栈必须经过相同的版本门禁。

新环境需要提前取得的内容如下：

| 内容 | 是否必需 | 获取方式 |
| --- | --- | --- |
| `dsv4-afd-hccl-install-delivery-*.zip` | 否 | 仅为可选单机离线交付容器 |
| vLLM、vLLM-Ascend、afd-plugin 源码 | 是 | 第 4.1 节按固定 commit 下载；内网环境使用包含相同 commit 的镜像 |
| Driver、Firmware、CANN/NNAL 9.0.1 | 是 | 使用目标 Atlas A3 SKU 对应的官方或内部安装介质 |
| DeepSeek-V4-Flash W8A8 模型 | 是 | 放到两台 NPU 节点的固定模型目录；本仓库不包含模型 |
| Mooncake 0.3.9 Ascend Direct 运行库 | 双机 PD 必需 | 优先复用镜像内置包并按第 17.2-17.5 节验收；不通过时才使用交付 wheel |
| `golden_results.json` | 完整验收必需 | 从相同模型和固定原生栈基线生成或由交付方提供；只做 smoke 时可暂缺 |

注意：afd-plugin 源码不能替代 Mooncake 二进制运行库。目标镜像已经包含
Mooncake 时不需要再传 wheel，但它必须位于服务实际使用的 Python 3.12 环境，
并通过版本、动态库、CANN 路径、本机 round-trip 和双机实传门禁。只看到
`pip show` 或 `import mooncake` 成功不能判定可用。

### 0.2 `/data/z00569729` 目录规划

A3-P、A3-D 使用相同目录布局，Proxy 推荐直接运行在 A3-P，避免准备第三套环境：

```text
/data/z00569729/
├── code/
│   ├── .ascend/cann-9.0.1/cann-9.0.1/
│   ├── .venvs/afd-v023-vllm-cann/
│   ├── vllm-release-v0.23.0/
│   ├── vllm-ascend-rfc-vllm-cann/
│   └── afd-plugin/
├── packages/
│   └── mooncake_transfer_engine-0.3.9-cp312-cp312-manylinux_2_39_aarch64.whl  # wheel 模式才需要
├── models/DeepSeek-V4-Flash-w8a8-mtp/
├── config/
│   ├── pd-prefill.env
│   ├── pd-decode.env
│   └── pd-proxy.env
├── run/dsv4-afd-mooncake-pd/
└── validation/
    └── dsv4_v023_vllm_cann_native_baseline/golden_results.json
```

先在两台目标机执行：

```bash
export DEPLOY_ROOT=/data/z00569729
export CODE_ROOT="${DEPLOY_ROOT}/code"
export PACKAGE_ROOT="${DEPLOY_ROOT}/packages"
export MODEL_PATH="${DEPLOY_ROOT}/models/DeepSeek-V4-Flash-w8a8-mtp"
export CONFIG_ROOT="${DEPLOY_ROOT}/config"
export RUN_ROOT="${DEPLOY_ROOT}/run/dsv4-afd-mooncake-pd"

mkdir -p \
  "${CODE_ROOT}" "${PACKAGE_ROOT}" "${DEPLOY_ROOT}/models" \
  "${CONFIG_ROOT}" "${RUN_ROOT}" "${DEPLOY_ROOT}/validation"
test -w "${DEPLOY_ROOT}"
```

本文后续命令均按该目录规划编写。若 CANN 由管理员安装在其他位置，只修改
配置中的 `CANN_ROOT`，不要复制或混用另一套 CANN；其余源码、venv、模型、配置、
日志和验收产物仍放在 `/data/z00569729` 下。

### 0.3 从空白目标机到双机 PD 验证

推荐按以下顺序执行，不要跳过版本和本地传输门禁：

1. 两台 A3 完成第 3 节的系统包、Driver/Firmware、CANN/NNAL 和 16 NPU 检查；
2. 两台 A3 按第 4.1 节下载并锁定三个源码 commit；
3. 两台 A3 按第 5-7 节创建 Python 3.12 venv、安装并验证固定运行栈；
4. 放置模型和 golden 文件；镜像没有合格 Mooncake 时再放置交付 wheel；
5. 按第 17.0 节生成三角色配置，执行 `install` 和 `check`；
6. 按 Prefill、Decode、Proxy 顺序 `start`，用 `status` 检查 readiness；
7. 从 Proxy 配置执行 `validate`，最后按第 17.12 节执行 `collect`。

任一步失败都先修复当前门禁，不要继续启动服务。特别是源码 commit、CANN 路径、
Mooncake 运行库指纹或本地 round-trip 不一致时，跨机验证结果无效；使用 wheel
模式时还必须核对 wheel SHA256。

## 1. 适用范围

本文给出 `P2pHcclAFDConnector` 的单机 Atlas A3 安装、启动和验收方法，并在
第 17 节增加两台 Atlas A3 上的 Mooncake PD + AFD 手工部署门禁。单机安装包主
路径严格复现以下已验证基线：

| 组件 | 固定版本 |
| --- | --- |
| Python | 3.12，验证环境为 3.12.9 |
| CANN / NNAL | 9.0.0 |
| torch | 2.10.0 |
| torch-npu | 2.10.0.post2 |
| vLLM | `releases/v0.23.0`，`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann`，`3da28f9414583d2d0b672a8f06d1fae142404bda` |
| afd-plugin（单机安装包） | tag `dsv4-afd-v023-hccl-mtp-m1-v1` |
| afd-plugin（Mooncake PD） | `feat/dsv4-afd-mooncake-pd` 上的 `49bb4a1dda5f7a59dcfbb45ea36d3ad1b2b89193`；M9 F0 前不创建功能 tag |
| transformers | 5.5.4 |
| numpy | 2.2.6 |
| 硬件 | 16 NPU Atlas A3，验证机型 SoC 为 `ascend910_9362` |

推荐部署为 Attention NPU 0-7、FFN NPU 8-15，即 A8F8；首轮使用
DP8/TP1/EP8，随后使用 DP4/TP2/EP8 验证 M8 已冻结的 TP2 契约。
单机安装包覆盖 eager/U1、eager/U2、等量 A/F 下的 `FULL_DECODE_ONLY`
Graph/U1，以及等量 A8F8 的 eager/U1 + MTP。MTP 首版只支持 1 个 MTP layer、
`method=mtp`、`num_speculative_tokens=1`。第 17 节先验收 Decode DP8/TP1 或
DP4/TP2、eager/U1、MTP off；脚本已为后续 PD + eager/U2、Graph/U1、Graph/U2
和一 token MTP 提供路径匹配的 control/AFD 配置入口，但每项在双 A3 F0 前都不是
冻结能力。TP2 full-draft Graph U2 + MTP、sequence parallel 和 Attention-side
gate 不在当前支持范围内。

截至 `2026-08-24`，Mooncake 0.3.9 的运行库/metadata contract 和单机两进程
Ascend 2 MiB round-trip 已通过，但两台 A3 的实模 F0 尚未执行。因此本文提供的
是可执行安装和验收流程，不得在完成第 17.11 节全部门禁前宣称 PD 功能基线通过。

本文不适用于 afd-plugin 主 README 当前的 vLLM 0.26 默认栈。不要把 0.26的 vLLM 或 vLLM-Ascend 快照混入本指南的 0.23 Graph/U1 环境。

## 2. 组件关系

三层 Python 组件按以下方向依赖：

```text
vLLM                 通用推理框架、API、调度和采样
  ^
  |
vLLM-Ascend          Ascend platform、NPU worker、DSV4、ACL Graph 和 NPU 算子
  ^
  |
afd-plugin           Attention/FFN 角色、AFD worker 和 P2pHcclAFDConnector

torch-npu -> CANN -> HCCL -> Atlas NPU
```

`P2pHcclAFDConnector` 在 afd-plugin 内，不是一个独立的 pip 包。它通过`torch.distributed.send/recv` 使用 torch-npu 注册的 HCCL backend。不要为此额外执行 `pip install hccl`。

HCCL 路径不依赖 afd-plugin 的 CAMP2P A2E/E2A 自定义算子，但 DSV4 模型仍依赖 vLLM-Ascend 自己的 `custom_transformer` 算子。因此运行时需要 source vLLM-Ascend 的 custom-op 环境，不能 source afd-plugin 的 CAMP2P 环境。

## 3. 系统和 CANN 前置条件

### 3.1 系统工具

需要 Linux aarch64、GCC/G++ 8 以上、C++17、CMake 3.26 以上、Ninja、Git、NUMA 开发库和足够的 `/dev/shm`。根据操作系统安装工具，例如：

```bash
# Ubuntu
sudo apt-get update
sudo apt-get install -y \
  gcc g++ cmake ninja-build libnuma-dev git curl jq iproute2

# openEuler
sudo yum install -y \
  gcc gcc-c++ cmake ninja-build numactl-devel git curl jq iproute
```

### 3.2 Driver、Firmware、CANN 和 NNAL

先安装与 CANN 9.0.1 匹配的 Ascend Driver/Firmware，再安装以下 CANN 组件：

- Toolkit 9.0.1；
- 与 SoC 匹配的 kernels/ops 包；
- NNAL/ATB 9.0.1，运行 DSV4 时需要 `libatb.so`。

安装介质和命令因发行版及硬件 SKU 不同，应以对应 CANN 9.0.1 发布包为准。
安装后在一个干净 shell 中检查：

```bash
export CANN_ROOT=/data/z00569729/code/.ascend/cann-9.0.1/cann-9.0.1

source "${CANN_ROOT}/set_env.sh"
if [[ -f "${CANN_ROOT}/nnal/atb/set_env.sh" ]]; then
  source "${CANN_ROOT}/nnal/atb/set_env.sh"
fi

"${CANN_ROOT}/query_pkg_version.sh" | sed -n '1,20p'
npu-smi info
npu-smi info -t board -i 0 -c 0
```

A8F8 要求 16 张 NPU 可见且没有遗留推理进程。验证机 `NPU Name` 为 `9362`，对应构建参数 `SOC_VERSION=ascend910_9362`。其他 A3 SKU 必须使用实际 SoC，不能直接照搬该值。

不要在同一个 shell 中先后 source CANN 9.0.1 和 9.1.0。若以下命令还看到其他 CANN 版本，重新打开干净 shell：

```bash
env | sort | grep -E '^(ASCEND|CANN|PATH|PYTHONPATH|LD_LIBRARY_PATH)='
```

## 4. 获取固定源码

单机 MTP M1 和双机 Mooncake PD 使用不同的 afd-plugin 源码点。新环境必须先
选择部署目标，不能先安装 MTP M1 补丁树，再在同一工作树上覆盖 PD 分支。两种
部署共用相同的 vLLM 和 vLLM-Ascend 固定提交。

### 4.1 双机 Mooncake PD：从 Git 下载完整源码

以下命令在 A3-P、A3-D 分别执行。Proxy 若放在 A3-P，不需要第三次下载。外网
不可达时可把三个 URL 改为内部 Git 镜像，但镜像必须包含表中精确 commit：

| 仓库 | 远端分支（用于定位） | 部署 commit |
| --- | --- | --- |
| vLLM | `releases/v0.23.0` | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann` | `3da28f9414583d2d0b672a8f06d1fae142404bda` |
| afd-plugin | `feat/dsv4-afd-mooncake-pd` | `49bb4a1dda5f7a59dcfbb45ea36d3ad1b2b89193` |

分支名只用于 `fetch`；服务必须 checkout 40 位 commit。本文更新时已确认
`origin/feat/dsv4-afd-mooncake-pd` 包含上述 afd-plugin commit。

```bash
export DEPLOY_ROOT=/data/z00569729
export CODE_ROOT="${DEPLOY_ROOT}/code"
export VLLM_COMMIT=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
export VLLM_ASCEND_COMMIT=3da28f9414583d2d0b672a8f06d1fae142404bda
export AFD_PD_COMMIT=49bb4a1dda5f7a59dcfbb45ea36d3ad1b2b89193

mkdir -p "${CODE_ROOT}"
test ! -e "${CODE_ROOT}/vllm-release-v0.23.0"
test ! -e "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann"
test ! -e "${CODE_ROOT}/afd-plugin"

git clone --no-checkout https://github.com/vllm-project/vllm.git \
  "${CODE_ROOT}/vllm-release-v0.23.0"
git -C "${CODE_ROOT}/vllm-release-v0.23.0" checkout --detach \
  "${VLLM_COMMIT}"

git clone --no-checkout https://github.com/vllm-project/vllm-ascend.git \
  "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann"
git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" checkout --detach \
  "${VLLM_ASCEND_COMMIT}"
git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" \
  submodule update --init --recursive

git clone --no-checkout https://github.com/wenhow/afd-plugin.git \
  "${CODE_ROOT}/afd-plugin"
git -C "${CODE_ROOT}/afd-plugin" fetch origin feat/dsv4-afd-mooncake-pd
git -C "${CODE_ROOT}/afd-plugin" checkout --detach "${AFD_PD_COMMIT}"
```

下载完成后立即执行门禁，三条 HEAD 必须与上表逐字一致，三个工作树必须为空：

```bash
test "$(git -C "${CODE_ROOT}/vllm-release-v0.23.0" rev-parse HEAD)" = \
  "${VLLM_COMMIT}"
test "$(git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" rev-parse HEAD)" = \
  "${VLLM_ASCEND_COMMIT}"
test "$(git -C "${CODE_ROOT}/afd-plugin" rev-parse HEAD)" = \
  "${AFD_PD_COMMIT}"

test -z "$(git -C "${CODE_ROOT}/vllm-release-v0.23.0" status --short)"
test -z "$(git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" status --short)"
test -z "$(git -C "${CODE_ROOT}/afd-plugin" status --short)"
```

如果目录已经存在，先查看 `remote -v`、`rev-parse HEAD` 和 `status --short`。
不要删除、覆盖或强制切换有本地改动的目录；改用新的空目录完成固定版本部署。

交付的 `afd-plugin-dsv4-mooncake-pd-update12-20260827.tar.gz` 是基于上述
afd-plugin commit 的文档/运行检查及管理工具 overlay，不修改推理模型代码。先校验同目录
下的 `.sha256`，再解压到代码根目录：

```bash
cd /data/z00569729/packages
sha256sum -c afd-plugin-dsv4-mooncake-pd-update12-20260827.tar.gz.sha256
tar -xzf afd-plugin-dsv4-mooncake-pd-update12-20260827.tar.gz \
  -C /data/z00569729/code
```

解压后 afd-plugin 的 HEAD 仍为 `49bb4a1dda5f7a59dcfbb45ea36d3ad1b2b89193`，
Git 会显示包内 8 个文档/运行检查及管理工具文件被修改。新版 `pd.sh` 只允许这 8 个路径作为
交付 overlay；任何 recipe、connector 或其他运行时代码改动仍会终止部署。

### 4.2 单机 HCCL MTP M1：可选 slim 包路径

本节只适用于单机 MTP M1，不用于双机 PD。使用轻量安装包时，在修改
`config.env` 后直接执行 `bash bin/02_prepare_sources.sh`。只有需要排查包内
源码恢复过程时，才手工执行下面的等价命令：

```bash
export INSTALL_BUNDLE_ROOT=/data/z00569729/packages/dsv4-afd-hccl-manual-install-slim-YYYYmmdd_HHMMSS
export CODE_ROOT=/data/z00569729/code
mkdir -p "${CODE_ROOT}"

git clone https://github.com/vllm-project/vllm.git \
  "${CODE_ROOT}/vllm-release-v0.23.0"
git -C "${CODE_ROOT}/vllm-release-v0.23.0" checkout --detach \
  0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665

git clone https://github.com/vllm-project/vllm-ascend.git \
  "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann"
git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" checkout --detach \
  3da28f9414583d2d0b672a8f06d1fae142404bda
git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" \
  submodule update --init --recursive

git clone https://github.com/wenhow/afd-plugin.git \
  "${CODE_ROOT}/afd-plugin"
git -C "${CODE_ROOT}/afd-plugin" checkout --detach \
  d7aeb9b7554803931e42bf405623f212030ed60f

(
  cd "${INSTALL_BUNDLE_ROOT}"
  sha256sum -c manifest/SHA256SUMS
)
git -C "${CODE_ROOT}/afd-plugin" apply --index \
  "${INSTALL_BUNDLE_ROOT}/manifest/afd-plugin-mtp-m1.patch"

test "$(git -C "${CODE_ROOT}/afd-plugin" write-tree)" = \
  8f2dfdb1533353d424ccfd78d66d8647df37ac85
```

MTP M1 tag 尚未发布到远端，因此不能直接按 tag checkout。上述已发布基础提交
加包内补丁会重建与 `dsv4-afd-v023-hccl-mtp-m1-v1` 完全相同的源码 tree。

核对三个固定点：

```bash
git -C "${CODE_ROOT}/vllm-release-v0.23.0" rev-parse HEAD
git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" rev-parse HEAD
git -C "${CODE_ROOT}/afd-plugin" rev-parse HEAD
git -C "${CODE_ROOT}/afd-plugin" write-tree
```

预期依次为 vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、
vLLM-Ascend `3da28f9414583d2d0b672a8f06d1fae142404bda`、afd-plugin 基础提交
`d7aeb9b7554803931e42bf405623f212030ed60f` 和 MTP M1 目标 tree
`8f2dfdb1533353d424ccfd78d66d8647df37ac85`。

## 5. 创建 Python 环境

### 5.1 创建 venv

验证基线使用 Python 3.12.9：

```bash
export DEPLOY_ROOT=/data/z00569729
export CODE_ROOT="${DEPLOY_ROOT}/code"
export CANN_ROOT="${CODE_ROOT}/.ascend/cann-9.0.1/cann-9.0.1"
export VENV_ROOT="${CODE_ROOT}/.venvs/afd-v023-vllm-cann"

test -f "${CANN_ROOT}/set_env.sh"
source "${CANN_ROOT}/set_env.sh"
if [[ -f "${CANN_ROOT}/nnal/atb/set_env.sh" ]]; then
  source "${CANN_ROOT}/nnal/atb/set_env.sh"
fi

command -v python3.12
python3.12 --version
if ldd "$(command -v python3.12)" | grep -q 'not found'; then
  echo "Python 3.12 has unresolved shared libraries" >&2
  exit 1
fi
python3.12 -m venv "${VENV_ROOT}"
source "${VENV_ROOT}/bin/activate"

python -m pip install --upgrade \
  pip "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8" \
  "setuptools-rust>=1.9.0" "packaging>=24.2" wheel jinja2 \
  "cmake>=3.26.1" ninja pybind11
```

此后所有 `python` 和 `pip` 命令都必须来自该 venv：

```bash
command -v python
python --version
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
python -m pip --version
```

四条命令中的 `python` 和 `pip` 都必须位于
`/data/z00569729/code/.venvs/afd-v023-vllm-cann/`，版本必须是 Python 3.12。
若输出仍指向 `/usr/local/python3.11*`，不要继续安装；说明 venv 没有激活或创建
时使用了错误解释器。Mooncake 的 `cp312` wheel 不能安装到 Python 3.11。

### 5.2 安装 NPU Python 运行时

torch、torch-npu 与 CANN 必须成套。华为源地址可按部署网络替换为内部镜像：

```bash
python -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  torch-npu==2.10.0.post2 triton-ascend==3.2.1

python -m pip install \
  -r "${CODE_ROOT}/vllm-release-v0.23.0/requirements/common.txt"
```

目标 `rfc/vllm_cann` 提交的 `requirements.txt` 仍写着`torch-npu==2.10.0`，而 Graph/U1 实际验证和运行检查要求
`2.10.0.post2`。安装该 requirements 后必须恢复验证版本：

```bash
python -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  -r "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann/requirements.txt"

python -m pip install --upgrade --force-reinstall --no-deps \
  torch-npu==2.10.0.post2 transformers==5.5.4 numpy==2.2.6
```

这个固定栈存在两个已知的包元数据偏差：vLLM-Ascend 源码元数据要求`torch-npu==2.10.0`，triton-ascend 3.2.1 元数据要求 numpy 1.26.4；已验证运行时分别使用 2.10.0.post2 和 2.2.6。因此 `pip check` 可能报告这两项，最终门禁应以第 7 节的实际 import、版本和 NPU 检查为准。不要在环境验证后执行无版本约束的 `pip install -U`。

## 6. 安装 vLLM、vLLM-Ascend 和 afd-plugin

安装顺序固定为 vLLM、vLLM-Ascend、afd-plugin。

### 6.1 安装 vLLM 0.23

Ascend 使用 `empty` target，避免构建 CUDA/ROCm 扩展：

```bash
test "$("${VENV_ROOT}/bin/python" -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = "3.12"
"${VENV_ROOT}/bin/python" -c \
  'import setuptools_rust, setuptools_scm, torch; print(torch.__version__)'

cd "${CODE_ROOT}/vllm-release-v0.23.0"
VLLM_TARGET_DEVICE=empty \
  "${VENV_ROOT}/bin/python" -m pip install \
  --no-build-isolation --no-deps --editable .
```

预期版本为 `0.23.0+empty`。这里必须保留 `--no-build-isolation`：该固定提交的
`pyproject.toml` 构建隔离依赖写有 `torch==2.11.0`，而本指南运行栈固定使用
torch 2.10.0 和 torch-npu 2.10.0.post2。无构建隔离意味着
`setuptools-rust` 等构建依赖必须先按第 5.1 节安装到目标 venv。

### 6.2 安装 vLLM-Ascend

构建前必须已经 source CANN 9.0.1，并使用与设备匹配的 SoC：

```bash
export SOC_VERSION=ascend910_9362
cd "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann"
"${VENV_ROOT}/bin/python" -m pip install \
  -v --no-build-isolation --no-deps --editable .
```

构建成功后必须存在以下文件：

```bash
export VLLM_ASCEND_OPS_ENV="${CODE_ROOT}/vllm-ascend-rfc-vllm-cann/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
test -f "${VLLM_ASCEND_OPS_ENV}"
```

### 6.3 安装 HCCL Connector

HCCL-only 部署不构建 afd-plugin 的 CAMP2P 自定义算子：

```bash
cd "${CODE_ROOT}/afd-plugin"
AFD_BUILD_ASCEND_OPS=0 \
  "${VENV_ROOT}/bin/python" -m pip install \
  -v --no-build-isolation --no-deps --editable .
```

如果同一个环境还要运行 `CAMP2pAFDConnector`，应改用
`AFD_BUILD_ASCEND_OPS=1` 重新安装，并配置对应 custom-op 环境。但运行
`P2pHcclAFDConnector` 时仍不要 source
`afd_plugin/_cann_ops_custom/vendors/afd-plugin/bin/set_env.bash`。

## 7. 激活和安装验收

### 7.1 使用固定激活脚本

```bash
cd "${CODE_ROOT}/afd-plugin"
export DSV4_CANN_ROOT="${CANN_ROOT}"
export DSV4_RUNTIME_VENV="${VENV_ROOT}"
export DSV4_VLLM_ROOT="${CODE_ROOT}/vllm-release-v0.23.0"
export DSV4_VLLM_ASCEND_ROOT="${CODE_ROOT}/vllm-ascend-rfc-vllm-cann"
source tools/dsv4/activate_v023_vllm_cann_runtime.sh
```

该脚本还会设置：

```text
VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd
```

仓库脚本清理旧环境时会保留验证基线的 `/opt/buildtools/python-3.12.9` 路径，
随后把目标 venv 放到 PATH 首位。目标机不要求存在该 `/opt` 目录，但创建 venv
所用的 Python 3.12 必须能在没有额外私有 `LD_LIBRARY_PATH` 的情况下启动，前述
`ldd` 不能出现 `not found`。不要为了改 Python 路径直接编辑 afd-plugin 的已
checkout 文件；`pd.sh check` 只接受第 4.1 节交付包中的 8 个工具 overlay 路径，
其他工作树改动都会拒绝部署。

NNAL 的 `set_env.sh` 会通过 `import torch` 探测 C++ ABI，首次 source 可能需要
数十秒。先等待命令返回，再执行后续检查，不要在探测过程中重复 source。

### 7.2 HCCL-only 运行检查

```bash
python - <<'PY'
from importlib.metadata import version

import torch
import torch_npu
import vllm
import vllm_ascend  # noqa: F401

from afd_plugin.connectors.npu.p2p_hccl import P2pHcclAFDConnector

assert torch.__version__.startswith("2.10.0")
assert version("torch-npu") == "2.10.0.post2"
assert vllm.__version__.startswith("0.23.0")
assert version("vllm-ascend").endswith("g3da28f941")
assert version("transformers") == "5.5.4"
assert version("numpy") == "2.2.6"
assert P2pHcclAFDConnector.__name__ == "P2pHcclAFDConnector"
assert torch.npu.is_available()
assert torch.npu.device_count() == 16

print("DSV4_AFD_HCCL_RUNTIME_OK")
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("vllm", vllm.__version__)
print("vllm_ascend", version("vllm-ascend"))
PY

vllm serve --help=all > /tmp/dsv4-vllm-help.txt
for flag in \
  --kv-transfer-config \
  --data-parallel-size \
  --tensor-parallel-size \
  --enable-expert-parallel \
  --additional-config; do
  grep -Fq -- "${flag}" /tmp/dsv4-vllm-help.txt || {
    echo "missing vLLM CLI option: ${flag}" >&2
    exit 1
  }
done
```

import 通过但 CLI 选项缺失，通常说明 venv 中的 editable install 指向了另一份
vLLM/vLLM-Ascend 源码，或者安装顺序、commit 不正确；此时不要继续启动服务。

现有 `tools/dsv4/check_v023_vllm_cann_runtime.sh` 还会检查 afd-plugin 的
CAMP2P 自定义算子，因此它只适合 `AFD_BUILD_ASCEND_OPS=1` 的完整构建。
HCCL-only 安装使用上面的检查，不要因该脚本的 CAMP2P 检查失败误判 HCCL。

## 8. 部署前配置

### 8.1 模型和网络

确认模型目录至少包含 config、tokenizer 和所有 safetensors/index 文件：

```bash
export MODEL_PATH=/data/z00569729/models/DeepSeek-V4-Flash-w8a8-mtp
test -f "${MODEL_PATH}/config.json"
find "${MODEL_PATH}" -maxdepth 1 -type f | sort | sed -n '1,40p'
```

选择 HCCL/Gloo 通信网卡，并将 `HCCL_IF_IP` 设置为该网卡的真实 IPv4 地址：

```bash
export GLOO_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
ip -o -4 addr show dev "${HCCL_SOCKET_IFNAME}"
export HCCL_IF_IP=192.169.91.106
```

`192.169.91.106` 只是验证机示例，不能复制到其他节点。单机 A8F8 的 AFD
rendezvous 可以使用 `127.0.0.1`。

### 8.2 端口和 NPU

默认端口如下：

| 用途 | 端口 |
| --- | ---: |
| Attention API | 8910 |
| FFN 启动进程 | 8911 |
| AFD rendezvous | 29761 |
| Attention HCCL base | 51000 |
| FFN HCCL base | 52000 |

部署前确认端口空闲，且 NPU0-15 没有其他任务：

```bash
ss -ltnp | grep -E ':(8910|8911|29761|51000|52000)([[:space:]]|$)' || true
npu-smi info
df -h /dev/shm
```

## 9. 启动 A8F8 Graph/U1

在同一个 shell 中完成激活和变量设置。FFN 与 Attention 必须前后紧邻地
启动，不能等待 FFN HTTP ready 后再启动 Attention，因为双方会在 AFD/HCCL
初始化阶段互相等待。

```bash
cd "${CODE_ROOT}/afd-plugin"
source tools/dsv4/activate_v023_vllm_cann_runtime.sh

export MODEL_PATH=/data/z00569729/models/DeepSeek-V4-Flash-w8a8-mtp
export HCCL_IF_IP=192.169.91.106
export GLOO_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export AFD_HOST=127.0.0.1
export AFD_PORT=29761
export ATTENTION_RANKS=8
export FFN_RANKS=8
export ATTENTION_DEVICES=0,1,2,3,4,5,6,7
export FFN_DEVICES=8,9,10,11,12,13,14,15
export EXECUTION_MODE=full-decode-only
export U_BATCHES=1

mkdir -p /data/z00569729/run/dsv4-afd-hccl/logs

bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_ffn.sh \
  > /data/z00569729/run/dsv4-afd-hccl/logs/ffn.log 2>&1 &
ffn_pid=$!

sleep 2

bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_attention.sh \
  > /data/z00569729/run/dsv4-afd-hccl/logs/attention.log 2>&1 &
attention_pid=$!
```

脚本会自动：

- 为两个角色选择 `P2pHcclAFDConnector`；
- 只 source vLLM-Ascend 的 `custom_transformer` 环境；
- Attention 使用 NPU0-7，FFN 使用 NPU8-15；
- 设置 DP8/TP1/EP8、W8A8 Ascend quantization；
- Graph 模式使用 `FULL_DECODE_ONLY`，capture size 为 1/2/4/8。

模型加载、编译和 Graph capture 可能持续数分钟。进程存活不等于服务 ready。

## 10. Readiness 和请求验证

只向 Attention API 发请求。FFN 是 connector-driven 后台角色，不要把 FFN
端口当成业务健康接口。

```bash
curl -fsS --max-time 10 http://127.0.0.1:8910/health

grep -Eo 'AFD FFN EngineCore started; workers run connector loop' \
  /data/z00569729/run/dsv4-afd-hccl/logs/ffn.log | wc -l

grep -E 'enable_npugraph_ex|Graph capturing finished|Replaying aclgraph' \
  /data/z00569729/run/dsv4-afd-hccl/logs/attention.log
```

A8F8 下 FFN ready marker 应出现 8 次。Graph 模式还应看到 8 个 Attention
rank 完成 capture，并在请求阶段看到 ACL Graph replay。

发送一个 OpenAI Completions 请求：

```bash
curl -fsS http://127.0.0.1:8910/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4-afd",
    "prompt": "Please explain why deterministic validation matters.",
    "max_tokens": 32,
    "temperature": 0
  }'
```

同时检查两侧日志中没有 fatal marker：

```bash
grep -En \
  'EngineCore encountered a fatal error|AFD NPU FFN worker loop failed|Communication_Error|507015|Traceback' \
  /data/z00569729/run/dsv4-afd-hccl/logs/attention.log \
  /data/z00569729/run/dsv4-afd-hccl/logs/ffn.log
```

## 11. 自动化功能验收

已有 golden 文件时，优先使用 recipe runner。它会检查端口、启动两个角色、
等待 Attention API、执行串行 golden 和 batch 验证、按 Attention 后 FFN 的
顺序退出，并检查 fatal 日志与 NPU 清理：

```bash
cd "${CODE_ROOT}/afd-plugin"
source tools/dsv4/activate_v023_vllm_cann_runtime.sh
export HCCL_IF_IP=192.169.91.106
export GLOO_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0

python recipe/npu/P2pHcclAFDConnector/deepseek_v4/run_validation.py \
  --execution-mode full-decode-only \
  --u-batches 1 \
  --golden /data/z00569729/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json \
  --cycles 1 \
  --idle-seconds 0 \
  --rounds 3 \
  --batch-sizes 1 8 32 \
  --output-dir /data/z00569729/validation/dsv4_afd_v023_hccl_graph_u1_install_gate
```

最终检查 `validation_summary.json` 中的 `passed`，并确认 Attention/FFN
返回码均为 0、NPU cleanup 通过。安装 smoke 不能替代 golden token 验收。

在加载大模型前，也可以先做 A1F1、U1 的 HCCL 组件 round-trip：

```bash
mkdir -p /data/z00569729/validation/dsv4_hccl_component_smoke
python tools/dsv4/validate_hccl_p2p_roundtrip.py \
  --attention-devices 0 \
  --ffn-devices 8 \
  --stages 1 \
  --steps 2 \
  --port 29841 \
  --output /data/z00569729/validation/dsv4_hccl_component_smoke/summary.json
```

## 12. eager/U1 和 eager/U2

eager/U1 只需把两个角色的环境改成：

```bash
export EXECUTION_MODE=eager
export U_BATCHES=1
```

等量 A8F8 的 eager/U1 + MTP 在此基础上增加：

```bash
export ENABLE_MTP=1
export MTP_NUM_SPECULATIVE_TOKENS=1
```

也可以直接运行自动化功能门禁：

```bash
python recipe/npu/P2pHcclAFDConnector/deepseek_v4/run_validation.py \
  --connector P2pHcclAFDConnector \
  --execution-mode eager \
  --u-batches 1 \
  --enable-mtp \
  --mtp-num-speculative-tokens 1 \
  --golden /data/z00569729/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json \
  --cycles 1 --idle-seconds 0 --rounds 3 --batch-sizes 1 8 32 \
  --output-dir /data/z00569729/validation/dsv4_afd_v023_hccl_mtp_m1_install_gate
```

当前 MTP draft 使用学习式 gate。Attention 先发送包含各 DP token count 的固定
header，再发送 post-HC `[T,4096]` BF16 hidden；FFN 返回同 shape 的 MoE output。
MTP phase 不发送 input IDs，connector 会拒绝错误的 pre-HC `[T,4,4096]` 输入。

eager/U2 使用：

```bash
export EXECUTION_MODE=eager
export U_BATCHES=2
export DBO_DECODE_TOKEN_THRESHOLD=2
export DBO_PREFILL_TOKEN_THRESHOLD=12
```

Graph 只能与 U1 组合。若设置 `EXECUTION_MODE=full-decode-only` 且
`U_BATCHES=2`，recipe 会直接拒绝启动。

MTP M1 只能与 eager/U1 和 A8F8 等量拓扑组合。Graph + MTP、U2 + MTP、
非等量 + MTP 或 `MTP_NUM_SPECULATIVE_TOKENS` 大于 1 会在启动前 fail-fast。

eager 支持 `A = k x F` 的整数倍 connector 拓扑，但 FFN 的
`max_num_batched_tokens` 至少要是 Attention 值的 `k` 倍。A3 上 A8F4 的完整
DeepSeek-V4 FFN EP4 已确认受 64 GiB HBM 容量限制，这不是 HCCL connector
故障。生产首选仍是已完整验证的 A8F8。

## 13. 正常停止

必须先停止 Attention，再停止 FFN，让 connector 按协议关闭：

```bash
kill -TERM "${attention_pid}"
wait "${attention_pid}"

kill -TERM "${ffn_pid}"
wait "${ffn_pid}"

npu-smi info
ss -ltnp | grep -E ':(8910|8911|29761|51000|52000)([[:space:]]|$)' || true
```

正式部署应由 systemd、Supervisor 或容器编排分别管理两个角色，并保留相同
的启动并发关系和停止顺序。不要用模糊的全局 `pkill` 清理服务。

## 14. 常见故障

| 现象 | 优先检查 |
| --- | --- |
| import 时找不到 `.so` 或出现 ABI 错误 | CANN、torch、torch-npu 是否成套；是否混入 9.1.0 路径 |
| `libatb.so` 找不到 | NNAL/ATB 是否安装并 source 对应 `set_env.sh` |
| `P2pHcclAFDConnector` 未注册 | afd-plugin 是否安装；`VLLM_PLUGINS` 是否包含 `afd` |
| vLLM-Ascend custom op 不存在 | 是否初始化 submodule；是否在正确 CANN/SoC 下重新构建 vLLM-Ascend |
| HCCL bind/connect 错误 | `HCCL_IF_IP`、两个 socket interface、AFD port 和 base port 是否一致且空闲 |
| FFN 启动后一直等待 | Attention 是否在 2 秒后并发启动；双方 AFD host/port/rank 数是否一致 |
| Attention health 正常但请求 hang | FFN 8 个 rank 是否都进入 connector loop；两侧 HCCL 日志是否报错 |
| Graph 首次请求失败 | 是否使用固定 tag、torch-npu 2.10.0.post2、Graph/U1 和等量 A/F |
| MTP 配置启动即拒绝 | 是否为 HCCL connector、A8F8、eager/U1、1 个 MTP layer 和 1 个 speculative token |
| MTP FFN 收到 shape 错误 | 远端边界必须是 post-HC `[T,4096]`；不要发送 target hidden buffer 的三维 view |
| 启动数分钟仍未 ready | 查看模型加载、编译和 capture 进度；不要只看父进程存活 |
| 重启报端口占用或显存未释放 | 先按角色 PID 正常停止，核对端口和 `npu-smi info`，不要叠加启动第二套服务 |

固定 vLLM 0.23 的 FFN API launcher 在计划内 SIGTERM 后可能于
`[shutdown] MPClient: complete` 之后打印 `KeyboardInterrupt: terminated` 和
`ERR99999`。只有两侧返回码为 0、请求阶段没有 fatal marker 且 NPU cleanup 通过
时，才将它归类为已知 shutdown 噪声；其他位置的 traceback 仍按失败处理。

## 15. 生产交付检查表

- 三个源码提交/tag 与第 1 节一致；
- CANN 环境中没有其他版本路径；
- `torch.npu.device_count() == 16`；
- Attention/FFN 使用不重叠的 0-7 和 8-15；
- HCCL/Gloo 网卡和本机 IP 已按部署机修改；
- 业务请求只进入 Attention API；
- 8 个 FFN rank 均进入 connector loop；
- Graph/U1 的 capture/replay 证据完整；
- 使用 MTP 时确认 eager/U1、proposal/acceptance 日志和 MTP phase 证据完整；
- golden、batch、fatal-log、正常退出和 NPU cleanup 门禁通过；
- 日志、版本、启动环境和验收产物已归档。

双机 Mooncake PD 还必须满足：

- Prefill 和 Decode 使用相同的 vLLM、vLLM-Ascend、afd-plugin commit、
  Mooncake 版本和运行库指纹；wheel 模式还要求 wheel SHA256 一致；
- 两端 `VLLM_HOST_IP` 都是对端可达的业务/传输网 IPv4，不能是
  `127.0.0.1` 或自动选择出的错误管理网地址；
- Decode FFN 没有 `--kv-transfer-config`，Mooncake consumer 只在 Decode
  Attention；
- Proxy 的请求确实先进入 Prefill，再携带 `kv_transfer_params` 进入 Decode；
- Decode Attention 日志出现成功的 KV cache transfer 记录；
- 通过 Proxy 完成 30/30 token exact、batch 1/8/32、二次启动和双机清理。

功能范围和正式验证证据见
[`DEEPSEEK_V4_AFD_HCCL_P2P_GRAPH_U1_VALIDATION_REPORT_ZH.md`](DEEPSEEK_V4_AFD_HCCL_P2P_GRAPH_U1_VALIDATION_REPORT_ZH.md)
和
[`DEEPSEEK_V4_AFD_HCCL_P2P_MTP_M1_VALIDATION_REPORT_ZH.md`](DEEPSEEK_V4_AFD_HCCL_P2P_MTP_M1_VALIDATION_REPORT_ZH.md)。

## 16. 可移植安装脚本包

用于其他 A3 环境手工安装的脚本和打包器位于：

```text
tools/dsv4/hccl_manual_install/
```

默认生成轻量 transfer archive。包内只包含脚本、固定版本清单和 afd-plugin
MTP M1 补丁；目标机重新下载 vLLM、vLLM-Ascend（含递归 submodule）和
afd-plugin 基础源码：

```bash
bash tools/dsv4/hccl_manual_install/build_bundle.sh /data/z00569729/artifacts
```

生成物名称包含 `slim`，同时提供 `.tar.gz.sha256`。源码、模型和 Python wheel
均不进入轻量包。目标机可联网安装；若必须携带源码，可设置
`INCLUDE_SOURCES=1` 生成名称包含 `with-sources` 的完整包。Python wheel 可在
构建时设置 `INCLUDE_WHEELHOUSE=/data/z00569729/packages/wheelhouse` 加入。
详细步骤见
[`hccl_manual_install/README_ZH.md`](../../tools/dsv4/hccl_manual_install/README_ZH.md)。

`build_bundle.sh` 直接生成的 `.tar.gz` 和 `.sha256` 已经足够安装；它不会再生成
一个必须嵌套携带的 delivery ZIP。若交付流程需要 ZIP，可在外层自行归档指导书、
tar 包和校验文件，但这不会改变安装内容，也不适用于第 17 节的 PD 源码和 wheel。

正式包固定使用 `dsv4-afd-v023-hccl-mtp-m1-v1`。由于该 MTP M1 tag 尚未发布
到远端，轻量包从已发布提交 `d7aeb9b7554803931e42bf405623f212030ed60f`
下载 afd-plugin，再应用包内补丁。打包和目标机安装都会校验最终 Git tree、
精确 commit 以及 SHA256。

该包当前不是 PD 交付包。M9 冻结正式 commit/tag 后，应另行升级 manifest、
加入 Mooncake 安装模式、运行库指纹、可选 wheel SHA256 和双机配置；在此之前
按第 17 节手工安装，不得把 MTP M1 包的 `06_verify_install.sh` 成功当作 PD
安装通过。

## 17. 双机 Mooncake PD + AFD 手工安装与验证

HCCL P2P 的 Attention/FFN launcher、MTP/Graph/PD 参数均由
`recipe/npu/P2pHcclAFDConnector/deepseek_v4/` 独立维护，不再转调或修改
`CAMP2pAFDConnector` launcher。连接器无关的运行栈激活、验证 runner 和 golden
比较工具统一位于 `recipe/npu/deepseek_v4/common/`。

### 17.0 推荐：使用统一脚本入口

> update12 重要变更：PD + AFD 不再直接使用 one-shot native golden 作为
> exact-token 基础设施门禁。必须先运行 `pd_control` 并执行 `record-control`，
> 再运行 `pd_afd` 并执行 `validate`。两台节点可直接照抄的命令、六个配置文件和
> 停止/收集顺序以
> [`tools/dsv4/mooncake_pd_manual/UPDATE12_RUNBOOK_ZH.md`](../../tools/dsv4/mooncake_pd_manual/UPDATE12_RUNBOOK_ZH.md)
> 为准。第 17.1-17.12 节保留的单模式手工命令只用于逐层排障，不能替代
> update12 的 path-matched 验收。

第 17.1-17.11 节保留完整命令，主要用于理解流程和逐层排障。正常手工部署推荐
使用 [`tools/dsv4/mooncake_pd_manual/pd.sh`](../../tools/dsv4/mooncake_pd_manual/pd.sh)，
避免手工遗漏环境变量、启动顺序、PID 管理或输出收集。

本节假定已完成第 3-7 节，三个源码 HEAD 与第 4.1 节一致，并且模型、golden
文件已经放到第 0.2 节规划的位置。Mooncake 使用镜像内置包或交付 wheel 二选一。
推荐把 Proxy 放在 A3-P：A3-P 保留
`pd-prefill.env` 和 `pd-proxy.env`，A3-D 只保留 `pd-decode.env`。

先在对应节点创建配置：

```bash
cd /data/z00569729/code/afd-plugin

# A3-P
bash tools/dsv4/mooncake_pd_manual/pd.sh init \
  /data/z00569729/config/pd-prefill.env
bash tools/dsv4/mooncake_pd_manual/pd.sh init \
  /data/z00569729/config/pd-proxy.env

# A3-D
bash tools/dsv4/mooncake_pd_manual/pd.sh init \
  /data/z00569729/config/pd-decode.env
```

编辑三个文件。以下路径和值必须写入三个配置，脚本不会自动搜索目标机目录：

```bash
CODE_ROOT="/data/z00569729/code"
VENV_ROOT="${CODE_ROOT}/.venvs/afd-v023-vllm-cann"
VLLM_ROOT="${CODE_ROOT}/vllm-release-v0.23.0"
VLLM_ASCEND_ROOT="${CODE_ROOT}/vllm-ascend-rfc-vllm-cann"
AFD_PLUGIN_ROOT="${CODE_ROOT}/afd-plugin"
CANN_ROOT="${CODE_ROOT}/.ascend/cann-9.0.1/cann-9.0.1"
CANN_VERSION="9.0.1"
ATB_ROOT=""
MODEL_PATH="/data/z00569729/models/DeepSeek-V4-Flash-w8a8-mtp"
GOLDEN_PATH="/data/z00569729/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json"
MOONCAKE_INSTALL_MODE="existing"
MOONCAKE_VERSION="0.3.9"
MOONCAKE_LIBRARY_DIR=""
MOONCAKE_WHEEL="/data/z00569729/packages/mooncake_transfer_engine-0.3.9-cp312-cp312-manylinux_2_39_aarch64.whl"

VLLM_COMMIT="0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"
VLLM_ASCEND_COMMIT="3da28f9414583d2d0b672a8f06d1fae142404bda"
AFD_PD_COMMIT="49bb4a1dda5f7a59dcfbb45ea36d3ad1b2b89193"

RUN_ROOT="/data/z00569729/run/dsv4-afd-mooncake-pd"
STATE_ROOT="${RUN_ROOT}/state/${NODE_ROLE}"
LOG_ROOT="${RUN_ROOT}/logs/${NODE_ROLE}"
VALIDATION_ROOT="${RUN_ROOT}/validation"
OUTPUT_ROOT="${RUN_ROOT}/output"
```

`MOONCAKE_INSTALL_MODE=existing` 表示保留镜像中已安装到该 `VENV_ROOT` 的
Mooncake，不读取 `MOONCAKE_WHEEL`；`wheel` 表示校验并安装交付文件。镜像场景
先选 `existing`，让 `check` 给出结果。如果 CANN 实际不在规划目录，必须同时把
`CANN_ROOT` 和 `CANN_VERSION` 改为目标机唯一且匹配的 CANN。不要修改 SHA256 来迁就另一
个 wheel。每个配置还必须填写：

当前 openEuler 镜像若使用 CANN 9.0.0，目标 venv 内有 Mooncake Python 扩展，
而配套库位于 `/usr/local/lib`，三个角色配置统一改为：

```bash
CANN_ROOT="/usr/local/Ascend/cann-9.0.0"
CANN_VERSION="9.0.0"
ATB_ROOT="/usr/local/Ascend/nnal/atb"
MOONCAKE_INSTALL_MODE="existing"
MOONCAKE_LIBRARY_DIR="/usr/local/lib"
```

这是对镜像现有 CANN/Mooncake 组合执行完整门禁，不表示 9.0.1 参考 wheel 可以
跨版本复用。`libtransfer_engine.so` 和 `ascend_transport.so` 必须来自该镜像的
同一套 Mooncake 构建产物。

`ATB_ROOT` 必须指向包含 `set_env.sh` 的目录。该镜像的 ATB 9.0.0 安装在 CANN
根目录之外；若不显式配置，worker 加载 `torch_npu/lib/libop_plugin_atb.so` 时会
报 `libatb.so: cannot open shared object file`。新版 `check` 会提前对该插件执行
`ldd`，不再把此错误推迟到 Prefill/Decode worker 启动阶段。
部分 ATB 9.0.0 `set_env.sh` 会直接读取未定义的 `ZSH_VERSION`，在 `pd.sh` 的
`set -u` 环境下会退出。交付脚本加载 CANN/ATB 环境时会临时关闭 `nounset` 并在
加载后恢复原状态；不要修改系统 ATB 文件，也不需要伪造 `ZSH_VERSION`。

- `NODE_ROLE`：分别为 `prefill`、`decode`、`proxy`；
- `PREFILL_IP`：A3-P 上目标网卡的、A3-D 可达的 IPv4；
- `DECODE_IP`：A3-D 上目标网卡的、A3-P 可达的 IPv4；
- `NIC_NAME`：上述 IP 所在网卡，不一定是 `eth0`。

三个配置的 commit、Mooncake 安装模式、网络端口和拓扑值必须一致。wheel 模式
还要求 wheel SHA256 一致；existing 模式则在 `check` 后比较两个 NPU 节点生成
的 `mooncake-libraries.sha256`。路径按每台机器本地文件系统填写；采用第 0.2
节布局时三份配置也应一致。保存后先确认没有任何占位符：

```bash
grep -En '^[A-Z_]+="[^"]*(CHANGE_ME|REPLACE_WITH)' \
  /data/z00569729/config/pd-*.env
```

没有输出才继续。然后分别安装并执行完整 preflight：

```bash
# A3-P
bash tools/dsv4/mooncake_pd_manual/pd.sh install \
  /data/z00569729/config/pd-prefill.env
bash tools/dsv4/mooncake_pd_manual/pd.sh check \
  /data/z00569729/config/pd-prefill.env
bash tools/dsv4/mooncake_pd_manual/pd.sh check \
  /data/z00569729/config/pd-proxy.env

# A3-D
bash tools/dsv4/mooncake_pd_manual/pd.sh install \
  /data/z00569729/config/pd-decode.env
bash tools/dsv4/mooncake_pd_manual/pd.sh check \
  /data/z00569729/config/pd-decode.env
```

`check` 默认在 A3-P、A3-D 各执行一次本机两 NPU、2 MiB、两轮 Mooncake
round-trip，并把包内所有 `.so` 的 SHA256 写入各自
`STATE_ROOT/mooncake-libraries.sha256`。两端该文件必须逐行一致。不得仅因
import 成功而设置 `RUN_LOCAL_ROUNDTRIP=0`。全部通过后，按 Prefill、Decode、
Proxy 顺序启动：

```bash
# A3-P
bash tools/dsv4/mooncake_pd_manual/pd.sh start \
  /data/z00569729/config/pd-prefill.env
bash tools/dsv4/mooncake_pd_manual/pd.sh status \
  /data/z00569729/config/pd-prefill.env

# A3-D；等待命令报告 8 个 FFN loop ready
bash tools/dsv4/mooncake_pd_manual/pd.sh start \
  /data/z00569729/config/pd-decode.env
bash tools/dsv4/mooncake_pd_manual/pd.sh status \
  /data/z00569729/config/pd-decode.env

# A3-P；后端 ready 后再启动 Proxy
bash tools/dsv4/mooncake_pd_manual/pd.sh start \
  /data/z00569729/config/pd-proxy.env
bash tools/dsv4/mooncake_pd_manual/pd.sh status \
  /data/z00569729/config/pd-proxy.env
bash tools/dsv4/mooncake_pd_manual/pd.sh validate \
  /data/z00569729/config/pd-proxy.env
```

随时用对应配置执行 `status`。最终必须同时看到 Prefill health、Decode Attention
health、8 个 FFN loop、Proxy health 和成功 KV transfer 证据。停止顺序固定为
Proxy、Decode、Prefill：

```bash
# A3-P
bash tools/dsv4/mooncake_pd_manual/pd.sh stop \
  /data/z00569729/config/pd-proxy.env
# A3-D
bash tools/dsv4/mooncake_pd_manual/pd.sh stop \
  /data/z00569729/config/pd-decode.env
# A3-P
bash tools/dsv4/mooncake_pd_manual/pd.sh stop \
  /data/z00569729/config/pd-prefill.env
```

脚本不会使用全局 `pkill`，只管理自己 PID 文件记录的 process group。完整说明见
[`tools/dsv4/mooncake_pd_manual/README_ZH.md`](../../tools/dsv4/mooncake_pd_manual/README_ZH.md)。

### 17.1 首个 M9 交付边界

首个双机功能门禁固定为：

| 节点 | 角色 | NPU | 并行配置 | 对外 HTTP |
| --- | --- | --- | --- | --- |
| A3-P | Prefill + Mooncake producer | 0-7 | DP2/TP4 | `8100` |
| A3-D | Decode Attention + Mooncake consumer | 0-7 | DP8/TP1 | `8910` |
| A3-D | Decode FFN | 8-15 | DP8/TP1/EP8 | 无业务 HTTP |
| 任一节点 | PD Proxy | CPU | 1 worker | `9000` |

TP1 路径匹配 F0 完成后，保持相同物理 NPU 和进程布局，仅把两个 Decode 角色
都切换到 DP4/TP2；Prefill 仍为 DP2/TP4。两种配置不能在同一轮混用。

两条通信链路相互独立：

```text
Client -> Proxy -> Prefill
                    |
                    | MooncakeHybridConnector: KV cache
                    v
                 Decode Attention
                    |
                    | P2pHcclAFDConnector: hidden state
                    v
                 Decode FFN
```

这个门禁只使用 eager/U1、MTP off、A8F8；先 TP1，后 TP2。不要为了减少机器数量把
Prefill 和完整 Decode A8F8 叠放在同一台 16-NPU A3；Decode 已经占满 16 张
NPU，Prefill 还需要 8 张 NPU。初次 F0 的 `MAX_MODEL_LEN` 固定为 4096，128K
留给功能基线后的容量专项。

### 17.2 冻结并同步源码和二进制

M9 功能 tag 只有在双机实模 F0 通过后才能创建。部署前由交付方提供一个已经
提交、已经推送的 40 位 commit，两个节点都按 commit checkout，不能直接跟随
会继续移动的分支头。vLLM、vLLM-Ascend 和 afd-plugin 运行时代码不能有未提交
修改；第 4.1 节校验过 SHA256 的 8 文件文档/管理工具 overlay 是唯一例外。本文固定的部署点是
`49bb4a1dda5f7a59dcfbb45ea36d3ad1b2b89193`；它位于
`feat/dsv4-afd-mooncake-pd`，但部署配置必须填写 commit 而不是分支名。

旧 `hccl_manual_install/bin/02_prepare_sources.sh` 会恢复 MTP M1 补丁树，不能
用它准备 PD 的 afd-plugin。新环境直接执行第 4.1 节；已有目录则在两台机器
分别执行以下复核：

```bash
export CODE_ROOT=/data/z00569729/code
export VLLM_COMMIT=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
export VLLM_ASCEND_COMMIT=3da28f9414583d2d0b672a8f06d1fae142404bda
export AFD_PD_COMMIT=49bb4a1dda5f7a59dcfbb45ea36d3ad1b2b89193

test "$(git -C "${CODE_ROOT}/vllm-release-v0.23.0" rev-parse HEAD)" = \
  "${VLLM_COMMIT}"
test "$(git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" rev-parse HEAD)" = \
  "${VLLM_ASCEND_COMMIT}"
test "$(git -C "${CODE_ROOT}/afd-plugin" rev-parse HEAD)" = "${AFD_PD_COMMIT}"
test -z "$(git -C "${CODE_ROOT}/vllm-release-v0.23.0" status --short)"
test -z "$(git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" status --short)"
git -C "${CODE_ROOT}/afd-plugin" status --short
```

未使用交付 overlay 时最后一条命令应无输出。使用 overlay 时只允许下面 8 个路径，
状态可以是 `M`，不能出现其他路径：

```text
docs/npu/DEEPSEEK_V4_AFD_HCCL_P2P_INSTALL_DEPLOYMENT_GUIDE_ZH.md
tests/unit/test_mooncake_pd_config.py
tools/dsv4/activate_runtime.sh
tools/dsv4/check_mooncake_runtime.sh
tools/dsv4/hccl_manual_install/bin/04_install_python_deps.sh
tools/dsv4/mooncake_pd_manual/README_ZH.md
tools/dsv4/mooncake_pd_manual/config.env.example
tools/dsv4/mooncake_pd_manual/pd.sh
```

两台机器还应保存以下输出，不能只记录分支名：

```bash
git -C "${CODE_ROOT}/vllm-release-v0.23.0" rev-parse HEAD
git -C "${CODE_ROOT}/vllm-ascend-rfc-vllm-cann" rev-parse HEAD
git -C "${CODE_ROOT}/afd-plugin" rev-parse HEAD
```

当前 A3 功能验证 wheel 为（镜像内置包通过后不必传输该文件）：

```text
mooncake_transfer_engine-0.3.9-cp312-cp312-manylinux_2_39_aarch64.whl
SHA256: 0f9964801b24fd683d6016e1196cc0606fc87b0285b45d89c433650b9477ca12
```

该 wheel 不在 afd-plugin Git 仓库，也不在旧单机 slim/ZIP 中。当前验证副本
位于交付工作区：

```text
/mnt/workspace/validation/dsv4_afd_v023_mooncake_pd_m9_contract_20260821_181148/
mooncake_transfer_engine-0.3.9-cp312-cp312-manylinux_2_39_aarch64.whl
```

它是从上游 `https://github.com/kvcache-ai/Mooncake` 的 tag `v0.3.9`、commit
`a00f75742469d6cc408bc5393b73c501e92dc74a` 在 aarch64 环境本地构建的 Ascend
Direct wheel。可追溯构建参数包括 Python 3.12.9、`USE_ASCEND_DIRECT=ON`、
`USE_ASCEND=OFF`、`WITH_STORE=ON`，编译/链接路径为
`/usr/local/Ascend/cann-9.0.1`。源码树还包含一处未提交修正：为
`mooncake_common` 增加 `yalantinglibs::yalantinglibs` 链接依赖。

生成过程是：在上述源码点编译 Mooncake C/C++ Transfer Engine、Ascend Direct
transport 和 Python 扩展，再由 `mooncake-wheel` 的 setuptools 配置将 Python
文件及这些共享库封装为 cp312/aarch64 wheel。wheel metadata 只记录版本
`0.3.9` 和上游项目地址，没有嵌入 Git commit 或 `direct_url.json`；工作区也
没有保存完整的原始构建命令日志。因此这里能说明构建来源和关键选项，不能声称
可按一条命令字节级复现。重新从 tag 构建的文件必须重新执行全部 Mooncake 门禁
并冻结新的 SHA256。

部署选择顺序如下：

1. 镜像已带 Mooncake：使用 `MOONCAKE_INSTALL_MODE=existing`，不传 wheel；
2. 镜像包未安装到目标 venv、版本不符、动态库缺失或门禁失败：改用已验证 wheel；
3. wheel 无法传输且镜像包失败：更换/重制包含合格 Mooncake 的镜像，不能从
   PyPI 临时下载普通 0.3.9 包代替 Ascend Direct 构建。

只有选择 wheel 模式时，才将它放到两个 A3 节点的相同路径并在两端执行：

```bash
export MOONCAKE_WHEEL=/data/z00569729/packages/mooncake_transfer_engine-0.3.9-cp312-cp312-manylinux_2_39_aarch64.whl
test -f "${MOONCAKE_WHEEL}"
test "$(sha256sum "${MOONCAKE_WHEEL}" | awk '{print $1}')" = \
  0f9964801b24fd683d6016e1196cc0606fc87b0285b45d89c433650b9477ca12
```

wheel 模式下任一节点缺少文件或 SHA256 不一致时立即停止。该 wheel 只作为
A3/CANN 9.0.1/Python 3.12 功能产物；A5 必须重新构建，不能复用。

### 17.3 两台 A3 安装 Mooncake 运行依赖

先按第 3-7 节在两端安装相同的 CANN 9.0.1、Python venv、vLLM、
vLLM-Ascend 和当前 M9 afd-plugin。Ubuntu 验证环境额外需要：

```bash
sudo apt-get update
sudo apt-get install -y \
  libgoogle-glog0v6t64 libjsoncpp25 libjemalloc2 netcat-openbsd
```

openEuler 或其他发行版应安装提供 `libglog.so`、`libjsoncpp.so` 和
`libjemalloc.so.2` 的对应包，然后以 `ldd` 门禁为准，不能照搬 Ubuntu 包名。
RPM/openEuler 基础镜像可以安装同时提供 `ip` 和 `ss` 的 `iproute`：

```bash
dnf install -y iproute || yum install -y iproute
command -v ip
command -v ss
```

Ubuntu/Debian 对应包为 `iproute2`。也可以在角色配置中设置
`INSTALL_SYSTEM_PACKAGES=1` 后重新执行 `pd.sh install`；新版脚本自动识别
`apt-get`、`dnf` 或 `yum`。`check` 本身始终只检查，不会安装系统包。

离线欧拉容器无法下载 `iproute` 时可使用镜像已有的 `net-tools`。新版 `pd.sh`
按 `ip -> ifconfig -> netstat -ie -> Python 标准库` 检查 `NIC_NAME` 的主 IPv4，
按 `ss -> netstat -> /proc/net/tcp*` 检查监听端口。该回退仍会拒绝错误网卡/IP
和已占用端口，不是跳过门禁；直接重新执行 `pd.sh check` 即可。

若配置使用 `MOONCAKE_INSTALL_MODE=existing`，先确认镜像中的包位于服务实际
使用的 Python，而不是另一个系统 Python：

```bash
export VENV_ROOT=/data/z00569729/code/.venvs/afd-v023-vllm-cann
"${VENV_ROOT}/bin/python" -c \
  'import importlib.util, sys; from importlib.metadata import version; s=importlib.util.find_spec("mooncake"); print(sys.executable); print(version("mooncake-transfer-engine")); print(s.origin if s else "NOT_FOUND")'
```

要求解释器是 `${VENV_ROOT}/bin/python`、版本为 `0.3.9`，且模块路径位于
`${VENV_ROOT}`。如果镜像只在系统 Python 中安装了 Mooncake，而服务使用新建
venv，则不能直接复用；不要用 `--system-site-packages` 把整套系统依赖泄漏进
固定环境，应改用包含完整目标 venv 的镜像或交付 wheel。

若配置使用 `MOONCAKE_INSTALL_MODE=wheel`，在两端安装同一个 wheel：

```bash
export VENV_ROOT=/data/z00569729/code/.venvs/afd-v023-vllm-cann
export MOONCAKE_WHEEL=/data/z00569729/packages/mooncake_transfer_engine-0.3.9-cp312-cp312-manylinux_2_39_aarch64.whl
"${VENV_ROOT}/bin/python" -m pip install \
  --no-deps --force-reinstall \
  "${MOONCAKE_WHEEL}"
```

然后在两端使用 PD launcher 将使用的同一环境执行：

```bash
cd /data/z00569729/code/afd-plugin
export DSV4_CANN_ROOT=/data/z00569729/code/.ascend/cann-9.0.1/cann-9.0.1
export DSV4_CANN_VERSION=9.0.1
export DSV4_RUNTIME_VENV=/data/z00569729/code/.venvs/afd-v023-vllm-cann
export DSV4_VLLM_ROOT=/data/z00569729/code/vllm-release-v0.23.0
export DSV4_VLLM_ASCEND_ROOT=/data/z00569729/code/vllm-ascend-rfc-vllm-cann

source tools/dsv4/check_mooncake_runtime.sh
python -c 'from importlib.metadata import version; print(version("mooncake-transfer-engine"))'
```

门禁要求输出 `MooncakeHybridConnector metadata contract passed`、
`Mooncake runtime check passed` 和版本 `0.3.9`。脚本会预加载
`libjemalloc.so.2`：Debian/Ubuntu 通常位于 `/usr/lib/aarch64-linux-gnu`，
openEuler/RPM 通常位于 `/usr/lib64`。脚本会检查常见路径并回退到 `ldconfig`；
特殊镜像可以显式设置 `MOONCAKE_JEMALLOC=/绝对路径/libjemalloc.so.2`。不能删除
这个 preload，否则
`torch_npu + Mooncake` 组合进程可能在退出阶段发生堆损坏。它还会检查 Mooncake
扩展来自目标 venv、两个配套库来自配置/自动探测的同一目录、`ldd` 没有
`not found`，且动态依赖中的 CANN 版本与 `CANN_VERSION` 一致。CANN 的
`PYTHONPATH` 即使包含同名 Mooncake，目标 venv 仍保持最高优先级。
统一入口的 `check` 还会执行本机实际 NPU round-trip；因此 existing 模式没有
降低功能门禁，只是省去 wheel 传输和重装。两个节点执行完 `check` 后比较：

```bash
sha256sum \
  /data/z00569729/run/dsv4-afd-mooncake-pd/state/prefill/mooncake-libraries.sha256
sha256sum \
  /data/z00569729/run/dsv4-afd-mooncake-pd/state/decode/mooncake-libraries.sha256
```

两端输出的总 SHA256 必须相同。existing 模式即使通过本机检查，也属于新的镜像
运行产物，最终仍必须以第 17.9-17.11 节真实跨机 KV transfer 和 golden 结果
完成验收，不能宣称与参考 wheel 字节一致。

### 17.4 双机网络和端口门禁

在开始部署前明确填写两个地址和网卡：

```bash
export PREFILL_IP="REPLACE_WITH_A3_P_IP"
export DECODE_IP="REPLACE_WITH_A3_D_IP"
export NIC_NAME="eth0"
```

分别在对应节点核对本机地址，不能把示例 IP 复制过去：

```bash
ip -o -4 addr show dev "${NIC_NAME}"
ping -c 3 "${PREFILL_IP}"
ping -c 3 "${DECODE_IP}"
```

`MooncakeHybridConnector` 通过 vLLM 的 `get_ip()` 把地址写入
`kv_transfer_params`，因此两个节点都必须显式设置：

```bash
export VLLM_HOST_IP="本机对端可达的 IPv4"
export HCCL_IF_IP="${VLLM_HOST_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"
```

当前 launcher 在 PD 模式下也会把未设置的 `VLLM_HOST_IP` 绑定到
`HCCL_IF_IP`，但正式部署仍建议显式设置并保存到环境记录。

需要允许的 TCP 端口如下：

| 节点/用途 | 端口 |
| --- | --- |
| A3-P Prefill HTTP | `8100` |
| A3-P Mooncake ZMQ handshake | `30000-30007` |
| A3-D Decode Attention HTTP | `8910` |
| A3-D Mooncake ZMQ handshake | `30100-30107` |
| Proxy HTTP | `9000` |
| 两端 Mooncake TransferEngine/Ascend transport 动态端口 | `15000-17000` |
| A3-D 本机 AFD rendezvous | `29761` |
| A3-D Attention/FFN HCCL base | `51000` / `52000` |

当前 Mooncake 源码使用以下名字限制动态 RPC 端口。`PRC` 是该版本已有的环境
变量拼写，不要自行改成 `RPC`：

```bash
export MC_MIN_PRC_PORT=15000
export MC_MAX_PRC_PORT=17000
```

防火墙必须允许两台机器双向访问 handshake 和 `15000-17000`。服务启动后再用
`ss -ltnp` 和 `nc -vz <peer> <port>` 验证实际监听；启动前端口未监听是正常的。

### 17.5 每台机器的本地 Mooncake NPU 组件门禁

在模型服务启动前，两台机器各自选择两个空闲 NPU 执行本地两进程传输。这个
update12 默认使用当前节点业务 IP 和网卡，使本机连接路径与正式 Mooncake
数据路径一致；它只证明本机 Mooncake Ascend 引擎可注册和传输 NPU buffer，
不代替第 17.9 节的跨机实模请求：

```bash
cd /data/z00569729/code/afd-plugin
source tools/dsv4/check_mooncake_runtime.sh
npu-smi info
python tools/dsv4/check_mooncake_npu_roundtrip.py \
  --producer-device 0 --consumer-device 1 \
  --host "REPLACE_WITH_LOCAL_BUSINESS_IP" \
  --interface "REPLACE_WITH_NIC_NAME"
npu-smi info
```

预期 JSON 包含 `"bytes":2097152`、`"iterations":2` 和
`"transfer_results":[0,0]`，两个子进程退出码均为 0。

### 17.6 启动 A3-P Prefill

在 A3-P 的一个干净 shell 中执行，所有 IP 都必须替换为实际值：

```bash
cd /data/z00569729/code/afd-plugin
export DSV4_CANN_ROOT=/data/z00569729/code/.ascend/cann-9.0.1/cann-9.0.1
export DSV4_RUNTIME_VENV=/data/z00569729/code/.venvs/afd-v023-vllm-cann
export DSV4_VLLM_ROOT=/data/z00569729/code/vllm-release-v0.23.0
export DSV4_VLLM_ASCEND_ROOT=/data/z00569729/code/vllm-ascend-rfc-vllm-cann
export MODEL_PATH=/data/z00569729/models/DeepSeek-V4-Flash-w8a8-mtp

export VLLM_HOST_IP="REPLACE_WITH_A3_P_IP"
export HCCL_IF_IP="${VLLM_HOST_IP}"
export GLOO_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export MC_MIN_PRC_PORT=15000
export MC_MAX_PRC_PORT=17000

export API_HOST=0.0.0.0
export API_PORT=8100
export PREFILL_DEVICES=0,1,2,3,4,5,6,7
export PREFILL_DP_SIZE=2
export PREFILL_TP_SIZE=4
export DECODE_DP_SIZE=8
export DECODE_TP_SIZE=1
export MOONCAKE_ENGINE_ID=dsv4-afd-prefill
export MOONCAKE_KV_PORT=30000
export MAX_MODEL_LEN=4096
export MAX_NUM_BATCHED_TOKENS=4096
export MAX_NUM_SEQS=16

export PD_LOG_ROOT=/data/z00569729/run/dsv4-afd-mooncake-pd/logs/manual
mkdir -p "${PD_LOG_ROOT}"
bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_pd/prefill.sh \
  > "${PD_LOG_ROOT}/prefill.log" 2>&1 &
prefill_pid=$!
printf '%s\n' "${prefill_pid}" > "${PD_LOG_ROOT}/prefill.pid"
```

### 17.7 启动 A3-D Decode A8F8

Decode FFN 与 Attention 必须前后紧邻启动。FFN 不配置 Mooncake，也没有业务
HTTP health；只有 Attention 使用 `ENABLE_PD=1` 成为 KV consumer：

```bash
cd /data/z00569729/code/afd-plugin
export DSV4_CANN_ROOT=/data/z00569729/code/.ascend/cann-9.0.1/cann-9.0.1
export DSV4_RUNTIME_VENV=/data/z00569729/code/.venvs/afd-v023-vllm-cann
export DSV4_VLLM_ROOT=/data/z00569729/code/vllm-release-v0.23.0
export DSV4_VLLM_ASCEND_ROOT=/data/z00569729/code/vllm-ascend-rfc-vllm-cann
export MODEL_PATH=/data/z00569729/models/DeepSeek-V4-Flash-w8a8-mtp

export VLLM_HOST_IP="REPLACE_WITH_A3_D_IP"
export HCCL_IF_IP="${VLLM_HOST_IP}"
export GLOO_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export MC_MIN_PRC_PORT=15000
export MC_MAX_PRC_PORT=17000

export AFD_HOST=127.0.0.1
export AFD_PORT=29761
export ATTENTION_RANKS=8
export FFN_RANKS=8
export ATTENTION_DEVICES=0,1,2,3,4,5,6,7
export FFN_DEVICES=8,9,10,11,12,13,14,15
export TENSOR_PARALLEL_SIZE=1
export EXECUTION_MODE=eager
export U_BATCHES=1
export ENABLE_MTP=0
export PREFILL_DP_SIZE=2
export PREFILL_TP_SIZE=4
export MOONCAKE_ENGINE_ID=dsv4-afd-decode
export MOONCAKE_KV_PORT=30100
export MAX_MODEL_LEN=4096

export PD_LOG_ROOT=/data/z00569729/run/dsv4-afd-mooncake-pd/logs/manual
mkdir -p "${PD_LOG_ROOT}"
API_HOST=0.0.0.0 API_PORT=8911 \
  bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_ffn.sh \
  > "${PD_LOG_ROOT}/ffn.log" 2>&1 &
ffn_pid=$!
printf '%s\n' "${ffn_pid}" > "${PD_LOG_ROOT}/ffn.pid"

sleep 2

ENABLE_PD=1 API_HOST=0.0.0.0 API_PORT=8910 \
  bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_attention.sh \
  > "${PD_LOG_ROOT}/attention.log" 2>&1 &
attention_pid=$!
printf '%s\n' "${attention_pid}" > "${PD_LOG_ROOT}/attention.pid"
```

不要等待 FFN HTTP ready 后才启动 Attention；FFN 会在 AFD/HCCL 初始化中等待
Attention。也不要向 `8911` 发送业务请求或健康检查。

### 17.8 Readiness 和启动 Proxy

从准备运行 Proxy 的节点检查两个后端：

```bash
curl -fsS --max-time 10 "http://REPLACE_WITH_A3_P_IP:8100/health"
curl -fsS --max-time 10 "http://REPLACE_WITH_A3_D_IP:8910/health"
```

在 A3-D 确认 8 个 FFN rank 都进入 connector loop：

```bash
grep -Eo 'AFD FFN EngineCore started; workers run connector loop' \
  /data/z00569729/run/dsv4-afd-mooncake-pd/logs/manual/ffn.log | wc -l
```

结果必须为 8。然后在任一能访问两端 HTTP 的节点启动 Proxy：

```bash
cd /data/z00569729/code/afd-plugin
export DSV4_RUNTIME_VENV=/data/z00569729/code/.venvs/afd-v023-vllm-cann
export DSV4_VLLM_ASCEND_ROOT=/data/z00569729/code/vllm-ascend-rfc-vllm-cann
export PREFILL_HOSTS="REPLACE_WITH_A3_P_IP"
export PREFILL_PORTS="8100"
export DECODE_HOSTS="REPLACE_WITH_A3_D_IP"
export DECODE_PORTS="8910"
export PROXY_HOST=0.0.0.0
export PROXY_PORT=9000

bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_pd/proxy.sh \
  > /data/z00569729/run/dsv4-afd-mooncake-pd/logs/manual/proxy.log 2>&1 &
proxy_pid=$!
printf '%s\n' "${proxy_pid}" \
  > /data/z00569729/run/dsv4-afd-mooncake-pd/logs/manual/proxy.pid

curl -fsS --max-time 10 http://127.0.0.1:9000/healthcheck
```

Proxy 的健康接口是 `/healthcheck`。Decode FFN 仍然没有 HTTP health。

### 17.9 跨机 PD smoke 和 golden 验收

业务请求只发送给 Proxy，不能直接请求 Decode Attention，否则会绕过 Prefill
和 Mooncake：

```bash
curl -fsS http://127.0.0.1:9000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"dsv4-afd",
    "prompt":"Please explain why deterministic validation matters.",
    "temperature":0,
    "seed":1024,
    "max_tokens":32,
    "stream":false,
    "return_token_ids":true
  }'
```

请求成功后，A3-D Attention 日志至少出现一条真实 KV transfer 成功记录：

```bash
grep -En 'KV cache transfer for request .* took .* remote_session_id' \
  /data/z00569729/run/dsv4-afd-mooncake-pd/logs/manual/attention.log
```

`remote_session_id` 中的地址必须是 A3-P 的可达 IP，不能是 `127.0.0.1` 或
`0.0.0.0`。随后使用目标栈原生 golden 对 Proxy 做 10 条 prompt、3 轮和 batch
1/8/32 验收：

```bash
export PD_VALIDATION_ROOT=/data/z00569729/validation/dsv4_afd_v023_mooncake_pd_m9_f0
mkdir -p "${PD_VALIDATION_ROOT}"

python recipe/npu/deepseek_v4/common/validate_golden.py \
  --endpoint http://127.0.0.1:9000/v1/completions \
  --model dsv4-afd \
  --golden /data/z00569729/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json \
  --rounds 3 \
  --batch-sizes 1 8 32 \
  --output "${PD_VALIDATION_ROOT}/golden.json"
```

`golden.json` 必须满足 `passed=true`、串行请求 30/30 token exact；batch 记录
用于验证请求结构和数量，不能用 smoke 文本相似替代 token IDs 比较。

再发送一个预期由客户端取消的长请求。`curl` 返回 28 表示客户端超时取消，
之后 Proxy、Prefill 和 Decode 必须仍然健康并能完成一个正常请求：

```bash
set +e
curl -fsS --max-time 1 http://127.0.0.1:9000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"dsv4-afd",
    "prompt":"Write a detailed deterministic systems validation checklist.",
    "temperature":0,
    "seed":1024,
    "max_tokens":512,
    "stream":false
  }'
cancel_rc=$?
set -e
test "${cancel_rc}" -eq 28

curl -fsS --max-time 10 http://127.0.0.1:9000/healthcheck
curl -fsS http://127.0.0.1:9000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4-afd","prompt":"Recovery check.","temperature":0,"max_tokens":8}'
```

如果请求在 1 秒内正常完成而 `cancel_rc=0`，该轮没有覆盖取消路径，应使用更长
prompt 或输出重新执行；不能把它记为取消恢复通过。

### 17.10 停止、二次启动和清理

先停止入口，再按 Decode Attention、Decode FFN、Prefill 的顺序停止：

```bash
# Proxy 所在节点
kill -TERM "$(cat /data/z00569729/run/dsv4-afd-mooncake-pd/logs/manual/proxy.pid)"

# A3-D
kill -TERM "$(cat /data/z00569729/run/dsv4-afd-mooncake-pd/logs/manual/attention.pid)"
kill -TERM "$(cat /data/z00569729/run/dsv4-afd-mooncake-pd/logs/manual/ffn.pid)"

# A3-P
kill -TERM "$(cat /data/z00569729/run/dsv4-afd-mooncake-pd/logs/manual/prefill.pid)"
```

应由保存 PID 的原 shell 分别 `wait` 并记录退出码。不要用全局 `pkill`。两端
检查：

```bash
npu-smi info
ss -ltnp | grep -E ':(8100|8910|9000|29761|3000[0-7]|3010[0-7])([[:space:]]|$)' || true
grep -En \
  'EngineCore encountered a fatal error|AFD NPU FFN worker loop failed|Mooncake transfer failed|Communication_Error|507015|Traceback' \
  /data/z00569729/run/dsv4-afd-mooncake-pd/logs/manual/*.log
```

第一次完整停止后，重新执行第 17.6-17.9 节，至少完成 Proxy health、一个 smoke
请求和一次 KV transfer 证据，再次正常停止。二次启动不能复用旧 PID、旧端口
监听或旧请求 metadata。

### 17.11 PD 安装验证通过条件和产物

双机 PD 安装只有同时满足以下条件才算完成：

1. 两端 CANN、venv、三个源码 commit、Mooncake 版本和 `.so` 指纹完全一致；
   wheel 模式还要求 wheel SHA256 一致；
2. 两端 `check_mooncake_runtime.sh` 通过且没有 CANN 9.1.0 路径；
3. 两端本地 Mooncake NPU round-trip 都为 2/2、2 MiB 逐字节一致；
4. Prefill、8 个 Attention rank、8 个 FFN loop 和 Proxy 全部 ready；
5. Proxy smoke 产生真实跨机 KV transfer，日志中的 remote IP/port 正确；
6. 10 条 golden 连续 3 轮达到 30/30 token IDs 完全一致；
7. batch 1/8/32、请求取消后的恢复、正常停止和二次启动通过；
8. 两端无 fatal 日志、无遗留端口、无遗留推理进程和 NPU 占用。

每台机器分别保存 `environment.txt`、`git.txt`、`npu_before.txt`、启动命令、
原始日志、`golden.json`、退出码和 `npu_after.txt`。推荐目录：

```text
/data/z00569729/validation/dsv4_afd_v023_mooncake_pd_m9_f0_<timestamp>/prefill/
/data/z00569729/validation/dsv4_afd_v023_mooncake_pd_m9_f0_<timestamp>/decode/
/data/z00569729/validation/dsv4_afd_v023_mooncake_pd_m9_f0_<timestamp>/proxy/
```

当前已有的
`dsv4_afd_v023_mooncake_pd_m9_contract_20260821_181148/summary.json` 状态是
`real_transfer_component_passed_f0_pending`，只能证明安装组件门禁，不能替代
本节双机实模 F0。完成上述产物并审核后，才提交 M9 验证结论并创建功能 tag。

### 17.12 回传小输出件

不需要回传第 17.11 节中的全部原始目录。三种角色分别执行 `collect`：

```bash
cd /data/z00569729/code/afd-plugin

bash tools/dsv4/mooncake_pd_manual/pd.sh collect \
  /data/z00569729/config/pd-prefill.env
bash tools/dsv4/mooncake_pd_manual/pd.sh collect \
  /data/z00569729/config/pd-decode.env
bash tools/dsv4/mooncake_pd_manual/pd.sh collect \
  /data/z00569729/config/pd-proxy.env
```

每个角色会在其 `OUTPUT_ROOT` 下生成：

```text
dsv4-m9-pd-<role>-<timestamp>.tar.gz
dsv4-m9-pd-<role>-<timestamp>.tar.gz.sha256
```

精简 openEuler 容器没有 `hostname` 命令时无需安装额外 RPM；`collect` 会依次从
`/proc/sys/kernel/hostname`、Shell `HOSTNAME`、可用的 `hostname` 或 `uname -n`
读取主机名，均不可用时记录为 `unknown`，不会中断产物收集。

将三个 `.tar.gz` 及对应 `.sha256` 发回即可。默认每包硬上限 2 MiB，三包合计
不超过约 6 MiB；通常会更小。包内只包含 commit/包版本、Mooncake 安装模式、
`.so` 指纹（wheel 模式另含 wheel SHA256）、PID 和端口状态、`npu-smi`、
runtime/round-trip 结果、每个角色日志末尾 256 KiB、
最近 50 条 KV transfer 证据、最近 200 条 fatal marker，以及 Proxy 的
smoke/golden/batch/取消恢复结果。

输出件明确不包含完整日志、模型、wheel、profiler、core dump、完整环境变量或
API key。若包超过上限，`collect` 会删除超限包并报错；此时降低配置中的
`ARTIFACT_LOG_TAIL_BYTES` 后重新收集，不要手工发送整个日志目录。
