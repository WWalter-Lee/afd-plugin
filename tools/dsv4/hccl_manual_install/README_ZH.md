# DeepSeek-V4 AFD HCCL P2P 手工安装包

该包用于把固定的 vLLM 0.23、vLLM-Ascend `rfc/vllm_cann` 和
`P2pHcclAFDConnector` 部署到另一台 Atlas A3。生成的 transfer archive 包含：

- vLLM 固定源码；
- vLLM-Ascend 固定源码及其递归 submodule；
- afd-plugin 固定 MTP M1 release tag 源码；
- 主机检查、建 venv、安装依赖、构建、验收、启停和请求 smoke 脚本；
- 每个文件的 SHA256 和精确源码 manifest。

Python 依赖 wheel 不默认放入包中。目标机可联网安装，或自行准备完整
wheelhouse 后设置 `OFFLINE=1`。

## 1. 解包和配置

```bash
tar -xzf dsv4-afd-hccl-manual-install-*.tar.gz
cd dsv4-afd-hccl-manual-install-*
vi config.env
```

解包前可在 transfer archive 所在目录校验外层文件：

```bash
sha256sum -c dsv4-afd-hccl-manual-install-*.tar.gz.sha256
```

必须修改：

- `CANN_ROOT`：目标机唯一使用的 CANN 9.0.1；
- `MODEL_PATH`：DeepSeek-V4-Flash W8A8 模型路径；
- `PYTHON_BIN`：Python 3.12；
- `SOC_VERSION`：目标机真实 SoC；
- `NIC_NAME` 和 `HCCL_IF_IP`；
- 安装、源码、日志目录。

不要把验证机的 IP 复制到其他机器，也不要在已经 source CANN 9.1.0 的
shell 中直接继续。

## 2. 校验包

```bash
sha256sum -c manifest/SHA256SUMS
bash bin/00_print_config.sh
bash bin/01_preflight.sh
```

## 3. 分步安装

```bash
bash bin/02_prepare_sources.sh
bash bin/03_create_venv.sh
bash bin/04_install_python_deps.sh
bash bin/05_install_stack.sh
bash bin/06_verify_install.sh
```

也可以一次执行安装阶段：

```bash
bash bin/install_all.sh
```

这些脚本默认拒绝复用非空源码目录或已有 venv。确认目标正确后，分别设置
`REUSE_SOURCES=1` 或 `REUSE_VENV=1`，不要通过删除未知目录绕过检查。

## 4. 离线安装

源码已在 transfer archive 中。Python wheel 需要在同架构、同 Python
版本的联网机器上预先下载到一个目录，并随包拷贝到目标机。配置：

```bash
OFFLINE="1"
WHEELHOUSE="/path/to/wheelhouse"
```

离线脚本会强制使用 `--no-index --find-links`，缺少任何直接或间接依赖时
立即失败，不会静默访问公网。

## 5. 启动和停止

默认配置启动已验证的 A8F8 Graph/U1：

```bash
bash bin/07_start.sh
bash bin/08_status.sh
bash bin/10_smoke_request.sh
bash bin/09_stop.sh
```

`07_start.sh` 先启动 FFN，2 秒后启动 Attention，并等待 Attention health；
不会探测 FFN HTTP。停止顺序固定为 Attention 后 FFN。

运行 eager/U1 + MTP M1 时修改：

```bash
EXECUTION_MODE="eager"
U_BATCHES="1"
ENABLE_MTP="1"
MTP_NUM_SPECULATIVE_TOKENS="1"
```

MTP M1 只支持等量 A8F8、eager/U1、一个 MTP layer 和一个 speculative token。

## 6. 当前 MTP 源码状态

正式安装包固定使用 `dsv4-afd-v023-hccl-mtp-m1-v1`。打包器要求该 tag
存在、当前 HEAD 与 tag 一致且工作树干净；不接受未提交的 runtime overlay。
部署或审计时以 `manifest/versions.env` 和 `manifest/SHA256SUMS` 为准。

## 7. 运行边界

- HCCL Connector 不需要 `pip install hccl`；
- HCCL-only 不构建 afd-plugin CAMP2P custom ops；
- vLLM-Ascend `custom_transformer` ops 仍必须构建和 source；
- Graph 只支持 U1 和等量 A/F；
- MTP M1 只支持 eager/U1 和等量 A8F8；
- 业务流量只进入 Attention API；
- 启动成功必须同时满足 Attention health 和全部 FFN connector loop ready。
