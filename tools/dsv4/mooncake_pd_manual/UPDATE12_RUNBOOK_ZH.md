# DeepSeek-V4 Mooncake PD update12 双 A3 操作手册

本手册用于在两台 A3 上依次完成：

1. Mooncake PD no-AFD control；
2. 生成 path-matched PD control golden；
3. Mooncake PD + AFD；
4. `PD control vs PD + AFD` 的 30/30 token IDs 验证。

固定节点分工：

| 节点 | 角色 |
|---|---|
| A3-P | Prefill、Proxy、生成和保存 control golden |
| A3-D | control 完整 Decode，或 AFD Attention + FFN Decode |

## 1. 两台机器安装 update12

把以下两个文件复制到两台机器的 `/data/z00569729/packages/`：

```text
afd-plugin-dsv4-mooncake-pd-update12-20260827.tar.gz
afd-plugin-dsv4-mooncake-pd-update12-20260827.tar.gz.sha256
```

A3-P、A3-D 都执行：

```bash
cd /data/z00569729/packages
sha256sum -c afd-plugin-dsv4-mooncake-pd-update12-20260827.tar.gz.sha256
tar -xzf afd-plugin-dsv4-mooncake-pd-update12-20260827.tar.gz \
  -C /data/z00569729/code

cd /data/z00569729/code/afd-plugin
bash -n tools/dsv4/mooncake_pd_manual/pd.sh
bash tools/dsv4/mooncake_pd_manual/pd.sh --help
```

update12 是基于 afd-plugin 提交
`49bb4a1dda5f7a59dcfbb45ea36d3ad1b2b89193` 的工具 overlay。解压后 Git
工作树出现本包列出的修改属于预期；vLLM 和 vLLM-Ascend 工作树必须保持干净。

如果源码由其他 Linux 用户准备，A3-P、A3-D 都以实际运行 `pd.sh` 的用户执行：

```bash
git config --global --add safe.directory /data/z00569729/code/afd-plugin
git config --global --add safe.directory /data/z00569729/code/vllm-release-v0.23.0
git config --global --add safe.directory /data/z00569729/code/vllm-ascend-rfc-vllm-cann
```

这只信任上述三个明确路径，不要配置 `safe.directory='*'`。

## 2. 准备六个配置

A3-P 执行：

```bash
cd /data/z00569729/code/afd-plugin
mkdir -p /data/z00569729/config

cp tools/dsv4/mooncake_pd_manual/config.env.example \
  /data/z00569729/config/pd-control-prefill.env
cp tools/dsv4/mooncake_pd_manual/config.env.example \
  /data/z00569729/config/pd-control-proxy.env
cp tools/dsv4/mooncake_pd_manual/config.env.example \
  /data/z00569729/config/pd-afd-prefill.env
cp tools/dsv4/mooncake_pd_manual/config.env.example \
  /data/z00569729/config/pd-afd-proxy.env
```

A3-D 执行：

```bash
cd /data/z00569729/code/afd-plugin
mkdir -p /data/z00569729/config

cp tools/dsv4/mooncake_pd_manual/config.env.example \
  /data/z00569729/config/pd-control-decode.env
cp tools/dsv4/mooncake_pd_manual/config.env.example \
  /data/z00569729/config/pd-afd-decode.env
```

编辑六个配置中的公共值：

```bash
AFD_PD_COMMIT="49bb4a1dda5f7a59dcfbb45ea36d3ad1b2b89193"
PREFILL_IP="A3-P 的业务网卡 IP"
DECODE_IP="A3-D 的业务网卡 IP"
NIC_NAME="上述 IP 所在网卡"
CANN_ROOT="/usr/local/Ascend/cann-9.0.0"
CANN_VERSION="9.0.0"
```

如果模型或源码不在模板默认位置，同时修改 `CODE_ROOT`、`MODEL_PATH` 和
`VENV_ROOT`。两台机器必须使用相同路径布局。

按文件设置角色和模式：

| 配置 | `NODE_ROLE` | `DEPLOYMENT_VARIANT` |
|---|---|---|
| `pd-control-prefill.env` | `prefill` | `pd_control` |
| `pd-control-decode.env` | `decode` | `pd_control` |
| `pd-control-proxy.env` | `proxy` | `pd_control` |
| `pd-afd-prefill.env` | `prefill` | `pd_afd` |
| `pd-afd-decode.env` | `decode` | `pd_afd` |
| `pd-afd-proxy.env` | `proxy` | `pd_afd` |

六个配置保持以下路径一致：

```bash
NATIVE_GOLDEN_PATH="/data/z00569729/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json"
PD_CONTROL_GOLDEN_PATH="/data/z00569729/validation/dsv4_m9_pd_control/golden_results.json"
```

`PD_CONTROL_GOLDEN_PATH` 此时不存在是正常的，它将在第 4 步由 A3-P 生成。
A3-D 不需要实际拥有这个文件。

## 3. 预检和安装

A3-P 执行：

```bash
cd /data/z00569729/code/afd-plugin
bash tools/dsv4/mooncake_pd_manual/pd.sh install /data/z00569729/config/pd-control-prefill.env
bash tools/dsv4/mooncake_pd_manual/pd.sh check /data/z00569729/config/pd-control-prefill.env
bash tools/dsv4/mooncake_pd_manual/pd.sh check /data/z00569729/config/pd-control-proxy.env
```

A3-D 执行：

```bash
cd /data/z00569729/code/afd-plugin
bash tools/dsv4/mooncake_pd_manual/pd.sh install /data/z00569729/config/pd-control-decode.env
bash tools/dsv4/mooncake_pd_manual/pd.sh check /data/z00569729/config/pd-control-decode.env
bash tools/dsv4/mooncake_pd_manual/pd.sh check /data/z00569729/config/pd-afd-decode.env
```

任一 `check` 失败都先停止。特别是 CANN 路径、提交、Mooncake 库指纹、本机
round-trip、网卡 IP 或 NPU 残留不通过时，不要启动服务。

update12 的本机 round-trip 使用配置中的本机业务 IP 和 `NIC_NAME`，与正式
Mooncake 数据路径一致；不再硬编码 `127.0.0.1`。成功 JSON 中的 `host`、
`interface` 必须等于当前节点配置，并包含 `transfer_results=[0,0]`。

## 4. 运行 PD no-AFD control 并生成 golden

A3-P 执行：

```bash
cd /data/z00569729/code/afd-plugin
bash tools/dsv4/mooncake_pd_manual/pd.sh start /data/z00569729/config/pd-control-prefill.env
```

A3-D 执行：

```bash
cd /data/z00569729/code/afd-plugin
bash tools/dsv4/mooncake_pd_manual/pd.sh start /data/z00569729/config/pd-control-decode.env
```

A3-P 执行：

```bash
cd /data/z00569729/code/afd-plugin
bash tools/dsv4/mooncake_pd_manual/pd.sh start /data/z00569729/config/pd-control-proxy.env
bash tools/dsv4/mooncake_pd_manual/pd.sh status /data/z00569729/config/pd-control-prefill.env
bash tools/dsv4/mooncake_pd_manual/pd.sh status /data/z00569729/config/pd-control-proxy.env
bash tools/dsv4/mooncake_pd_manual/pd.sh record-control /data/z00569729/config/pd-control-proxy.env
```

A3-D 同时执行状态检查：

```bash
cd /data/z00569729/code/afd-plugin
bash tools/dsv4/mooncake_pd_manual/pd.sh status /data/z00569729/config/pd-control-decode.env
```

A3-P 检查生成结果：

```bash
test -s /data/z00569729/validation/dsv4_m9_pd_control/golden_results.json
jq '{passed,rounds,prompt_count,mismatched_prompt_indices,metadata,reference_comparison}' \
  /data/z00569729/validation/dsv4_m9_pd_control/golden_results.json
```

必须满足 `passed=true`、`rounds=3`、`prompt_count=10`，并且
`mismatched_prompt_indices=[]`。`reference_comparison` 是 one-shot native 与 PD
control 的信息性结果，允许不是 30/30。

## 5. 停止 control

A3-P：

```bash
cd /data/z00569729/code/afd-plugin
bash tools/dsv4/mooncake_pd_manual/pd.sh collect /data/z00569729/config/pd-control-proxy.env
bash tools/dsv4/mooncake_pd_manual/pd.sh collect /data/z00569729/config/pd-control-prefill.env
bash tools/dsv4/mooncake_pd_manual/pd.sh stop /data/z00569729/config/pd-control-proxy.env
```

A3-D：

```bash
cd /data/z00569729/code/afd-plugin
bash tools/dsv4/mooncake_pd_manual/pd.sh collect /data/z00569729/config/pd-control-decode.env
bash tools/dsv4/mooncake_pd_manual/pd.sh stop /data/z00569729/config/pd-control-decode.env
npu-smi info
```

A3-P：

```bash
cd /data/z00569729/code/afd-plugin
bash tools/dsv4/mooncake_pd_manual/pd.sh stop /data/z00569729/config/pd-control-prefill.env
npu-smi info
```

确认两台机器无推理进程后再继续。

## 6. 运行 PD + AFD 并验证

A3-P：

```bash
cd /data/z00569729/code/afd-plugin
bash tools/dsv4/mooncake_pd_manual/pd.sh check /data/z00569729/config/pd-afd-prefill.env
bash tools/dsv4/mooncake_pd_manual/pd.sh check /data/z00569729/config/pd-afd-proxy.env
bash tools/dsv4/mooncake_pd_manual/pd.sh start /data/z00569729/config/pd-afd-prefill.env
```

A3-D：

```bash
cd /data/z00569729/code/afd-plugin
bash tools/dsv4/mooncake_pd_manual/pd.sh check /data/z00569729/config/pd-afd-decode.env
bash tools/dsv4/mooncake_pd_manual/pd.sh start /data/z00569729/config/pd-afd-decode.env
bash tools/dsv4/mooncake_pd_manual/pd.sh status /data/z00569729/config/pd-afd-decode.env
```

A3-P：

```bash
cd /data/z00569729/code/afd-plugin
bash tools/dsv4/mooncake_pd_manual/pd.sh start /data/z00569729/config/pd-afd-proxy.env
bash tools/dsv4/mooncake_pd_manual/pd.sh validate /data/z00569729/config/pd-afd-proxy.env
```

`validate` 会先检查 control golden 内的 CANN、提交、模型、DP/TP、eager/U1 和
MTP-off 元数据，再执行 10 条 prompt x 3 轮、batch 1/8/32 和请求取消恢复。
必须得到 path-matched `PD control vs PD + AFD = 30/30`。

## 7. 收集、停止和回传

A3-P：

```bash
cd /data/z00569729/code/afd-plugin
bash tools/dsv4/mooncake_pd_manual/pd.sh collect /data/z00569729/config/pd-afd-proxy.env
bash tools/dsv4/mooncake_pd_manual/pd.sh collect /data/z00569729/config/pd-afd-prefill.env
bash tools/dsv4/mooncake_pd_manual/pd.sh stop /data/z00569729/config/pd-afd-proxy.env
```

A3-D：

```bash
cd /data/z00569729/code/afd-plugin
bash tools/dsv4/mooncake_pd_manual/pd.sh collect /data/z00569729/config/pd-afd-decode.env
bash tools/dsv4/mooncake_pd_manual/pd.sh stop /data/z00569729/config/pd-afd-decode.env
npu-smi info
```

A3-P：

```bash
cd /data/z00569729/code/afd-plugin
bash tools/dsv4/mooncake_pd_manual/pd.sh stop /data/z00569729/config/pd-afd-prefill.env
npu-smi info
```

回传两种模式生成的 `.tar.gz` 和 `.sha256`。文件名会明确包含
`pd-control` 或 `pd-afd`，每包默认不超过 2 MiB。不要回传完整日志、模型、wheel、
profiler 或 core dump。

## 8. 判定口径

- `native vs PD control`：信息性语义参考，不作为 AFD 基础设施硬门禁；
- `PD control vs PD + AFD`：必须 30/30，是本阶段的正确性硬门禁；
- control Decode 日志出现 AFD model/connector marker：直接失败；
- AFD Decode 缺少 8 个 FFN loop 或 AFD marker：直接失败；
- CANN、提交、模型或拓扑元数据不同：拒绝比较，重新生成 control golden。
