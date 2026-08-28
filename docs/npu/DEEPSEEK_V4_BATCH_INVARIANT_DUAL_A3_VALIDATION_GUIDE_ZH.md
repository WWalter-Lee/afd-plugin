# DeepSeek-V4 Batch Invariance 双 A3 验证指南

> 状态：`deferred_after_functional_development`。本专项延后到全部计划功能开发和
> 本机 F0-local 完成后执行；它不是当前 `afd-plugin` 开发或 M9 组合代码的前置
> 门禁。F0 暂不包含 golden，本指南用于后续 F1 正确性冻结。

## 1. 目标和边界

本验证用于确认 vLLM-Ascend `rfc/vllm_cann` 的 batch-invariant 修复在两台 A3
上是否能够消除同一执行路径内的跨轮、跨冷启动漂移。它不启用 Mooncake，也不
启用 AFD，因此结果不作为 PD + AFD 功能验收；它只关闭或保留上游确定性问题。

两台机器分别运行相同的 direct Prefill：NPU0-7、DP2/TP4/EP8、eager。每台机器
连续执行两次冷启动，每次运行 10 条 prompt x 3 轮。两台机器必须使用同一模型、
CANN 9.0.0、代码提交、补丁、OPP 和 Python wheel。

本地已验证的补丁解决两处 vLLM-Ascend 适配问题：

1. DeepSeek-V4 HC head 对三维张量 `dim=1` 求和，而 batch-invariant ReduceSum
   只支持最后一维；适配层先移动目标轴，算完后恢复 `keepdim` 位置。
2. 自定义 ReduceSum 不能注册到无 `dim` 参数的 `aten::sum` 默认重载；该错误
   会破坏 `torch.repeat_interleave`。补丁取消 dispatcher 覆盖，保留显式
   `torch.sum` 适配。

## 2. 交付物

小包：

```text
afd-plugin-dsv4-batch-invariant-dual-a3-20260828.tar.gz
```

包内包含：

- `bi.sh`：补丁、安装、预检、两次冷启动及收集入口；
- `config.env.example`：每台机器一份配置；
- `patches/vllm-ascend-dsv4-batch-invariant-3da28f9.patch`；
- `packages/batch_invariant_ops-1.0.0-cp312-cp312-linux_aarch64.whl`；
- `compare_batch_invariant_runs.py`。

官方 A3 OPP 约 105 MiB，不放入小包。文件名与校验值为：

```text
cann-ops-batch_invariant-A3-1.0.0-linux.aarch64.run
SHA256 9fc692978e9420336e3fea03a92c2a85df1b50a65a7df50173e3bf8bedaea70e
```

脚本也可以从 vLLM-Ascend 固定下载地址获取该文件。

## 3. 两台机器共同准备

以下命令在 A3-P 和 A3-D 都执行。先把小包放到
`/data/z00569729/packages/`：

```bash
cd /data/z00569729/packages
mkdir -p dsv4-bi-dual-a3
tar -xzf afd-plugin-dsv4-batch-invariant-dual-a3-20260828.tar.gz \
  -C dsv4-bi-dual-a3
cd dsv4-bi-dual-a3

cp config.env.example config.env
```

编辑当前节点的 `config.env`，只需要先确认这三项：

```bash
NODE_LABEL="A3-P"                 # Decode 节点改成 A3-D
LOCAL_IP="7.150.2.43"            # 当前节点 eth0 的实际 IP
NIC_NAME="eth0"
```

其余默认路径对应当前双 A3 部署：

```text
vLLM        /data/z00569729/code/vllm-release-v0.23.0
vLLM-Ascend /data/z00569729/code/vllm-ascend-rfc-vllm-cann
afd-plugin  /data/z00569729/code/afd-plugin
CANN        /usr/local/Ascend/cann-9.0.0
模型        /data/z00569729/models/DeepSeek-V4-Flash-w8a8-mtp
```

若 OPP 已手工拷贝到配置中的 `BATCH_INVARIANT_OPP_RUN`，无需下载；否则执行：

```bash
bash bi.sh download-opp config.env
```

然后在当前节点依次执行：

```bash
bash bi.sh print-config config.env
bash bi.sh apply-patch config.env
bash bi.sh install config.env
bash bi.sh check config.env
```

`apply-patch` 不设置全局 `safe.directory`，而是对每次 Git 调用使用固定仓库路径。
补丁后 vLLM-Ascend 工作树出现且只出现下面两个修改是预期状态：

```text
 M tests/ut/test_batch_invariant.py
 M vllm_ascend/batch_invariant.py
```

`check` 必须得到 `12 passed`、
`HAS_ASCENDC_BATCH_INVARIANT=True` 和 `reduce/repeat smoke passed=True`。

## 4. 每台机器执行两次冷启动

先停止该节点已有的 vLLM 服务，确认 NPU0-7、端口 8930、29650、29651 和
HCCL 基础端口 50000 可用。脚本只管理自己启动的进程组，不会执行 `pkill`。

先在 A3-P 执行：

```bash
cd /data/z00569729/packages/dsv4-bi-dual-a3
bash bi.sh run-twice config.env
```

A3-P 完成并停止后，在 A3-D 使用其自己的 `config.env` 执行同一命令：

```bash
cd /data/z00569729/packages/dsv4-bi-dual-a3
bash bi.sh run-twice config.env
```

每次启动都必须在服务日志中出现至少 8 条：

```text
Enabling batch-invariant mode
```

且不得出现：

```text
backend unavailable
```

## 5. 验收标准

每个节点单独满足：

- Start 1 的 10 x 3 轮内稳定，`passed=true`；
- Start 2 的 10 x 3 轮内稳定，`passed=true`；
- 两次启动逐请求 token IDs 为 `30/30` 完全一致；
- 服务停止后无脚本所属进程和 NPU 占用残留。

两节点结果合并后还要比较 A3-P Start 1 与 A3-D Start 1，要求同一当前执行栈
`30/30` 一致。与旧 native golden 的 `21/30` 等结果只记录为历史路径差异，
不作为当前确定性门禁；batch invariance 保证同路径稳定，不负责恢复另一历史执行
路径的 token IDs。

若双机驱动为 26.0.rc1，而本地通过环境为 25.5.5，本次结果标记为“非同驱动栈
复验”。这不妨碍判断两台双 A3 是否相互稳定，但不能仅据此排除或确认驱动根因。

## 6. 回传文件

`run-twice` 会在结果目录生成不超过 2 MiB 的包：

```text
/data/z00569729/validation/dsv4_batch_invariant_dual_a3/
  A3-P-<timestamp>/A3-P-batch-invariant-support.tar.gz
  A3-D-<timestamp>/A3-D-batch-invariant-support.tar.gz
```

只需回传这两个 `tar.gz`。包内只有两份 golden JSON、跨启动汇总、环境和提交
指纹、`npu-smi` 结果及截尾日志，不包含模型、完整服务日志、wheel、OPP 或 profiler。

如果 `run-twice` 中断，可在对应节点重新生成收集包：

```bash
bash bi.sh collect config.env
```

## 7. direct Prefill 通过后的 PD 配置

两台机器的 direct Prefill 都通过后，保留补丁、独立 venv 和 OPP。在六份
Mooncake PD 配置中统一设置：

```bash
VENV_ROOT="/data/z00569729/code/.venvs/afd-v023-vllm-cann-batch-invariant"
VLLM_ASCEND_WORKTREE_MODE="batch_invariant_patch"
ENABLE_BATCH_INVARIANT="1"
BATCH_INVARIANT_OPP_ROOT="/data/z00569729/code/.ascend/custom-opp/batch-invariant-a3-1.0.0"
```

新版 `pd.sh` 只接受本指南补丁的两文件状态和固定 SHA256；出现第三个上游修改会
立即失败。角色激活时会在清理旧 CANN/OPP 路径后重新加载该固定 OPP，`check`
还会要求 `HAS_ASCENDC_BATCH_INVARIANT=True`。control golden metadata 包含
`batch_invariant=1`，因此未启用和已启用的结果不能误作路径匹配对照。
