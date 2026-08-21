# DeepSeek-V4 AFD HCCL P2P 非等量 Graph 组件验证报告

## 1. 结论

在固定 vLLM 0.23 + vLLM-Ascend `rfc/vllm_cann` 目标栈上，
`P2pHcclAFDConnector` 已完成 `A = k x F` 非等量拓扑的 target
`FULL_DECODE_ONLY` Graph U1/U2 组件级功能闭环。

A2F1 和 A4F2 均通过真实 NPU HCCL 的两 stage capture/replay；A4F2 还通过了
target Graph + eager draft MTP 的组合组件。input IDs 保持在 Graph 外，hidden 和
FFN output 在 NPUGraph 内按多 peer slice 收发，所有 worker 返回码为 0 且正常 close。

本结果是 component functional snapshot，不是 A8F4 实模 E2E 或性能基线。A3 的
64 GiB HBM 无法完成 FFN EP4 模型构造，因此 A8F4 golden、batch、生命周期和性能门禁
保留到高 HBM A5。

## 2. 固定环境

| 项目 | 固定值 |
|---|---|
| 日期 | 2026-08-21 |
| 基础提交 | `4aafb101f75b37912f2074a68be7d410fc96d3de` |
| 开发分支 | `feat/dsv4-afd-hccl-graph-unequal` |
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| venv | `/mnt/workspace/code/.venvs/afd-v023-vllm-cann` |
| vLLM | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `3da28f9414583d2d0b672a8f06d1fae142404bda` |
| torch-npu | `2.10.0.post2` |

通信仍使用标准 HCCL P2P。没有使用 CAMP2P 自定义 op、`isend/irecv`、后台通信线程
或新增 connector。

## 3. 关键实现

### 3.1 FFN Graph key 保留精确 peer layout

非等量拓扑下，一个 FFN rank 会捕获多个 Attention peer 的 `_recv/_send`。旧 key 只保存
每个 FFN rank 的聚合 token 数，例如以下两个 A4F2 layout 都会得到 `[5, 9]`：

```text
[2, 3, 4, 5]
[1, 4, 3, 6]
```

但两者的 peer slice shape 不同，不能复用同一个通信图。新 key 保存扩展后的 A 长度精确
token layout；TP 场景仍先把 DP count 复制到对应 TP workers。这样相同聚合、不同 peer
分布会捕获或命中不同 Graph，避免错误 shape replay。

### 3.2 多 peer Graph 收发

connector 已有的 deterministic subgroup mapping 和 `peer_slices` 被直接复用。FFN capture
按 Attention role rank 升序接收所有 peer hidden，执行合并计算后按相同 slice 顺序返回。
U2 的两个 stage 使用各自 HCCL group，但进入同一个 layer-major Graph 顺序。

### 3.3 Graph op 注册前置

torch-npu 2.10.0.post2 的 `npu_define::_send/_recv` 由 `npugraph_ex` 模块注册。完整服务通常
由编译器先加载该模块，但独立组件暴露了这一隐式顺序。Graph 配置下，connector 现在在创建
process group 前显式完成注册并校验两个 op；eager 不增加该初始化成本。

### 3.4 IDs 和 MTP 边界

input IDs 仍在 capture/replay 前通过一次性 int32 HCCL side channel 传输，不进入动态 shape
Graph。普通 decoder Graph replay 完成后，MTP 继续走现有 eager header/hidden/output phase。
本阶段没有开启 full draft ACL Graph，也没有改变 MTP header 协议。

## 4. 验证结果

### 4.1 CPU/Mock

相关回归覆盖：

- 同聚合、不同 peer token layout 使用不同 FFN Graph key；
- A2F1 Graph 和 target Graph + eager MTP 通过 feature validation；
- recipe 和角色脚本允许 P2p HCCL 的整数倍 Graph；
- FFN capture 对每个 peer 使用精确 recv/send slice；
- CAMP2P 等量限制、Graph U3、Attention-side gate 和非法 A/F 继续 fail-fast。

固定 CANN 9.0.1 环境下，Graph helper、connector 和 recipe 相关测试共 137 项通过；
DSV4 runtime feature/key 专项 37 项通过。与既有 Graph/U2 基线相同范围的 NPU runtime、
MLA Graph、Attention runner、connector、DSV4 construction 和 recipe 完整回归共 356 项通过；
Ruff、compileall、脚本语法和固定运行栈检查通过。

### 4.2 真实 NPU HCCL NPUGraph

每组先运行两个不同 eager step，再捕获两个 stage，并在修改静态 hidden 值后 replay。
capture 本身只作为记录动作；正确性检查读取 replay 输出，符合生产 Graph 使用语义。

| 拓扑 | 配置 | 结果 | 关键覆盖 |
|---|---|---|---|
| A2F1 | A:0,1；F:8 | passed | 双 peer fan-in/out、两 stage、Graph 外 IDs、更新输入后 replay |
| A4F2 | A:0-3；F:8,9 | passed | 两个独立 subgroup、不同 peer shape、六进程 close |
| A4F2 + MTP | 同上 | passed | target Graph capture/replay + 每 step eager MTP phase |

正式通过产物：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_unequal_a2f1_20260821_m6_retry3
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_unequal_a4f2_20260821_m6
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_mtp_unequal_a4f2_20260821_m6
```

三组 `summary.json` 均为 `passed: true`，全部 worker exit code 为 0，全部 connector 和
组件默认 process group 均已关闭。

### 4.3 组件工具修正记录

前三次 A2F1 尝试依次暴露了独立工具与生产服务的差异：Graph op 注册模块未加载、缺少默认
distributed group registry，以及把 capture 当成一次 replay 读取结果。对应修复是提前注册 op、
仅在 Graph 组件中建立默认 Gloo registry，并把门禁改为 capture 成功后更新输入再 replay。
这些失败产物被保留，没有覆盖正式通过目录。

## 5. 支持边界和下一步

当前代码支持 `P2pHcclAFDConnector`、`A >= F`、`A % F == 0`、target
`FULL_DECODE_ONLY` Graph U1/U2，以及可选的 eager draft MTP。以下能力仍未由本阶段支持：

- Graph U3；
- full draft ACL Graph；
- `A < F` 或非整数 A/F；
- 多 speculative token；
- TP、PP、SP、CP、DCP 大于当前基线；
- Mooncake PD 和 Attention-side gate。

下一功能阶段是 full draft ACL Graph。先在等量 A8F8 定位已有 6/30 golden 一致的根因，
达到完整 F0 后再复用本阶段的精确 peer Graph key 扩展到非等量组件。A8F4 实模仍需在 A5
完成，M6 不创建性能 tag，也不关闭 `P8D-PERF-001`。
