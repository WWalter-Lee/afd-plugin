# DeepSeek-V4 AFD HCCL P2P Graph/U2 验证报告

## 1. 结论

标准 HCCL P2P connector 已在 vLLM 0.23 + vLLM-Ascend
`rfc/vllm_cann` 目标栈完成 A8F8 Graph/U2 功能闭环。当前支持范围严格限定为：

- `P2pHcclAFDConnector`；
- A8F8，Attention/FFN rank 数量相等；
- `FULL_DECODE_ONLY`；
- 两个 microbatch；
- MTP、PD、sequence parallel、Attention gate 关闭；
- hidden、FFN output 和 input IDs 均使用 HCCL P2P，不调用 CAMP2P 自定义传输 op。

完整 F0 在两次独立冷启动中均达到 30/30 golden token IDs 一致，batch
1/8/32、Graph capture/replay、U2 双 stage、正常停止、fatal 日志和 NPU 清理均通过。

固定口径 P1 为 128/128 请求成功，output throughput 107.189 token/s，p50
TPOT 217.268 ms。该数据是单轮轻量 guard，只能说明没有灾难性回退并出现强性能候选
信号；它不是三轮正式收益结论，也不关闭 `P8D-PERF-001`。

## 2. 背景和问题

P8D 已将 eager/U2 的两个 Python stage 线程收敛为单线程
`layer -> stage 0 -> stage 1`，但 eager P1 仍只有 16.472 token/s。下一功能阶段要在不引入
异步 HCCL 的前提下支持 Graph/U2，使两个 microbatch 的 decoder 执行、HCCL hidden/output
传输和 stage 顺序进入同一个 FULL ACL Graph。

初始实现沿用 Graph/U1 的 stage-major 捕获，每个 stage 由一个 Python 线程执行；FFN 侧却按
layer-major 顺序收发。两侧 HCCL communicator op 顺序不一致，导致 capture timeout，且通信
stream 没有正确加入 capture stream。另一个独立问题是 DSV4 DSA compressed attention 被误判为
普通 MLA full-graph，触发了不适用的 FIA workspace 路径。

因此本阶段不是简单删除 Graph/U2 fail-fast，而是统一 Graph warmup、capture 和 FFN 的通信顺序，
并明确 DSV4 sparse/compressed attention 的 Graph 能力边界。

## 3. 固定环境

| 项目 | 固定值 |
|---|---|
| 日期 | 2026-08-20 |
| 基础提交 | `1d8db03f6c59c224b14204251bed5169bec31bc2` |
| 开发分支 | `feat/dsv4-afd-hccl-graph-u2` |
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| venv | `/mnt/workspace/code/.venvs/afd-v023-vllm-cann` |
| vLLM | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，`releases/v0.23.0` |
| vLLM-Ascend | `3da28f9414583d2d0b672a8f06d1fae142404bda`，`rfc/vllm_cann` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| golden | `/mnt/workspace/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json` |

拓扑为 Attention NPU 0-7、FFN NPU 8-15、DP8/TP1/EP8，seed 1024、temperature
0。性能 guard 关闭 vLLM async scheduling，通信仍使用同步 `send/recv`。

## 4. 关键实现和原因

### 4.1 Graph capture 改为单线程 layer-major

Graph/U2 的 warmup 和真实 capture 都调用 DSV4 的
`forward_ubatches_layer_major()`，在一个 host 线程内按每层 stage 0、stage 1 的顺序推进。两个
stage 被合并为一个 `AscendModelOutput` 并绑定到同一个 NPUGraph。

这样做的原因是 HCCL P2P op 顺序必须在 Attention、FFN、warmup 和 capture 之间完全一致。它也
消除了 stage-major 两线程在 capture 期间的 GIL 交接和 communicator 排序不确定性。

### 4.2 capture 期间使用 graph-visible HCCL op

编译或真实 Graph capture 时，connector 使用 torch-npu 注册的 HCCL `_send/_recv` op；普通
eager 和 graph 外路径继续调用 `torch.distributed.send/recv`。真实 capture 状态也被纳入判断，
避免 Dynamo trace 结束后回退到不可捕获的 Python wrapper。

Attention 的 connector-owned stream/event 在 layer-major capture 中显式汇合到 Graph 主
compute stream，保证 Graph 包含所需通信依赖。这里没有使用 `isend/irecv`、后台通信线程或
`torch.ops.vllm.afd_camp2p_send_attn_output()`。

### 4.3 子 forward context 保留父 Graph 状态

U2 子 context 的 `cudagraph_runtime_mode` 会显示为 `NONE`。实现增加
`afd_graph_ubatching` 标记保存父 FULL Graph 状态，避免 connector 把 Graph 子 forward 错当成
普通 eager。编译期判断放在 stream/event 状态读取之前，避免 Dynamo 检查 Python dict 和 event
对象。

### 4.4 DSV4 compressed attention 不进入普通 MLA FIA workspace

runner 同时检查 `index_topk`、`use_sparse` 和 `use_compress`，识别 DSV4 DSA
sparse/compressed attention。该模型仍执行正常 ACL Graph capture，但跳过只适用于普通 MLA 的
full-graph FIA workspace 分配，避免错误的 workspace 需求阻塞启动。

### 4.5 Fail-fast 边界

只为 `P2pHcclAFDConnector` 解除 Graph/U2 门禁。以下配置继续启动前拒绝：

- CAMP2P Graph/U2；
- Graph/U3；
- Graph 下 Attention/FFN rank 数量不相等；
- 非 `FULL_DECODE_ONLY` Graph；
- MTP/U2、PD、Attention gate、sequence parallel；
- TP、PP、CP 或 DCP 大于 1。

## 5. 自动化验证

固定目标运行栈执行了以下相关回归，共 328 项，全部通过：

```text
tests/unit/v1/worker/test_npu_runtime.py
tests/unit/v1/worker/test_npu_mla_graph.py
tests/unit/v1/worker/test_cuda_graph.py
tests/unit/v1/worker/test_attention_model_runner.py
tests/unit/connectors/test_p2p_hccl_connector.py
tests/unit/model_executor/models/test_deepseek_v4_construction.py
tests/e2e/test_dsv4_recipe.py
```

覆盖范围包括 feature accept/reject、Graph transport 选择、编译期分支、子 context、Graph
layer-major capture、DSV4 sparse MLA 判断、recipe 门禁以及 eager/U1/U2 既有路径回归。Ruff 和
`git diff --check` 通过。

## 6. A8F8 F0 结果

正式产物：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_f0_20260820
```

| 门禁 | Cycle 1 | Cycle 2 |
|---|---:|---:|
| startup | 414.229 s | 400.233 s |
| 10 prompt x 3 轮 | 30/30 token exact | 30/30 token exact |
| batch 1/8/32 | valid | valid |
| U2 双 stage | observed | observed |
| fatal log | 0 | 0 |
| shutdown | 两侧 rc=0 | 两侧 rc=0 |
| NPU cleanup | 无运行进程 | 无运行进程 |

两轮 batch 的诊断项均为 1/1、3/8、1/32 token exact。并发 DP 调度不保证逐样本 token
确定性，所以 batch 门禁检查返回数量、结构和生成有效性；严格 token 一致性由每轮 30 个串行
请求负责。该诊断值在报告中保留，不能改写成 batch 全量 token exact。

Graph capture size 1/2/4/8 全部完成，Attention 各 rank 额外 Graph HBM 约
1.22-1.24 GiB。正式 F0 前的 smoke 也完成单请求精确匹配和 batch8 有效性：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_smoke_20260820_1200
```

## 7. P1 轻量性能 guard

产物：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_p1_20260820
```

固定 A8F8、C32、输入 1024、精确输出 128、128 请求，16 条短请求预热，只测一轮：

| 指标 | Graph/U2 P1 |
|---|---:|
| 请求成功 | 128/128 |
| output throughput | 107.189 token/s |
| request throughput | 0.837 req/s |
| p50 TTFT | 11586.782 ms |
| p50 TPOT | 217.268 ms |
| p99 TPOT | 314.613 ms |
| output token/s/NPU | 6.699 |
| Attention 最大 HBM | 61,819 MiB |
| FFN 最大 HBM | 44,391 MiB |

相对 async-scheduling-off eager/U1 单点 30.615 token/s 为 +250.116%，相对 eager/U2
P8D 单点 16.472 token/s 为 +550.748%。这两个比较证明 P1 没有触发 20% 回退暂停条件，但
Graph/U2 同时改变了执行模式，不能把差值全部归因于 microbatch，也不能替代 Graph/U1 和同预算
native Graph 的公平对照。

P1 仅一轮，未采 profiler。`P8D-PERF-001` 继续 Open；只有完成 Graph/U1、Graph/U2 和同预算
native Graph 的三轮 P2、波动门禁及必要 profile 后，才能宣称 AFD + microbatch 有性能收益。

## 8. 下一步

本阶段通过 F0 + P1，可以作为 Graph/U2 功能候选提交并冻结功能 tag，但不能命名为性能
baseline。按“先功能、后统一性能”的路线，下一功能阶段是等量 A8F8 的 HCCL P2P MTP/U2：

1. 先完成 eager/U2 + MTP 的 phase、target hidden、draft/verify 和双 stage 消息契约；
2. 通过独立 F0 和一次 P1；
3. 再评估 target Graph/U2 + draft eager MTP，不直接开放 full draft Graph；
4. 功能组合闭环后统一执行 Graph/U1、Graph/U2、native Graph 的 P2 正式矩阵。

Graph/U3、Graph 非等量、PD 和 A5 实机适配继续保持独立里程碑。
