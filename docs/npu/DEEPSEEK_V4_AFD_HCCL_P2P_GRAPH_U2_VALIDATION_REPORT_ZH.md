# DeepSeek-V4 AFD HCCL P2P Graph/U2 验证报告

## 1. 结论

标准 HCCL P2P connector 已在 vLLM 0.23 + vLLM-Ascend `rfc/vllm_cann` 目标栈完成 A8F8 Graph/U2 功能闭环。当前支持范围严格限定为：

- `P2pHcclAFDConnector`；
- A8F8，Attention/FFN rank 数量相等；
- `FULL_DECODE_ONLY`；
- 两个 microbatch；
- MTP、PD、sequence parallel、Attention gate 关闭；
- hidden、FFN output 和 input IDs 均使用 HCCL P2P，不调用 CAMP2P 自定义传输 op。

完整 F0 在两次独立冷启动中均达到 30/30 golden token IDs 一致，batch 1/8/32、Graph capture/replay、U2 双 stage、正常停止、fatal 日志和 NPU 清理均通过。

固定口径 P1 为 128/128 请求成功，output throughput 107.189 token/s，p50 TPOT 217.268 ms。该数据是单轮轻量 guard，只能说明没有灾难性回退并出现强性能候选信号；它不是三轮正式收益结论，也不关闭 `P8D-PERF-001`。

2026-08-29 在上述功能基线上完成了 Graph/U2 显式多流增量优化：Graph-visible HCCL仍在原始 capture stream，Attention 和 FFN 的模型计算移到 event 连接的 side compute stream。A8F8 组件、实模 F0 和单轮 C32 profile 均通过；设备时间线按 43 层精确观察到每 step 43 组预期重叠。同提交三轮公平对照中，优化后 U2 为 `139.300 token/s`、CV `5.635%`，相对优化前 U2 均值 `+11.217%`，相对稳定 Graph/U1 `+3.284%`。由于优化前 U2 的 CV 为 `10.628%`、未通过 10% 稳定性门槛，该增量仍标记为“功能通过、性能未闭环”，尚未创建性能 tag。

> 说明：2026-08-31 工作树新增了 stage-local `recv_done -> 下一层 compute` 混合 DAG
> 候选及物理 stream plan。候选已通过 A8F8 单轮 Graph/U2 smoke、固定 CANN 9.0.0 的
> C32 on/off 三轮和 Attention/FFN DP0 双侧 profile；吞吐候选提升 `6.365%`，Attention
> non-overlap communication 中位数降低 `50.118%`。完整结果和结论边界见本报告第 9 节及
> [v0.23/plugin 开发串讲报告 2.7 节](DEEPSEEK_V4_AFD_V023_PLUGIN_DEVELOPMENT_REPORT_ZH.md#27-2026-08-31-混合-dag-与物理-stream-解耦候选)。

## 2. 背景和问题

P8D 已将 eager/U2 的两个 Python stage 线程收敛为单线程`layer -> stage 0 -> stage 1`，但 eager P1 仍只有 16.472 token/s。下一功能阶段要在不引入异步 HCCL 的前提下支持 Graph/U2，使两个 microbatch 的 decoder 执行、HCCL hidden/output传输和 stage 顺序进入同一个 FULL ACL Graph。

初始实现沿用 Graph/U1 的 stage-major 捕获，每个 stage 由一个 Python 线程执行；FFN 侧却按layer-major 顺序收发。两侧 HCCL communicator op 顺序不一致，导致 capture timeout，且通信stream 没有正确加入 capture stream。另一个独立问题是 DSV4 DSA compressed attention 被误判为普通 MLA full-graph，触发了不适用的 FIA workspace 路径。

因此本阶段不是简单删除 Graph/U2 fail-fast，而是统一 Graph warmup、capture 和 FFN 的通信顺序，并明确 DSV4 sparse/compressed attention 的 Graph 能力边界。

## 3. 固定环境

| 项目 | 固定值 |
|---|---|
| 日期 | 2026-08-20 |
| 多流增量验证日期 | 2026-08-29 |
| 基础提交 | `1d8db03f6c59c224b14204251bed5169bec31bc2` |
| 开发分支 | `feat/dsv4-afd-hccl-graph-u2` |
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| 多流增量 CANN | `/mnt/workspace/code/.ascend/cann-9.0.0/cann-9.0.0` |
| venv | `/mnt/workspace/code/.venvs/afd-v023-vllm-cann` |
| vLLM | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，`releases/v0.23.0` |
| vLLM-Ascend | `3da28f9414583d2d0b672a8f06d1fae142404bda`，`rfc/vllm_cann` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| golden | `/mnt/workspace/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json` |

拓扑为 Attention NPU 0-7、FFN NPU 8-15、DP8/TP1/EP8，seed 1024、temperature 0。性能 guard 关闭 vLLM async scheduling，通信仍使用同步 `send/recv`。

## 4. 关键实现和原因

### 4.1 Graph capture 改为单线程 layer-major

Graph/U2 的 warmup 和真实 capture 都调用 DSV4 的`forward_ubatches_layer_major()`，在一个 host 线程内按每层 stage 0、stage 1 的顺序推进。两个stage 被合并为一个 `AscendModelOutput` 并绑定到同一个 NPUGraph。

这样做的原因是 HCCL P2P op 顺序必须在 Attention、FFN、warmup 和 capture 之间完全一致。它也消除了 stage-major 两线程在 capture 期间的 GIL 交接和 communicator 排序不确定性。

### 4.2 capture 期间使用 graph-visible HCCL op

编译或真实 Graph capture 时，connector 使用 torch-npu 注册的 HCCL `_send/_recv` op；普通eager 和 graph 外路径继续调用 `torch.distributed.send/recv`。真实 capture 状态也被纳入判断，避免 Dynamo trace 结束后回退到不可捕获的 Python wrapper。

Attention 的 connector-owned stream/event 在 layer-major capture 中显式汇合到 Graph 主compute stream，保证 Graph 包含所需通信依赖。这里没有使用 `isend/irecv`、后台通信线程或`torch.ops.vllm.afd_camp2p_send_attn_output()`。

### 4.3 子 forward context 保留父 Graph 状态

U2 子 context 的 `cudagraph_runtime_mode` 会显示为 `NONE`。实现增加`afd_graph_ubatching` 标记保存父 FULL Graph 状态，避免 connector 把 Graph 子 forward 错当成普通 eager。编译期判断放在 stream/event 状态读取之前，避免 Dynamo 检查 Python dict 和 event 对象。

### 4.4 DSV4 compressed attention 不进入普通 MLA FIA workspace

runner 同时检查 `index_topk`、`use_sparse` 和 `use_compress`，识别 DSV4 DSA sparse/compressed attention。该模型仍执行正常 ACL Graph capture，但跳过只适用于普通 MLA 的full-graph FIA workspace 分配，避免错误的 workspace 需求阻塞启动。

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

覆盖范围包括 feature accept/reject、Graph transport 选择、编译期分支、子 context、Graph layer-major capture、DSV4 sparse MLA 判断、recipe 门禁以及 eager/U1/U2 既有路径回归。Ruff 和 `git diff --check` 通过。

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

两轮 batch 的诊断项均为 1/1、3/8、1/32 token exact。并发 DP 调度不保证逐样本 token 确定性，所以 batch 门禁检查返回数量、结构和生成有效性；严格 token 一致性由每轮 30 个串行请求负责。该诊断值在报告中保留，不能改写成 batch 全量 token exact。

Graph capture size 1/2/4/8 全部完成，Attention 各 rank 额外 Graph HBM 约 1.22-1.24 GiB。正式 F0 前的 smoke 也完成单请求精确匹配和 batch8 有效性：

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

相对 async-scheduling-off eager/U1 单点 30.615 token/s 为 +250.116%，相对 eager/U2 P8D 单点 16.472 token/s 为 +550.748%。这两个比较证明 P1 没有触发 20% 回退暂停条件，但Graph/U2 同时改变了执行模式，不能把差值全部归因于 microbatch，也不能替代 Graph/U1 和同预算native Graph 的公平对照。

P1 仅一轮，未采 profiler。`P8D-PERF-001` 继续 Open；只有完成 Graph/U1、Graph/U2 和同预算native Graph 的三轮 P2、波动门禁及必要 profile 后，才能宣称 AFD + microbatch 有性能收益。

## 8. 2026-08-29 Graph/U2 显式多流增量

### 8.1 优化前后

优化前的 Graph/U2 已经做到 layer-major capture/replay，也把 HCCL `_send/_recv` 放进NPUGraph，但模型计算和这些通信都在原始 capture stream 上按序执行。它解决了 Graph 正确性和 host 下发问题，没有显式建立 U0/U1 的通信计算重叠。

本次只优化以下边界：

- vLLM `releases/v0.23.0` + vLLM-Ascend `rfc/vllm_cann`；
- `P2pHcclAFDConnector`、A8F8、TP1、`FULL_DECODE_ONLY`、U2；
- MTP、PD 和非等量拓扑关闭；
- 只修改 afd-plugin，不修改 vLLM 或 vLLM-Ascend。

目标 DAG 不是把 eager 的三个 stream 原样搬进 Graph，而是保留 HCCL 所在的 parent capture stream，只把模型计算 fork 到一个 compute stream：

```text
Attention，每层 L：
compute stream:  C(L,U0) ----------------> C(L,U1)
parent stream:      wait C0 -> send U0 ------ wait C1 -> send U1 -> recv U0 -> recv U1
                                  |<--- send U0 与 C(L,U1) 重叠 --->|

FFN，每层 L：
parent stream:   recv U0 -------- recv U1 ---------------- wait C0 -> send U0 -> wait C1 -> send U1
compute stream:          wait R0 -> C(L,U0) -> wait R1 -> C(L,U1)
                         |<-- recv U1/C0 -->|       |<-- send U0/C1 -->|
```

这仍是同步 HCCL API，重叠来自 NPU stream 和 event DAG，不代表改成了异步 connector。

### 8.2 为什么 Graph 不直接复用 eager 三 stream

eager 路径的普通 `dist.send/recv` 是 Python/c10d 调用：调用发生时立即向当前 stream 提交HCCL，并在 host 侧等待 `Work`。Graph 路径的 `_send/_recv` 则是 torch-npu 注册的PrivateUse1 op，它在 capture 时成为图节点，replay 时由图恢复 HCCL 顺序。两者协议相同，但执行时机和 capture 可见性不同：

| 维度 | 普通 `dist.send/recv` | Graph-visible `_send/_recv` |
|---|---|---|
| 调用位置 | eager Python 路径 | NPUGraph capture 路径 |
| 图可见性 | 不是稳定的 Graph 节点 | 作为 HCCL 节点 capture/replay |
| shape 处理 | Python wrapper 接收运行时 tensor | plugin 直接调用 op，并把可选 shape 设为 `None` |
| 同步语义 | `Work.wait()` 后返回 | 当前 torch-npu lowering 内部同样等待已提交 work |
| 本次 stream | eager A2F/F2A 通信 stream | 必须保留 parent capture stream |

关键分派仍由一个 connector 完成：

```python
if self._graph_transport_active():
    _graph_hccl_send(send_tensor, dst=dst, group=group)
    return
dist.send(send_tensor, dst=dst, group=group)
```

第一版实验曾把 graph-visible HCCL 也移到 side stream。结果 A1F1 在 `_send` lowering 的
`c10d.send(...).wait()` 处互等，随后 `capture_end` 报告 side stream 未 join 原始 stream：

```text
capture model contains a stream that was not joined to the original stream
```

失败产物保留在：

```text
/mnt/workspace/results/afd_graph_multistream_component_a1f1_20260829/summary.json
```

根因是 side HCCL 先等待一个尚未结束的 parent Graph event，而 parent capture 又在同步等待HCCL work，形成 capture 期环形依赖。最终方案让 `_send/_recv` 留在 parent，只把不会在 host同步等待的模型计算移到 side stream，从依赖结构上消除该环。

### 8.3 修改点与代码对应

| 修改点 | 当前源码与符号 | 关键变化 | 目的 |
|---|---|---|---|
| Attention Graph stream/event | `afd_plugin/connectors/npu/p2p_hccl.py::attention_graph_compute`、`wait_for_attention_graph_compute` | parent 记录 ready，compute stream 等待并执行，parent 再等待 compute_done | 建立可 capture 的 fork/join |
| eager/Graph stream 分界 | `P2pHcclAFDConnector._attention_stream_pipeline_active` | `afd_graph_ubatching` 时禁用 eager A2F/F2A side streams | 防止 graph-visible HCCL 离开 parent stream |
| Attention 调度拆分 | `afd_plugin/model_executor/models/deepseek_v4.py::_forward_ubatches_graph_compute_pipeline` | 先排 U0/U1 compute，再按 U0/U1 dispatch，最后 receive | 让 `send(U0)` 覆盖 `compute(U1)` |
| 远端 MoE 两阶段接口 | `deepseek_v2.py::RemoteFFNProxy.dispatch_remote_ffn`、`receive_remote_ffn` | 把原来的 send+recv 原子调用拆成 transfer handle | 调度器可在 receive 前插入另一 stage |
| FFN Graph 调度 | `afd_plugin/v1/worker/npu/ffn_model_runner.py::_ffn_forward` | Graph recv 留 parent；FFN compute 放 side stream；两 stage compute 排队后 parent join/send | 形成 `recv(U1)/compute(U0)` 与 `send(U0)/compute(U1)` |
| 同源码公平对照 | `p2p_hccl.py::_graph_u2_compute_overlap_enabled`、`run_performance.py::--graph-u2-compute-overlap` | 严格 `0/1` 环境开关，默认开启；runtime manifest 记录 on/off | 不回退源码即可复现实验前后 DAG |
| 组件验证 | `tools/dsv4/validate_hccl_p2p_roundtrip.py::_validate_graph_transport` | 新增 `--graph-multistream`，覆盖 capture、replay、输入更新和双 stage | 在实模前验证 DAG 与 HCCL 生命周期 |

Attention connector 的 fork/join 是本次多流的底层原语：

```python
events.ready.record(parent_stream)
with torch.npu.stream(compute_stream):
    events.ready.wait(compute_stream)
    tensor.record_stream(compute_stream)
    yield
    events.compute_done.record(compute_stream)

events.compute_done.wait(parent_stream)
tensor.record_stream(parent_stream)
```

模型调度不再把远端 MoE 看作必须立即收回的原子调用，而是显式分成三段：

```python
for stage_idx in (0, 1):
    with compute_scope(layer_idx=layer.layer_idx, stage_idx=stage_idx, ...):
        hidden[stage_idx], continuation[stage_idx] = (
            layer.forward_attention_to_remote_ffn_input(...)
        )

for stage_idx in (0, 1):
    wait_for_compute(layer_idx=layer.layer_idx, stage_idx=stage_idx, ...)
    transfers[stage_idx] = layer.dispatch_remote_ffn(hidden[stage_idx])

for stage_idx in (0, 1):
    hidden[stage_idx] = layer.receive_remote_ffn(transfers[stage_idx])
```

FFN 侧同样保留 parent HCCL，只在收到一个 stage 后把 MoE 排到 compute stream：

```python
payload = self.connector.recv_attn_output(...)
recv_event.record(torch.npu.current_stream())
with torch.npu.stream(self.ffn_compute_stream):
    recv_event.wait(self.ffn_compute_stream)
    output = self.model.compute_ffn_output(...)
    compute_event.record(self.ffn_compute_stream)
graph_sends.append((output, context, stage_idx, compute_event))

for output, context, stage_idx, compute_event in graph_sends:
    compute_event.wait(torch.npu.current_stream())
    _send_ffn_output(self.connector, output, context, stage_idx=stage_idx)
```

### 8.4 外部实现和原 CAMP2P U2 的参考边界

外部 `cann-recipes-infer` 提交
[`c6c7315f`](https://gitcode.com/yijie19/cann-recipes-infer/commit/c6c7315f4bc0cd2dd1646540bdd1a4799e36a561?ref=dsv4-asyn)
值得参考的是 DAG 组织：按 microbatch 分 event、tensor `record_stream`、Attention 的 send/recv stream 和 FFN 的 recv/compute/send 分段。它直接在自己的推理运行时中调用 `dist.send/recv`，没有 afd-plugin 当前的 graph-visible HCCL、Graph key、双侧 fallback 和 forward-context 契约，因此不能直接复制其普通通信调用。

原 [CAMP2P 使用指南](CAM_P2P_CONNECTOR_USER_GUIDE.md) 中的 U2 主要做两件事：vLLM DBO 切出两个 ubatch，并为每个 ubatch 创建独立 HCCL AFD group；数据面由 CAMP2P 自定义 op执行。它证明了“stage 独立 communicator + U2 消息身份”的设计，但当前 feature validation明确拒绝 CAMP2P Graph/U2，所以不能作为本次标准 HCCL Graph 多流实现。此次优化不需要修改vLLM-Ascend，依赖的是当前 torch-npu 已有的 NPUGraph 多 stream/event 和 `_send/_recv` 能力，调度、生命周期及门禁全部留在 afd-plugin。

### 8.5 验证结果

#### 8.5.1 静态与单元验证

Ruff、`py_compile`、`git diff --check` 通过。与本次修改直接相关的定向测试共 98 项通过：

```text
tests/unit/connectors/test_p2p_hccl_connector.py                    79 passed
runner/manifest 定向子集                                             16 passed
tests/unit/v1/worker/test_npu_runtime.py（真实 NPU 多流定向）          3 passed
```

测试检查的不是只有分支命中，还包括 parent/compute event 的 fork/join 顺序、Attention 每层`compute0, compute1, send0, send1, recv0, recv1` 顺序，以及 FFN Graph HCCL 必须在 parent stream、计算必须在 side stream。

#### 8.5.2 真实 NPU 组件与实模 F0

| 门禁 | 结果 |
|---|---|
| A1F1 Graph 多流组件 | 通过，双方 rc=0 |
| A8F8 Graph 多流组件 | 16 个进程全部 rc=0；capture/replay、updated input、IDs、fan-in/out、close 通过 |
| A8F8 实模 F0 | 30/30 golden token exact；batch 1/8/32 合法 |
| Graph/U2 | capture、replay和在线请求均观察到 stage 0/1 |
| 生命周期 | Attention/FFN rc=0，fatal marker 为空，NPU 第一次清理通过 |

证据目录：

```text
/mnt/workspace/results/afd_graph_multistream_component_a1f1_root_hccl_20260829/summary.json
/mnt/workspace/results/afd_graph_multistream_component_a8f8_20260829/summary.json
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_multistream_f0_20260829/validation_summary.json
```

F0 的 batch token-exact 诊断为 1/1、3/8、7/32；与既有验证口径一致，batch 的硬门禁是choice 数、结构和有限输出，30 条串行 golden 才承担逐 token 精确门禁。

#### 8.5.3 CANN 时间线重叠

单轮 profile 固定 C32、input 1024、output 128、128 请求、active 20 step。服务与 raw profile 门禁、fatal log、双侧 rc=0 和 NPU cleanup 全部通过。离线解析使用采集时同一 CANN 9.0.0，不能与历史 9.0.1 raw 数据混用。

重叠统计只计算 `hcom_send_`/`hcom_receive_` 与本次 compute stream 上 AI Core task 的时间交集，排除 FFN 内部 `hcom_allGather_`、capture/warmup stream 和 HCCL AICPU task：

| Role/窗口 | P2P 类型 | op 数 | 有计算交集 op | 交集时长 / P2P 时长 | 结构证据 |
|---|---|---:|---:|---:|---|
| Attention step 67-86，stream 204 | send | 1760 | 860 | 5.010 / 28.342 ms，17.678% | 20 step x 43 send |
| Attention step 67-86，stream 204 | receive | 1720 | 0 | 0 / 316.332 ms | 当前 DAG 预期不重叠 |
| FFN 稳态 step 74-86，stream 149 | send | 1118 | 559 | 4.230 / 8.308 ms，50.922% | 13 step x 43 send |
| FFN 稳态 step 74-86，stream 149 | receive | 1146 | 559 | 180.147 / 532.122 ms，33.854% | 13 step x 43 receive |

每个稳定 step 精确出现 43 个重叠 op，与 DeepSeek-V4 的 43 层一一对应，证明预期 DAG已经落到设备时间线。Attention receive 为 0 也符合当前顺序：两 stage send 完成后才按 U0/U1 receive；本次没有声称覆盖该段等待。

本轮带 profiler 的服务 guard 为 128/128、151.655 token/s、p50 TPOT 157.069 ms、p99 TPOT 210.364 ms。该数字包含 profiler 开销且只有一轮，只证明功能和重叠存在，不能与此前107.189 token/s 直接比较，也不能表述为正式收益。

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_multistream_profile_c32_20260829/performance_summary.json
.../profiles/attention/*_ascend_pt/ASCEND_PROFILER_OUTPUT/{step_trace_time.csv,kernel_details.csv,trace_view.json}
.../profiles/ffn/*_ascend_pt/ASCEND_PROFILER_OUTPUT/{step_trace_time.csv,kernel_details.csv,trace_view.json}
```

#### 8.5.4 FFN 全 rank 到达偏斜诊断

为判断 FFN 时间线中较长的 `MoeDistributeDispatchV2` 是 Graph 未启用、内核计算慢，还是
MC2 集合通信等待，本轮新增 role-local rank 可选 profiler，并对 FFN rank 0-7 同窗口采集。
8 个 rank 均记录到 `Model ID=46`、`OP State=static` 的 Graph kernel，以及 8 次
`AscendCL@aclmdlRIExecuteAsync`；因此 FFN Graph 确实已启用。

首层 stage0 对应 Graph task 12。按设备全局时间戳对齐 8 个 replay 后得到：

| replay | 有效 rank | task 12 开始偏斜 | duration 最小/最大 | 结束时间偏差 |
|---:|---:|---:|---:|---:|
| 0 | 7 | 17,312.000 us | 37.021 / 17,340.520 us | 19.000 us |
| 1 | 8 | 6,377.500 us | 34.760 / 6,415.908 us | 19.750 us |
| 2 | 8 | 2,660.000 us | 38.140 / 2,701.034 us | 21.500 us |
| 3 | 8 | 3,515.000 us | 33.780 / 3,547.050 us | 23.000 us |
| 4 | 8 | 3,127.250 us | 36.241 / 3,147.903 us | 25.500 us |
| 5 | 8 | 3,474.500 us | 35.920 / 3,495.930 us | 26.500 us |
| 6 | 8 | 4,018.000 us | 35.861 / 4,034.980 us | 27.250 us |
| 7 | 8 | 4,698.250 us | 35.160 / 4,739.100 us | 27.000 us |

首轮 rank 2 的 profiler 在 Graph 中段进入 active 窗口，缺少 task 12，所以该轮只使用 7 个
rank；后 7 轮均为 8/8。63 个有效观测中，`median(dispatch_end) - dispatch_start` 与
实际 duration 的相关系数为 `0.999997`，duration 与预计等待的中位差仅 `0.021 us`；各
rank 的 dispatch 结束时间绝对偏差中位数为 `3.000 us`、最大 `22.000 us`。这组证据说明
长 duration 几乎全部是在 MC2 dispatch 中等待其他 EP rank，不是 3-17 ms 的持续计算。

向前追踪 Graph replay API 后，稳态 7 轮的 `aclmdlRIExecuteAsync` 发出时间已经存在
`3.947-6.026 ms` 的跨 rank 偏斜，其与 task 12 开始时间的逐轮相关系数为
`0.689-0.987`。Graph 内首个 dispatch 前只有约 34 us 的 cast、matmul、gating 等固定任务；
stage0 AFD `hcom_receive_` 已与该段并行，不能解释毫秒级偏斜。因此当前主因在 Graph replay
进入前的 host/rank 调度到达差，而不是 FFN Graph 开关或 `MoeDispatch` 本体。

FFN host 路径可以继续拆成
`recv_control_payload(Gloo) -> decode/update state -> recv input IDs(U0/U1) -> graph.replay()`。
稳态各 rank 的 control-body `gloo:recv` 开始时间只相差 `0.068-0.141 ms`，但该阻塞接收
包含等待对应 Attention peer 发送 metadata 的时间，单 rank duration 为 `17.312-27.034 ms`，
结束时间相差 `1.494-5.647 ms`；这不能直接解释成 Gloo wire latency。Graph 前最后一次
input-ID `c10d::recv_` 完成时间相差 `3.329-6.009 ms`，与
`aclmdlRIExecuteAsync` 发出时间的逐轮相关系数为 `0.974-0.998`，Graph API 只在其后
`0.265-1.372 ms` 发出。由此可把下一轮定位收敛到 Attention metadata/input-ID 发送准备和
FFN Graph 前 host 控制路径，而不是 Graph 内 MoE。

不能据此在 Graph 前直接加 barrier：barrier 只会把相同等待从 `MoeDispatch` 移到同步点，
不会缩短最晚 rank 的关键路径。后续优化应先对齐 host 控制路径和 replay enqueue，定位最晚
rank 在请求接收、metadata/Gloo 控制、input-ID 入队与 Graph 提交之间的 CPU 调度差。下一份
profile 应设置 Attention 和 FFN 的 profiler rank 均为 `all`，同窗口对齐每个 Attention peer
的 control-body send、U0/U1 input-ID send 与对应 FFN receive；只有能缩短最晚 rank 到达时间
的改动才进入正式性能对照。

本轮 64/64 请求成功，Graph/U2 双 stage 和 NPU cleanup 通过。服务 summary 的整体状态为
failed，原因一是原 profile gate 只接受单 trace，原因二是 Attention 强制退出后的 TBE
manager `EOFError` 被 fatal marker 捕获；二者都发生在成功 workload 之后。本次同步修复了
多 trace profile gate，shutdown marker 仍保留严格判定，不把本轮误写为完整功能门禁。

```text
/mnt/workspace/validation/dsv4_afd_v023_graph_u2_ffn_allranks_profile_c32_20260829/performance_summary.json
.../profiles/ffn/*_ascend_pt/ASCEND_PROFILER_OUTPUT/{kernel_details.csv,trace_view.json}
```

#### 8.5.5 同提交三轮公平对照

本轮固定 afd-plugin `6c636961`、vLLM `0fc695fc`、vLLM-Ascend `3da28f94`、CANN
9.0.0、A8F8/TP1、MTP off、PD off 和 `FULL_DECODE_ONLY`。每组均为 C32、128 请求、
input 1024、output 128、3 轮、request rate `inf`、seed 1024、temperature 0、ignore EOS；
正式吞吐轮关闭 profiler。Graph/U1 和优化前 U2 设置 overlap off，优化后 U2 设置 overlap on。

| 模式 | 三轮 output throughput，token/s | 均值，token/s | CV | token/s/NPU | 稳定性门禁 |
|---|---|---:|---:|---:|---|
| Graph/U1 | 133.716 / 126.078 / 144.819 | 134.871 | 5.705% | 8.429 | 通过 |
| 优化前 Graph/U2 | 141.451 / 108.845 / 125.457 | 125.251 | 10.628% | 7.828 | 未通过，仅 CV 超限 |
| 优化后 Graph/U2 | 145.990 / 128.283 / 143.627 | 139.300 | 5.635% | 8.706 | 通过 |

| 模式 | p50 TTFT 均值 | p50 TPOT 均值 | p99 TPOT 均值 | Attention/FFN 最大 HBM |
|---|---:|---:|---:|---:|
| Graph/U1 | 8499.655 ms | 171.104 ms | 235.675 ms | 60332 / 43952 MiB |
| 优化前 Graph/U2 | 8884.964 ms | 176.329 ms | 269.954 ms | 61950 / 44438 MiB |
| 优化后 Graph/U2 | 6560.809 ms | 159.324 ms | 227.682 ms | 61582 / 44528 MiB |

优化后相对优化前，吞吐均值 `+11.217%`，p50 TTFT `-26.158%`、p50 TPOT
`-9.644%`、p99 TPOT `-15.659%`，吞吐 CV 从 `10.628%` 降到 `5.635%`。优化后相对
Graph/U1，吞吐 `+3.284%`，p50 TTFT `-22.811%`、p50 TPOT `-6.884%`、p99 TPOT
`-3.391%`。这里的负延迟百分比表示延迟下降。

三组共 1152/1152 正式请求成功；所有 warmup、双 stage、fatal log、双侧退出、NPU monitor
和 cleanup 门禁均通过。优化前 U2 的 summary 整体为 failed，只因为 output throughput CV
超过 10%，不是功能或生命周期失败。因此 `+11.217%` 是有方向性的候选证据，不能当成已经
冻结的精确收益；优化后 U2 自身稳定，并高于稳定 U1，但 `+3.284%` 还不足以关闭
`P8D-PERF-001`。

三份 runtime manifest 的提交、CANN、拓扑、模型、负载和五个启动/验证文件哈希一致。测试期间
tracked worktree diff SHA 从 `3acc7a8b...` 变为 `ef900a2b...`，核对结果是 Markdown 报告被
保存；实际运行源码前后聚合指纹均为 `7a365b38...`。该差异已披露，未把文档改动误当作运行
代码变化。

```text
/mnt/workspace/validation/dsv4_afd_v023_graph_u2_overlap_p2_20260829/comparison_summary.json
.../graph_u1/{runtime.json,performance_summary.json}
.../graph_u2_pre_overlap/{runtime.json,performance_summary.json}
.../graph_u2_post_overlap/{runtime.json,performance_summary.json}
```

### 8.6 当前状态和剩余工作

当前可以下结论：预期的 Graph/U2 多流重叠已经实现，A8F8 正确性和生命周期没有回退；
优化后的三轮点通过稳定性门禁，吞吐高于本轮 Graph/U1 和优化前 U2 均值。
当前不能下结论：`+11.217%` 是可冻结的精确收益，或已经关闭 `P8D-PERF-001`。

三轮对照已完成，但优化前 U2 的单组波动超限，且优化后相对稳定 Graph/U1 只提升
`3.284%`。正式闭环需要交错执行或增加轮数以稳定优化前对照，并补齐同预算 native Graph、
必要的双侧同窗口 profile 及 MTP on/off 矩阵，再决定是否创建性能 tag。MTP、PD、TP2 和
非等量拓扑与本次增量没有组合验证，继续沿用原有门禁。

## 9. 2026-08-31 混合 DAG CANN 9.0.0 性能验证

### 9.1 对照口径

固定 afd-plugin `6c636961`、vLLM `0fc695fc`、vLLM-Ascend `3da28f94`，使用
`/mnt/workspace/code/.ascend/cann-9.0.0/cann-9.0.0`、A8F8/TP1、
`FULL_DECODE_ONLY`、Graph/U2、MTP off、async scheduling off。两组均为 C32、input
1024、output 128、128 请求/轮、3 轮，且保持 `graph-u2-compute-overlap=on`；唯一变量是
`graph-u2-hybrid-dag=on/off`。

### 9.2 三轮无 profiler 结果

| 模式 | 三轮 output throughput，token/s | 均值 | CV | token/s/NPU | p50 TPOT | p90 TPOT | p99 TPOT |
|---|---|---:|---:|---:|---:|---:|---:|
| hybrid off | 121.695 / 142.886 / 114.236 | 126.272 | 9.611% | 7.892 | 186.525 ms | 220.264 ms | 239.696 ms |
| hybrid on | 135.374 / 141.692 / 125.861 | 134.309 | 4.845% | 8.394 | 172.687 ms | 211.340 ms | 239.833 ms |

hybrid on 相对 off 的吞吐均值提升 `6.365%`；p50/p90 TPOT 下降
`7.419%/4.052%`，p99 TPOT 变化 `+0.057%`。p50/p90/p99 TTFT 分别变化
`-1.415%/-17.157%/-12.170%`。两组共 768/768 正式请求成功，均观察到真实 U2 双
stage；warmup、fatal log、双侧退出、NPU monitor 和 cleanup gate 全部通过。

三组同序号吞吐差值为 `+11.240%/-0.836%/+10.177%`。两组 CV 都通过预设 10% 门槛，
因此可以把 `+6.365%` 记为当前 C32 候选收益；样本仍只有 3 轮，不能外推为跨并发、MTP
或 native 同预算的精确收益。

### 9.3 双侧 profile 结果

profile 固定 Attention DP0、FFN DP0、`skip_first=64, wait=2, warmup=1, active=20,
repeat=1` 和 `with_stack=false`。四份 raw `profiler_info_0.json` 都记录
`cann_version=9.0.0`，并由同一路径 CANN 9.0.0 串行离线解析。每份均生成非空的
`step_trace_time.csv`、`kernel_details.csv`、`communication.json`、`trace_view.json` 和
数据库。

Attention DP0 取 on/off 都完整稳定的 matched steps 67-80，中位数如下：

| 指标 | hybrid off | hybrid on | 变化 |
|---|---:|---:|---:|
| Computing | 38.855 ms | 38.940 ms | +0.218% |
| Communication(Not Overlapped) | 18.855 ms | 9.405 ms | -50.118% |
| Overlapped | 14.795 ms | 29.277 ms | +97.882% |
| Overlapped / Communication | 43.939% | 75.646% | +31.707 pp |
| Stage | 53.216 ms | 43.128 ms | -18.958% |
| Bubble | 31.948 ms | 32.048 ms | +0.313% |

计算时间和 Bubble 基本不变，而未重叠通信减半、overlap ratio 提升 31.707 个百分点，
说明混合 DAG 的收益来自 `recv_done(L,S0) -> compute(L+1,S0)` 与 S1 exchange 的真实设备
重叠，不是少算或 U2 退化。Attention Graph capture 和正式 benchmark 均记录
`stage_count=2`。

FFN active step 与角色启动/跨请求 receive 等待没有完全对齐：hybrid on/off 分别只有 7/6
个非零计算 step，后续 step 主要是通信和 preparing，因此不能直接比较 FFN 20-step 总和。
FFN trace 只用于确认 Graph 执行和通信因果，不用于计算收益百分比。

profile-on 的 128/128 workload、U2 和 raw profile gate 通过，但 Attention 正常 shutdown
完成后，CANN 9.0.0 TBE manager 队列线程收到 `EOFError`，触发通用 fatal marker，故该次
`performance_summary.json` 保持 `passed=false`。异常发生在 workload 和 raw trace 落盘
之后；离线产物完整，但该单轮 profile 吞吐不纳入正式性能均值。profile-off 的全部门禁通过。

### 9.4 证据与状态

```text
/mnt/workspace/validation/dsv4_afd_v023_cann900_hccl_graph_u2_hybrid_off_c32_r3_20260831_codex1
/mnt/workspace/validation/dsv4_afd_v023_cann900_hccl_graph_u2_hybrid_on_c32_r3_20260831_codex1
/mnt/workspace/validation/dsv4_afd_v023_cann900_hccl_graph_u2_hybrid_off_c32_profile_20260831_codex1
/mnt/workspace/validation/dsv4_afd_v023_cann900_hccl_graph_u2_hybrid_on_c32_profile_20260831_codex1
```

当前可以下结论：逻辑 DAG 与物理 stream 解耦后的混合 DAG 已真实执行，C32 三轮稳定，
Attention 时间线和端到端吞吐方向一致。当前不能下结论：该结果已关闭
`P8D-PERF-001`，或能代表其他并发、MTP、native Graph、PD、TP2 和非等量拓扑。

## 10. 下一步

混合 DAG 的 C32 候选收益和设备机制已经闭环，但仍不能命名为性能 baseline，也暂不创建
性能 tag。当前优先级是：

1. 补齐混合 DAG 的最小 Graph component、A1F1 连续 capture/replay 和 A8F8 完整 F0；
2. 交错执行或增加 hybrid on/off 轮数，并补 C16/C64，确认收益不依赖单一 C32 工作点；
3. 补齐同预算 native Graph、MTP on/off 和对齐 active window 的 FFN profile，判断是否关闭
   `P8D-PERF-001`；
4. 独立 send/recv stream 先在最小 HCCL Graph component 连续 capture/replay 100 次通过，
   再修改 V1 物理映射；
5. TP2、非等量拓扑、Mooncake PD、Graph/U3 和 A5 继续保持独立里程碑，未验证前维持门禁。
