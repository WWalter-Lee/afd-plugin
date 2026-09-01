# DeepSeek-V4 AFD v0.23 / afd-plugin 开发串讲报告

## 1. 报告定位

本文面向组内技术串讲，按特性而不是按提交或文件组织 DeepSeek-V4 AFD 的开发内容，回答四个问题：改了什么、为什么要改、修改的工程意义是什么、验证到了什么程度。

| 项目 | 固定口径 |
|---|---|
| 报告截止日期 | 主功能基线 2026-08-25；Graph/U2 混合 DAG 实现更新 2026-08-31 |
| vLLM | `releases/v0.23.0`，`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann`，`3da28f9414583d2d0b672a8f06d1fae142404bda` |
| afd-plugin | `feat/dsv4-afd-mooncake-pd`，当前已提交 HEAD `6c636961d3791df545ae065811368dc8beb1e4ab` |
| 开发范围 | 从 `99ee0ef6` 的 v0.23 兼容迁移到当前 Mooncake PD M9 |
| 主要模型 | `DeepSeek-V4-Flash-w8a8-mtp` |
| 功能验证工具链 | 历史功能基线使用 CANN 9.0.1；2026-08-29 Graph/U2 多流增量固定 CANN 9.0.0；Python 3.12、`torch_npu 2.10.0.post2`、`afd-v023-vllm-cann` venv |
| 当前工作树 | M9 手工部署与 Graph/U2 多流增量仍有未提交改动；只能按各自证据标记状态，不能整体视为冻结基线 |

早期在 `zingercode_vllm-ascend` 中直接修改上游源码的探索不属于本文范围；本文只讨论迁移到 v0.23 后以 `afd-plugin` 为唯一 AFD 扩展边界的实现。

### 1.1 状态定义

| 状态 | 含义 |
|---|---|
| 已冻结功能基线 | 实模 golden、目标组合、生命周期和清理门禁通过，并已创建功能 tag |
| 组件闭环 | CPU/Mock 和真实 NPU 组件通过，但受硬件或资源限制，未完成目标实模 E2E |
| 功能通过、性能未闭环 | 正确性可用，但单轮或 profile 显示性能缺口，不能创建性能 tag |
| 进行中 | 代码或工具已有阶段性证据，但完整 F0 尚未通过 |
| fail-fast | 未验证或已知有问题的组合在启动前显式拒绝，不能视为支持能力 |

### 1.2 建议串讲顺序

1. 先用 AF 分离解释 AFD 的核心：为什么拆、拆在哪里、A/F 两侧分别执行什么。
2. 再讲 Microbatch 和 NPU Graph：前者改变执行流水，后者改变算子下发方式，两者都要求 A/F 消息顺序严格一致。
3. 然后讲 MTP：它在 AF 基础协议上增加一轮 draft MoE 远程调用，并与 U2、Graph 组合。
4. 再讲 AF 非等量拓扑和 TP2：分别扩展 A/F 数量关系与单角色内部并行维度。
5. 最后讲 PD 分离：在 Decode AF 之外增加 Prefill 到 Decode Attention 的 KV 通道。
6. v0.23/plugin 兼容、验证工具和性能分析作为横向工程底座穿插说明，不作为独立产品特性。

### 1.3 代码与证据阅读约定

- “修改点与代码对应”表中的路径均相对 afd-plugin 仓库根目录；同一表内重复出现的短文件名沿用该表首次给出的完整目录。
- 表中符号以当前 `feat/dsv4-afd-mooncake-pd` 工作树为准；第 13.1 节记录引入或冻结该能力的提交，便于查看历史 diff。
- 代码块是从对应符号中保留关键分支后的短摘录；`...`、注释或缩短的错误文本表示省略构造参数、日志和非关键校验，完整实现以表中的源码位置为准。
- 每个特性末尾给出专项报告；状态和数字以专项报告及其验证 JSON 为证据，不根据代码存在与否推断功能已经完成。

## 2. 整体架构和当前结论

### 2.1 数据流

```text
                         Mooncake KV transfer（M9）
Prefill ----------------------------------------------> Decode Attention
                                                            |
                                                            | input IDs（step 级）
                                                            | hidden [T, H]（layer 级）
                                                            v
                                                     Decode FFN / MoE
                                                            |
                                                            | FFN output [T, H]
                                                            v
                                                     Decode Attention

控制面：DP metadata、Graph/eager 决策、MTP phase、TP/DP 拓扑
数据面：标准 HCCL P2P send/recv；PD 的 KV 数据由 Mooncake 独立承载
```

DeepSeek-V4 的拆分边界放在远端 MoE，而不是把整个 FFN 子层搬走。Attention 节点保留 HC-Attention、Attention、HC-FFN-pre、Norm 和 HC-FFN-post；FFN 节点只持有并执行 MoE。这样跨 AF 只传二维 `[tokens, hidden_size]`，同时复用 afd-plugin 已有的远端 experts 抽象。

### 2.2 支持矩阵

| 特性 | 当前状态 | 已验证范围 | 不能外推的范围 |
|---|---|---|---|
| AF 分离 | 已冻结功能基线 | DeepSeek-V4 角色化加载、A8F8、HCCL P2P eager U1/U2 | 不代表异步 HCCL |
| Microbatch | 功能通过、性能未闭环 | U1/U2、两个 microbatch、真实双 stage | U3；正式性能收益 |
| NPU Graph（ACL Graph） | P8F 功能增量通过；混合 DAG 单轮实机 smoke 通过、性能待闭环 | `FULL_DECODE_ONLY`、U1/U2、target/full-draft；P8F 的 parent-HCCL + side compute/send 已有证据；混合 DAG A8F8 Graph capture/replay、10/10 serial 和 batch32 通过 | 混合 DAG 完整 F0/on-off/profile；Graph U3；TP2 最大组合；C64 补位退化与正式净收益 |
| MTP | 已冻结功能基线 | M0-M7、eager/Graph、U1/U2、1 个 MTP layer、1 speculative token | 更多 speculative token |
| AF 非等量拓扑 | 组件闭环 | A1F1、A2F1、A4F2，eager/Graph/MTP 组件 | A8F4 实模 E2E 与性能 |
| TP2 | 已冻结功能基线 | 等量 A8F8、DP4/TP2、eager/U1 | CAMP2P TP2、非等量 TP2、TP3、最大 Graph+MTP 组合 |
| PD 分离 | 进行中 | Mooncake contract、runtime、两进程 NPU round-trip | 双机实模 F0、TP2、Graph、U2、MTP |
| v0.23/plugin 工程底座 | 已冻结功能基线 | 同栈 golden、兼容层、部署和验证工具 | 旧栈性能数字不能作为 v0.23 基线 |
| 正式性能验收 | 未完成 | 已有 P1、Graph U1/优化前 U2/优化后 U2 三轮对照和双侧 profile | pre-U2 稳定性超限，尚无可发布的目标栈性能 tag |

### 2.3 特性关系

这些特性不是按 M0-M9 串行替换，而是在 AF 分离底座上逐层组合：

```text
v0.23 / afd-plugin 工程底座
            |
            v
        AF 分离（核心）
            |
            +-- Microbatch：U1/U2 stage 调度
            +-- NPU Graph：capture/replay、Graph 内 HCCL
            +-- MTP：target 后增加 draft AF phase
            +-- AF 非等量拓扑：一个 FFN 聚合多个 Attention peer
            +-- TP2：单角色内部增加 TP rank 维度

组合能力：Microbatch + Graph、MTP + Microbatch/Graph、非等量 + MTP/Graph

PD 分离（外层组合）：
Prefill --Mooncake KV--> Decode Attention --HCCL--> Decode FFN
```

- AF 分离定义模型、worker 和 A2F/F2A 基础协议。
- Microbatch 改变一个 step 的 stage 调度；NPU Graph 改变执行与下发方式。
- MTP 在 target decoder 之后新增 draft AF phase，并分别与 Microbatch、Graph 组合。
- AF 非等量和 TP2 改变 rank/peer/shape 解释，不改变远端 MoE 的模型边界。
- PD 分离在 AF 外层增加 Prefill/Decode KV 数据面，复用已有 Decode AF 能力。

### 2.4 2026-08-29 第一版多流优化（历史）

本次不是新增一种 connector，也不是把同步 HCCL 改成异步 HCCL。优化对象严格限定为
`P2pHcclAFDConnector` 的 A8F8、TP1、MTP off、PD off、`FULL_DECODE_ONLY`、Graph/U2：

1. Graph-visible `_send/_recv` 保留在原始 parent capture stream，确保 HCCL op 顺序和 replay 可见性不变。
2. Attention 每层先把 U0/U1 计算排到 side compute stream，再由 parent 依次 join/send，目标是 `send(U0)` 覆盖 `compute(U1)`。
3. FFN parent 依次 recv U0/U1，收到一个 stage 就把 MoE 排到 side compute stream，目标是 `recv(U1)` 覆盖 `compute(U0)`，`send(U0)` 覆盖 `compute(U1)`。
4. 把远端 MoE 从“send 后立即 recv”的原子接口拆成 dispatch/receive 两阶段，给 layer-major调度器留下跨 stage 排序空间。
5. 新增真实 NPU Graph 多流组件门禁，并用 A8F8 F0 和 CANN 时间线验证设备端重叠。

当前状态：预期重叠已在每个稳定 step 按 43 层精确出现，功能和生命周期通过。同提交 C32 三轮对照已完成：优化后 U2 为 `139.300 token/s`、CV `5.635%`，相对优化前 U2 均值 `+11.217%`，相对稳定 Graph/U1 `+3.284%`。优化前 U2 的 CV 为 `10.628%`、未通过 10% 门槛，因此 `P8D-PERF-001` 和性能 tag 仍未闭环。

### 2.5 2026-08-29 ubatch 严格闭环增量

2.4 节记录的是第一版 Graph/U2 多流 DAG 及其三轮性能结果。本节是在该版本之后的依赖正确性增量：原 DAG 每层为 `compute0, compute1, send0, send1, recv0, recv1`，导致 U0 已发送后仍要等待 U1 发送才能接收自己的 FFN output；下一层 U0 Attention 计算也被 U1 receive阻塞。最终实现改为：

```text
Attention:
compute(L,0), compute(L,1), send(L,0), recv(L,0),
compute(L+1,0), send(L,1), recv(L,1), compute(L+1,1)

FFN:
recv(L,0), compute(L,0), send(L,0),
recv(L,1), compute(L,1), send(L,1)
```

该版本的 A8F8 C32 单轮 profile guard 达到 128/128、0 failed，设备时间线 860/860 层严格满足 `send0 -> recv0 -> send1 -> recv1`。但端到端吞吐从第一版多流 profile 的151.655 降至 119.810 token/s，回退 20.998%。因此本增量只证明依赖关系和结果正确，不覆盖 2.4 节历史三轮数据，也不创建性能 tag。当时计划验证 connector 级 Graph-safe异步 `send0`；实际采用的同 layer DAG 和结果见 2.6 节。

### 2.6 2026-08-31 同 layer 跨 stage FFN 流水增量

最终需求不是 P8E 的跨 layer 对角流水，而是同一 layer 内 `FFN compute U1` 与 `FFN recv U2` 重叠。P8F 将调度固定为：

```text
Attention parent, layer L:
send U1 -> post recv U1 -> send U2 -> post recv U2 -> record layer-ready

Attention side compute:
wait layer-ready -> compute layer L+1 U1/U2

FFN, layer L:
parent recv U1 -> side compute U1 -> side send U1(wait compute)
parent recv U2 -> side compute U2 -> side send U2(wait compute)
parent join both sends -> layer L+1
```

这里的 `post recv` 是 graph op 已排入设备流，不是 recv 已完成。若要求 Attention 必须等U1 recv 完成后才 send U2，U2 数据就只能在 FFN 完成并回送 U1 后到达，与 `compute U1` / `recv U2` 重叠目标互斥。当前实现保持 Attention Graph HCCL 在 parent capture stream；两个 current-layer recv 都排入后才记录 `ready` event，下一层任一 side compute 都必须等待该 event，因此不会重新形成跨 layer 对角流水。

最终 16 卡 C32 profiler guard 为 128/128、0 failed、140.116 token/s、p50 TPOT 159.936 ms。相对 P8E 的 119.810 token/s 提升 16.948%，相对第一版允许跨 layer 对角重叠的 151.655 token/s 仍低 7.609%。FFN step 69/70 的 86 个 layer-stage 配对中，70 个满足 `recv U1 < recv U2 < send U1`；U2 recv 有 80.477% 落在 U1 FFN 执行窗口，排除直接收发 AICPU 后真实 device kernel 覆盖 80.327%。该单轮结果证明 DAG 生效，不是正式 P2 性能收益，`P8D-PERF-001` 继续 Open。

#### 2.6.1 无 profiler 三点三轮结果

同一 P8F worktree 随后完成 C16/C32/C64 各三轮，输入 1024、精确输出 128、每轮
`4 x concurrency` 请求。九轮共 1344/1344 成功、0 failed、172032 个 output token；
warmup、U2、fatal log、NPU monitor、shutdown 和 cleanup gate 全部通过。

| 并发 | throughput 原始值（token/s） | 均值 | CV | p50 TPOT 均值 | p99 TTFT 均值 | token/s/NPU | 稳定性 |
|---:|---|---:|---:|---:|---:|---:|---|
| 16 | 112.409 / 138.771 / 102.291 | 117.824 | 13.051% | 99.871 ms | 9485.457 ms | 7.364 | 未通过 |
| 32 | 133.647 / 116.987 / 128.176 | 126.270 | 5.491% | 177.366 ms | 25150.606 ms | 7.892 | 通过 |
| 64 | 39.580 / 41.137 / 40.490 | 40.403 | 1.581% | 1778.321 ms | 237559.907 ms | 2.525 | 通过但性能不可接受 |

整体 `passed=false` 的直接原因是 C16 CV 超过 10%。C64 的 CV 虽通过，但吞吐相对 C32
下降 68.003%，不能把“低波动”写成“性能通过”。三轮 C64 的首批 64 请求单独折算仍有
109.815 / 108.329 / 102.663 token/s；详细 ITL 显示后续三个补位波次从首波约
`0.274-0.299 s` 放大到约 `1.57-1.99 s`。stage `(4,4)` 已在启动时 capture，且无
live eager fallback，问题集中在满载持续运行和请求补位之后。

现有 C32 profile 已把 16/86 个未重叠 FFN 配对定位到远端 Attention U2 send 尚未 ready；
但无 profiler C64 数据不能证明二者因果。后续需要 C64 单波次对照和第二波补位窗口的
Attention/FFN 同窗口 profile，并分别核对 prefill、DP 补位不齐、MoE all-to-all 与 layer
barrier。第一版跨 layer 多流和 P8E 仍缺同源码无 profiler 三点三轮对照，所以当前不能
宣称 P8F 的正式净收益。

证据目录：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_same_layer_cross_stage_perf_c16_c32_c64_r3_20260831
```

### 2.7 2026-08-31 混合 DAG 与物理 stream 解耦候选

在 P8F 已验证版本之上，当前工作树新增混合 DAG 候选。它不改变 A/F wire protocol，也不
立即增加物理 stream；先把依赖关系从“整层双 stage barrier”改为“同 stage receive 完成后
释放下一层同 stage 计算”，并把逻辑 compute/send/recv 到物理 stream 的映射集中到
`HCCLAttentionGraphStreamPlan`。

```text
逻辑依赖（每个 stage 独立）：
A_compute(L,S) -> A_send(L,S) -> A_recv(L,S) -> A_continue+compute(L+1,S)

V1 物理映射：
compute -> connector-owned side compute stream
send    -> parent capture stream
recv    -> parent capture stream

V1 parent issue 顺序：
send(L,S0) -> recv(L,S0) -> send(L,S1) -> recv(L,S1)
```

Graph 构造仍是单线程 layer-major；设备执行时，`compute(L+1,S0)` 只等待
`recv_done(L,S0)`，因此可与 parent 上随后执行的 S1 exchange 重叠。S1 下一层计算仍等待
`recv_done(L,S1)`。最终 layer 的 continuation 计算通过 `join_attention_graph_compute` 显式
汇回 parent，避免 Graph 返回前仍有 side-stream 工作。

逻辑 DAG 不直接持有“第几条物理流”的假设。当前 V1 计划只把 compute 映射到 side
stream；后续 Attention 三流实验可把 send/recv 映射到独立 stream，但必须先通过最小 HCCL
Graph component 的连续 capture/replay 门禁。`AFD_HCCL_GRAPH_U2_HYBRID_DAG=0` 或
`--graph-u2-hybrid-dag off` 可在同一源码恢复 P8F 的整层 ready barrier，用于公平对照。

真实 A8F8 单轮 smoke 已在 2026-08-31 完成：`FULL_DECODE_ONLY`、U2、batch32，10/10
串行 prompt 逐 token 一致，batch32 `valid=true`、32 choices 完整，8 个 Attention worker
均观察到 ACL Graph replay，双 stage、日志、关闭顺序和 NPU 清理门禁通过。证据目录：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_hybrid_dag_smoke_20260831_codex3
```

该 smoke 不等同于 30/30 + batch 1/8/32 的完整 F0；完整 F0、最小 Graph component 和
A1F1 连续 capture/replay 仍需补齐。随后完成的 CANN 9.0.0 C32 on/off 与双侧 profile 见
2.7.1 节。

#### 2.7.1 CANN 9.0.0 C32 性能与 profile 对照

本轮固定 afd-plugin `6c636961d3791df545ae065811368dc8beb1e4ab`、vLLM
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、vLLM-Ascend
`3da28f9414583d2d0b672a8f06d1fae142404bda`，运行栈固定为：

```text
CANN: /mnt/workspace/code/.ascend/cann-9.0.0/cann-9.0.0
ATB:  /mnt/workspace/code/.ascend/cann-9.0.0/nnal/atb
venv: /mnt/workspace/code/.venvs/afd-v023-vllm-cann
```

两组均为 A8F8/TP1、`FULL_DECODE_ONLY`、Graph/U2、MTP off、async scheduling off、C32、
input 1024、output 128、128 请求/轮、3 轮；`graph-u2-compute-overlap=on` 保持不变，
唯一实验变量是 `graph-u2-hybrid-dag=on/off`。

| 模式 | 三轮 output throughput，token/s | 均值 | CV | token/s/NPU | p50 TPOT | p90 TPOT | p99 TPOT |
|---|---|---:|---:|---:|---:|---:|---:|
| 整层 barrier，hybrid off | 121.695 / 142.886 / 114.236 | 126.272 | 9.611% | 7.892 | 186.525 ms | 220.264 ms | 239.696 ms |
| 混合 DAG，hybrid on | 135.374 / 141.692 / 125.861 | 134.309 | 4.845% | 8.394 | 172.687 ms | 211.340 ms | 239.833 ms |

hybrid on 相对 off 的吞吐均值为 `+6.365%`，p50/p90 TPOT 为 `-7.419%/-4.052%`，
p99 TPOT 基本不变（`+0.057%`）。p50/p90/p99 TTFT 分别为 `-1.415%/-17.157%/-12.170%`。
两组 768/768 正式请求成功，U2 `observed_two_stages=true`；warmup、fatal log、双侧退出、
NPU monitor 和 cleanup gate 均通过。三组同序号差值为 `+11.240%/-0.836%/+10.177%`；
虽然两组 CV 都通过预设 10% 门槛，3 轮样本仍不足以把 `+6.365%` 当成跨负载精确收益。

双侧 profile 固定 Attention DP0、FFN DP0、`skip_first=64, wait=2, warmup=1,
active=20, repeat=1`、`with_stack=false`。raw `profiler_info_0.json` 和离线 parser 均为
CANN 9.0.0，四份 trace 都生成非空 `step_trace_time.csv`、`kernel_details.csv`、
`communication.json`、`trace_view.json` 和数据库。Attention matched steady steps 67-80 的
中位数如下，时间单位已从 us 转为 ms：

| Attention DP0 指标，中位数 | hybrid off | hybrid on | 变化 |
|---|---:|---:|---:|
| Computing | 38.855 ms | 38.940 ms | +0.218% |
| Communication(Not Overlapped) | 18.855 ms | 9.405 ms | -50.118% |
| Overlapped | 14.795 ms | 29.277 ms | +97.882% |
| Overlapped / Communication | 43.939% | 75.646% | +31.707 pp |
| Stage | 53.216 ms | 43.128 ms | -18.958% |
| Bubble | 31.948 ms | 32.048 ms | +0.313% |

这组 trace 说明吞吐方向与目标机制一致：Attention compute 基本不变，收益来自更多通信被
下一层同 stage compute 覆盖。FFN 的 active step 边界包含跨请求 receive 等待，on/off
分别只有 7/6 个非零计算 step，不能直接比较 FFN 20-step aggregate；FFN trace 只作为
Graph 执行和通信因果证据。profile-on 的 128/128 workload 和 raw profile gate 通过，但
Attention 正常 shutdown 后 CANN 9.0.0 TBE 队列线程产生 `EOFError`，使通用 fatal-log gate
失败；异常发生在 workload 和 raw 数据落盘之后，仍按门禁失败披露，不把该单轮 profile
吞吐用于正式收益。

证据目录：

```text
/mnt/workspace/validation/dsv4_afd_v023_cann900_hccl_graph_u2_hybrid_off_c32_r3_20260831_codex1
/mnt/workspace/validation/dsv4_afd_v023_cann900_hccl_graph_u2_hybrid_on_c32_r3_20260831_codex1
/mnt/workspace/validation/dsv4_afd_v023_cann900_hccl_graph_u2_hybrid_off_c32_profile_20260831_codex1
/mnt/workspace/validation/dsv4_afd_v023_cann900_hccl_graph_u2_hybrid_on_c32_profile_20260831_codex1
```

## 3. 工程底座：vLLM 0.23 和 plugin 兼容边界

### 3.1 问题背景

AFD 早期功能依赖特定 vLLM/vLLM-Ascend 分支接口。升级到 vLLM 0.23 后，EngineCore、DP metadata、forward context、Graph 和模型加载接口均发生变化。如果继续在上游仓库直接堆 patch，升级成本、回退成本和能力归属都会失控。

### 3.2 修改点

- 在 `afd_plugin/compat/` 中收敛 vLLM 和 vLLM-Ascend 差异，包括 EngineCore、DP 协调、配置校验、force load balance 和 NPU forward context。
- 适配 v0.23 的 DP metadata、async scheduling 与 DBO/U2 数据结构。
- 新增目标栈环境审计、同栈原生 golden 生成和 native baseline 工具。
- 保持 vLLM 与 vLLM-Ascend 目标工作树不修改，AFD 行为由 plugin 注册、继承和小范围兼容 patch 注入。

### 3.3 为什么修改及意义

最重要的变化不是 API 改名，而是建立“固定上游 + 可卸载插件”的责任边界：

- 上游升级差异集中在兼容层，模型与 connector 不需要到处判断版本。
- AFD 开关关闭时仍使用固定上游语义，便于做 native 对照和回退。
- 同栈 golden 消除了“上游版本变化导致 token 差异”这一类假失败。
- runtime manifest 记录源码、参数和工作树状态，使验证结果可复现。

### 3.4 关键代码

主要入口：

- `afd_plugin/compat/patches/engine_core.py`：v0.23 EngineCore/DP 行为适配。
- `afd_plugin/compat/patches/npu/force_load_balance.py`：目标栈 NPU DP load-balance 兼容。
- `afd_plugin/compat/npu/feature_validation.py::fail_if_unsupported_npu_afd_features`：能力组合统一门禁。
- `tools/dsv4/check_v023_vllm_cann_runtime.sh`：版本、源码和运行环境审计。
- `tools/dsv4/generate_golden.py`：生成同栈非 AFD golden。

能力不是在深层运行时“试试看”，而是在启动期显式限定：

```python
if tensor_parallel_size not in (1, 2):
    raise RuntimeError(
        "DeepSeek-V4 AFD supports only tensor_parallel_size=1 or 2"
    )
```

### 3.5 验证结果与支持边界

- 同栈原生 10 条 prompt 连续 3 轮稳定。
- AFD eager/U2 达到 30/30 token exact，并通过 batch 1/8/32、双 stage、正常停止、fatal log 和 NPU cleanup。
- v0.23 目标栈 C32 三轮：U1 `17.082 token/s`，U2 `12.582 token/s`，U2 回退 `26.342%`。这里只冻结功能兼容性，不创建性能 tag。
- 功能 tag：`dsv4-afd-v023-vllm-cann-eager-u2-functional-v1`。

## 4. 特性一：AF 分离

### 4.1 特性原理

AF 分离把一个 decoder layer 的 Attention 计算与 MoE/Experts 计算放到两组独立 worker 上。DeepSeek-V4 不能沿着整个 FFN 子层生硬切开，因为 HC pre/post 和 residual 状态都在 Attention 侧参与连续计算；最终拆分点选择远端 MoE 边界：

```text
Attention role                         FFN role
--------------                         --------
HC-Attention + Attention
HC-FFN-pre + Norm
        |
        | hidden [T, H] + 首层 input IDs
        v
                                  Remote MoE/Experts
        ^
        | FFN output [T, H]
        |
HC-FFN-post + 下一层
```

模型参数也按角色拆分：Attention worker 不构造 MoE，FFN worker 不构造 Attention/HC。数据面通过 `P2pHcclAFDConnector` 做 A2F/F2A 传输，控制面提前同步当前 step 的 DP token metadata。

### 4.2 端到端流程

1. 两侧根据 `additional_config['afd'].role` 构造角色化模型并只加载本角色权重。
2. Attention 在 step 开始时发送 input IDs；FFN layer 0 接收，前三层 hash routing 复用。
3. 每层 Attention 完成 HC pre、Attention、HC FFN pre 和 Norm，得到二维 hidden。
4. connector 把 hidden 发给对应 FFN peer；FFN 执行远端 MoE。
5. FFN 按原 peer/slice 返回 output；Attention 完成 HC FFN post，进入下一层。
6. Attention 正常退出后，控制面通知 FFN loop 结束并清理 process group、buffer 和 worker。

### 4.3 为支持该特性需要的适配

- 模型适配：定义 Attention/FFN 的模块构造边界、forward 边界和权重所有权。
- runner 适配：Attention 继续驱动完整请求，FFN 进入 connector-driven worker loop。
- 通信适配：建立 IDs、hidden、output 三类消息，以及 A2F/F2A process group。
- metadata 适配：把 layer、stage、phase、token count 和 shape 作为明确协议。
- 生命周期适配：两侧并发初始化，Attention-first shutdown，FFN 无 HTTP endpoint。
- 能力门禁：只接受已经验证的 connector、并行度和执行模式。

### 4.4 修改点与代码对应

| 适配点 | 代码位置 | 具体修改 | 作用 |
|---|---|---|---|
| 角色化模型注册 | `afd_plugin/model_executor/models/deepseek_v4.py::AFDDeepseekV4ForCausalLM` | 基于原生 DSV4 模型构造 AFD wrapper | 让 plugin 接管 DSV4 AFD 模型入口 |
| 普通权重归属 | `deepseek_v4.py::_checkpoint_weight_roles`、`_iter_role_weights` | 按 checkpoint key 过滤 Attention/FFN 权重 | 防止两侧重复加载或漏载参数 |
| 角色化 decoder layer | `deepseek_v4.py::AFDDeepseekV4DecoderLayer.__init__` | Attention 构造 Attention/HC 和无参数 proxy；FFN 只构造 MoE | 固化模型拆分边界并降低 HBM |
| A2F 前向边界 | `AFDDeepseekV4DecoderLayer.forward_attention_to_remote_ffn` | 执行到远端 MoE 输入并保留 HC continuation | 跨 AF 只传二维 hidden |
| F2A 完成边界 | `AFDDeepseekV4DecoderLayer.complete_remote_ffn` | 收到远端 output 后执行 HC post | 保持原模型 residual/HC 数学语义 |
| 远端代理 | `deepseek_v4.py::AFDDeepseekV4RemoteMoEProxy.forward` | 把本地 `mlp()` 调用转换为 connector send/recv | 对上层模型隐藏远端调用细节 |
| 传输协议 | `afd_plugin/connectors/metadata.py::AFDTransferMetadata`、`AFDTransferState` | 描述 layer/stage/phase、tensor shape 和传输状态 | 将隐式消息顺序变为可校验协议 |
| HCCL 数据面 | `afd_plugin/connectors/npu/p2p_hccl.py::P2pHcclAFDConnector` | 实现 IDs、hidden、output 的 send/recv 和 buffer 管理 | 提供标准 HCCL P2P AF 通道 |
| FFN 执行循环 | `afd_plugin/v1/worker/npu/ffn_model_runner.py::execute_connector_driven_step` | FFN 不接请求，由 connector metadata 驱动每个 step | 适配无 scheduler/无 HTTP 的 FFN 角色 |
| FFN EngineCore | `afd_plugin/compat/patches/engine_core.py::_initialize_ffn_engine_core`、`_run_ffn_busy_loop` | 建立 FFN 专用 no-op scheduler 和 worker loop | 支持独立 FFN 服务生命周期 |
| 支持边界 | `afd_plugin/compat/npu/feature_validation.py::_fail_if_unsupported_deepseek_v4_features` | 启动期拒绝未验证组合 | 避免深层 HCCL recv 挂死 |

#### 4.4.1 关键代码串讲

修改点一：`_checkpoint_weight_roles` 用 checkpoint key 明确普通 decoder 权重归属。`layers.<n>.ffn.*` 只属于 FFN，其他 decoder 权重属于 Attention，MTP key 留给独立 loader：

```python
def _checkpoint_weight_roles(name: str) -> frozenset[str]:
    normalized = name.removeprefix("model.")
    if normalized.startswith("mtp."):
        return _NO_ROLE

    parts = normalized.split(".")
    if len(parts) >= 3 and parts[0] == "layers" and parts[1].isdigit():
        return _FFN_ROLE if parts[2] == "ffn" else _ATTENTION_ROLE
    return _ATTENTION_ROLE
```

修改点二：`AFDDeepseekV4DecoderLayer.__init__` 在构造阶段完成角色拆分。Attention 侧用无参数 remote proxy 替代本地 MoE，FFN 侧不构造 Attention，只保留真实 MoE：

```python
if self.afd_role == "attention":
    self.self_attn = attention_class(...)
    self.mlp = AFDDeepseekV4RemoteMoEProxy(
        layer_idx=self.layer_idx,
        phase="mtp" if is_draft_layer else "decoder",
    )
    # Attention 侧继续构造 Norm 和 HC 参数。
else:
    self.self_attn = native.PPMissingLayer()
    self.mlp = native.DeepseekV4MoE(
        config=config,
        parallel_config=parallel_config,
        quant_config=quant_config,
        prefix=f"{prefix}.mlp",
        is_draft_layer=is_draft_layer,
    )
```

修改点三：`AFDDeepseekV4RemoteMoEProxy.forward` 保持上层的 `self.mlp(...)` 调用不变，但把它变为一次 A2F/F2A RPC；首层额外携带 input IDs：

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if self.phase == "mtp":
        return self._send_and_receive(hidden_states)
    input_ids = None
    if self.layer_idx == 0:
        input_ids = getattr(get_forward_context(), "input_ids", None)
        if input_ids is None:
            raise RuntimeError("DSV4 layer 0 requires input_ids in the forward context")
    return self._send_and_receive(hidden_states, input_ids=input_ids)
```

修改点四：真正的拆分边界在 HC FFN pre 之后、HC FFN post 之前。Attention 保存 continuation，收到远端 MoE output 后再恢复原数学路径：

```python
hidden_states = self.mlp(hidden_states)  # remote proxy
return hidden_states, (residual, post, comb)

def complete_remote_ffn(self, ffn_output, continuation):
    residual, post, comb = continuation
    return self.hc_post(ffn_output, residual, post, comb)
```

### 4.5 修改意义

- 参数和计算真正分角色持有，为独立扩缩容 Attention/FFN 奠定基础。
- 通信边界与原模型数学边界一致，避免传输高维 HC 状态。
- 统一 connector 协议成为 Microbatch、Graph、MTP、非等量和 TP2 的共同底座。
- 角色化 worker 和控制面让 FFN 从普通服务进程转为可独立管理的内部计算后端。

### 4.6 验证结果与支持边界

- A1F1/U2 真实 NPU round-trip 通过。
- A8F8 eager U1/U2 各 30/30 golden，batch 1/8/32、退出与清理通过。
- connector 测试覆盖消息顺序、shape、buffer 生命周期、IDs、MTP 和拓扑错误。
- 本文目标路径不使用 CAMP2P 自定义传输 op，也没有引入 `isend/irecv` 或后台通信线程。

详细证据见 [HCCL P2P 验证报告](DEEPSEEK_V4_AFD_HCCL_P2P_VALIDATION_REPORT_ZH.md)。

## 5. 特性二：Microbatch

### 5.1 特性原理

Microbatch 把一个 decode step 的 token 拆成多个 stage。U1 表示整批一次执行，U2 表示拆成两个 stage。AF 场景的目标是让不同 stage 分别处于 Attention、A2F、FFN 和 F2A 阶段，从而形成流水；它不是简单把输入 tensor 切成两半。

```text
时间 --->
stage 0: Attention L0 -> A2F -> FFN L0 -> F2A -> Attention L1 ...
stage 1:                 Attention L0 -> A2F -> FFN L0 -> F2A ...
```

因为两侧使用同一组 HCCL communicator，Attention 和 FFN 必须对 layer、stage 和消息类型做出完全相同的排序决策。

### 5.2 端到端流程

1. Attention runner 根据 token/request metadata 判断是否满足 U2 门限。
2. 普通 decode 可按 token 切分；MTP verify 必须按请求边界切分，避免拆散相关 target/draft token。
3. DP 控制面汇总每个 rank 的 stage token count；任一 rank 无法形成两个非空 stage 时，全局回退 U1。
4. `AscendUBatchWrapper` 为两个 stage 构造独立 input、forward context 和 metadata。
5. DSV4 模型用单 host 线程按 `layer -> stage 0 -> stage 1` 推进。
6. A2F/F2A 使用独立 NPU stream/event；主 compute stream 到消费点才等待 receive 完成。
7. 两个 stage 的输出按原 token 顺序合并并返回 scheduler。

### 5.3 为支持该特性需要的适配

- 切分适配：普通 token 切分、MTP 请求边界切分、空 stage padding 与全 DP 一致决策。
- metadata 适配：每个 stage 保留独立 token count、query offset、Attention metadata 和 forward context。
- 模型适配：从 stage-major 多线程改为 layer-major 单线程。
- connector 适配：为 stage 保存独立 pending state、buffer 和 stream event。
- Graph 适配：warmup/capture/replay 必须复用同一个 layer-major 顺序。
- 输出适配：合并 stage 输出及 TP/DP 下的 gather 结果。

### 5.4 修改点与代码对应

| 适配点 | 代码位置 | 具体修改 | 作用 |
|---|---|---|---|
| U2 开关和门限 | `afd_plugin/v1/worker/npu/ubatch_utils.py::check_enable_ubatch` | 判断当前 step 是否启用 ubatching | 避免不满足条件的请求进入 U2 |
| 普通切分 | `ubatch_utils.py::create_ubatch_slices`、`maybe_create_ubatch_slices` | 生成两个 token slice 并处理空 stage | 建立 U2 输入边界 |
| MTP 请求边界切分 | `ubatch_utils.py::create_request_boundary_ubatch_slices` | 按完整 request 分配 target tokens | 保持 speculative verify 数值语义 |
| Attention metadata 切分 | `ubatch_utils.py::split_attn_metadata` | 重建每个 stage 的 query/slot/position metadata | 保证 Attention 使用正确 token 视图 |
| U2 wrapper | `afd_plugin/v1/worker/npu/npu_ubatch_wrapper.py::AscendUBatchWrapper` | 构造、运行、合并两个 stage | 对接 vLLM model runner 的统一入口 |
| layer-major 调度 | `AscendUBatchWrapper._run_ubatches_layer_major`、`AFDDeepseekV4Model.forward_ubatches_layer_major` | 单线程按 layer 后 stage 推进 | 对齐 A/F HCCL op 顺序并减少 GIL 交接 |
| stage 控制 metadata | `attention_model_runner.py::_build_ubatch_control_metadata`、`_build_attention_metadata_with_ubatches` | 生成跨 DP 的 stage token 决策 | 防止部分 rank U1、部分 rank U2 |
| A2F stream | `p2p_hccl.py::_enqueue_attention_send` | compute event 后在独立 stream 提交 send | 尝试重叠 Attention 与通信 |
| F2A 延迟等待 | `p2p_hccl.py::recv_ffn_output`、`wait_for_attention_stage_receive` | 保存 receive event，到消费 output 时再 wait | 保证正确性同时减少过早同步 |
| FFN stage 执行 | `ffn_model_runner.py::execute_model`、`_ffn_forward_connector_driven` | 按控制面 metadata 执行每个 stage | 与 Attention 侧保持同一消息序列 |

#### 5.4.1 关键代码串讲

修改点一：MTP/U2 不能机械地从 token 中间切开请求。`create_request_boundary_ubatch_slices` 只枚举请求边界，并选择两个 stage token 数最接近的位置：

```python
split_req = min(
    range(1, num_reqs),
    key=lambda req_idx: (
        abs(int(cu_num_tokens[req_idx]) * 2 - total_tokens),
        abs(req_idx * num_ubatches - num_reqs),
    ),
)
split_token = int(cu_num_tokens[split_req])
return [
    UBatchSlice(slice(0, split_req), slice(0, split_token)),
    UBatchSlice(slice(split_req, num_reqs), slice(split_token, total_tokens)),
]
```

修改点二：前置切分无法形成两个非空 stage 时回退 U1；`_build_ubatch_control_metadata` 继续把全局 DP token vector 投影到两个 stage，并对已经进入 U2 协议却仍出现空 stage 的异常状态 fail-fast，避免两侧进入不同 HCCL 序列：

```python
stage_counts = (
    global_token_counts.clamp(max=stage_stop) - stage_start
).clamp(min=0)
if torch.any(stage_counts == 0):
    raise RuntimeError(
        "DeepSeek-V4 AFD U2 does not support an empty stage on any DP rank; "
        f"stage={stage_idx} counts={stage_counts.tolist()}"
    )
metadata[stage_idx] = AFDDPMetadata(
    num_tokens_across_dp_cpu=stage_counts,
)
```

修改点三：核心调度从 stage-major 多线程改成单线程 `layer -> stage`。每个 stage 消费上一层 F2A 结果后，立即发起当前层远端 MoE：

```python
for layer_offset, layer in enumerate(
    islice(self.layers, self.start_layer, self.end_layer)
):
    for stage_idx, (item, forward_context) in enumerate(
        zip(ubatch_metadata, stage_contexts, strict=True)
    ):
        with override_forward_context(forward_context):
            if layer_offset > 0:
                wait_for_receive(
                    stage_idx=stage_idx,
                    tensor=hidden_ubatches[stage_idx],
                )
                pending_layer = pending_layers[stage_idx]
                continuation = pending_continuations[stage_idx]
                if pending_layer is None or continuation is None:
                    raise RuntimeError("stage has no pending remote FFN layer")
                hidden_ubatches[stage_idx] = pending_layer.complete_remote_ffn(
                    hidden_ubatches[stage_idx], continuation
                )
            hidden_states, continuation = layer.forward_attention_to_remote_ffn(
                item.positions, hidden_ubatches[stage_idx], None, llama_4_scaling
            )
```

修改点四：F2A receive event 不在通信提交后立刻等待，而是在下一层真正消费该 tensor 时由 compute stream 建立依赖：

```python
dependency = self.attention_receive_dependencies.pop(stage_idx, None)
if dependency is None or dependency.tensor is not tensor:
    raise RuntimeError("invalid deferred F2A receive")
compute_stream = torch.npu.current_stream()
dependency.event.wait(compute_stream)
self._record_stream(tensor, compute_stream)
```

因此主循环的稳定顺序是：

```text
for layer in decoder_layers:
    for stage in (stage0, stage1):
        Attention -> A2F send -> FFN -> F2A recv -> HC post
```

### 5.5 修改意义

- 建立了 AF 计算/通信流水的功能基础，后续 Graph/U2 和 MTP/U2 都复用同一 stage 语义。
- 全 DP 一致决策把潜在 HCCL 死锁转为统一 U1 fallback。
- layer-major 单线程让执行顺序确定，可用于 NPU Graph capture/replay。
- 独立 stream/event 为性能优化保留空间，同时不改变标准同步 HCCL API。

### 5.6 验证结果与支持边界

- eager U2 的 golden、batch、双 stage、空闲恢复、生命周期和 cleanup 已通过。
- P8D 相对 comm-stream 版本 P8C 提升 `10.099%`，但相对 U1 仍回退 `46.197%`。
- 当前 U2 是功能能力，不是性能基线；U3 始终 fail-fast。

## 6. 特性三：NPU Graph（ACL Graph）

### 6.1 特性原理

NPU Graph 把一段稳定的 NPU 算子与通信序列在 warmup 后 capture，在线请求只 replay，减少 Python/host 逐算子下发。AF 场景的图不只包含模型算子，还包含 A2F/F2A HCCL op；因此 Attention 和 FFN 必须同时选择 eager、capture 或 replay，并使用完全相同的消息顺序。

```text
启动期：shape bucket -> A/F 同步 warmup -> capture -> graph cache
在线：  metadata -> 生成 graph key
          | key hit                 | key miss
          v                         v
       A/F 双侧 replay           A/F 整步 eager fallback
```

### 6.2 端到端流程

1. 启动期根据 capture size、U1/U2 stage layout、LoRA/MTP signature 构造 graph key。
2. A/F 两侧按同一 layer-major 顺序 warmup；connector 注册 graph-visible HCCL `_send/_recv`。
3. capture 时 Graph-visible HCCL 保留在 parent capture stream；U2 模型计算通过 event
   fork 到 side compute stream。混合 DAG 下，每个 stage 的下一层计算由自己的
   `recv_done` 释放，最终 continuation 再 join 回 parent。
4. input IDs 和需要在 CPU 解析的动态 header 放在 Graph 外准备，避免固化请求数据。
5. 在线 step 命中 key 时 A/F 双侧 replay；任一侧 key miss 时整步走 eager。
6. full-draft 模式分别维护 target decoder Graph 和 MTP draft Graph，按 target 后 draft 的顺序 replay。

### 6.3 为支持该特性需要的适配

- HCCL lowering：将 eager 的 Python send/recv 映射为 torch-npu 可捕获 op。
- 动态 shape：不向 HCCL op 传会被专门化的 token shape。
- Graph key：纳入 stage、每个 peer 的精确 token layout、TP 展开和 MTP signature。
- 数据生命周期：IDs、MTP header 等动态控制数据在 Graph 外预传或写入稳定 buffer。
- A/F 决策同步：禁止 Attention 单边在线 capture，key miss 整步 eager。
- U2 capture：warmup/capture/replay 全部采用 layer-major。
- U2 stream plan：逻辑 compute/send/recv 先映射到统一 plan；V1 只把 compute 放到 side
  stream，HCCL 留在 parent。独立 send/recv stream 仍需先过最小 Graph component 门禁。
- full-draft：target 与 draft 使用独立 cache、key、buffer 和 capture/replay。

### 6.4 修改点与代码对应

| 适配点 | 代码位置 | 具体修改 | 作用 |
|---|---|---|---|
| Graph HCCL send/recv | `afd_plugin/connectors/npu/p2p_hccl.py::_graph_hccl_send`、`_graph_hccl_recv` | 调用 `torch.ops.npu_define._send/_recv`，shape 参数保持 `None` | 让标准 HCCL 语义进入 NPU Graph 且不固化动态 token 长度 |
| eager/Graph 分派 | `P2pHcclAFDConnector._send_tensor`、`_recv_tensor` | 编译/capture 使用 Graph op，普通请求使用 `dist.send/recv` | 一套 connector 覆盖两种执行模式 |
| Attention Graph stream plan | `P2pHcclAFDConnector.attention_graph_stream_plan`、`HCCLAttentionGraphStreamPlan` | 逻辑 compute/send/recv 独立映射到物理 stream；V1 为 side/parent/parent | 后续改物理 stream 不重写模型 DAG |
| Attention Graph fork/join | `P2pHcclAFDConnector.attention_graph_compute`、`wait_for_attention_graph_compute`、`join_attention_graph_compute` | compute 等待 stage-local ready/recv_done；send 等 compute_done；最终结果 join parent | 建立可 capture 的跨 stream 依赖闭环 |
| Attention Graph/U2 混合 DAG | `afd_plugin/model_executor/models/deepseek_v4.py::_forward_ubatches_graph_compute_pipeline` | 首层从 parent fork；后续层的每个 stage 只等待同 stage 上一层 recv_done | 允许 `compute(L+1,S0)` 与 S1 exchange 重叠 |
| 远端 MoE 两阶段接口 | `afd_plugin/model_executor/models/deepseek_v2.py::RemoteFFNProxy.dispatch_remote_ffn`、`receive_remote_ffn` | send 与 receive 之间返回 transfer handle | 允许 layer-major 调度跨 stage 插入工作 |
| FFN Graph/U2 DAG | `afd_plugin/v1/worker/npu/ffn_model_runner.py::_ffn_forward` | recv/send 留 parent；每收到一个 stage 就排 MoE，parent 等 compute event 后先 send 同 stage，再 recv 下一 stage | 建立无环的 `recv0 -> compute0 -> send0 -> recv1` |
| 公平对照开关 | `p2p_hccl.py::_graph_u2_compute_overlap_enabled`、`_graph_u2_hybrid_dag_enabled`，以及 performance runner 对应 CLI | 分别控制 side compute 与 stage-local 混合 DAG，均默认开启 | 同源码比较串行、P8F barrier 和混合 DAG |
| Graph 运行状态 | `afd_plugin/v1/worker/cuda_graph.py::graph_run_mode` | 统一区分 warmup、capture、replay、eager | 防止两侧对当前 step 状态理解不一致 |
| FFN decoder key | `cuda_graph.py::make_ffn_graph_key` | 保存 stage 和每个 Attention peer 的精确 token layout | 防止同聚合、不同 peer shape 错图复用 |
| MTP draft key | `cuda_graph.py::make_mtp_ffn_graph_key` | 合并 target stages，并保留 draft 的 peer layout | 为 full-draft Graph 建立独立 shape 身份 |
| Attention U2 Graph | `npu_ubatch_wrapper.py::_capture_ubatches_layer_major`、`_replay_mla_graph` | 用单线程 layer-major capture/replay 两个 stage | 保持 Attention/FFN HCCL op 顺序一致 |
| FFN Graph cache | `ffn_model_runner.py::_make_graph_key`、`_capture_graphs`、`capture_model` | 创建/命中 decoder Graph，重复 key 时双侧 replay | 避免 Attention replay 而 FFN 无匹配 recv |
| full-draft cache | `ffn_model_runner.py::_make_mtp_graph_key`、`_capture_mtp_graphs`、`_replay_mtp_graph` | target 与 draft 分离 capture/replay | 避免 MTP virtual layer 复用 decoder 图 |
| 能力门禁 | `feature_validation.py::_fail_if_unsupported_deepseek_v4_features` | 限定 `FULL_DECODE_ONLY`、U1/U2 和已验证组合 | 对 Graph/U3、TP2 最大组合等显式 fail-fast |

#### 6.4.1 关键代码串讲

修改点一：Graph 内的 HCCL 不能直接沿用会专门化动态 shape 的 tracing wrapper，因此 `_graph_hccl_send/_recv` 直接调用可捕获 op，并把两个可选 shape 参数设为 `None`：

```python
torch.ops.npu_define._send.default(
    tensor,
    dst,
    ranks,
    pg_tag,
    0,
    None,
    None,
)
```

普通 `dist.send/recv` 与 graph-visible `_send/_recv` 的差别不是 HCCL wire protocol，而是capture 语义。普通接口在 Python 调用时立即提交并等待；Graph op 在 capture 时成为图节点，replay 时由图恢复。connector 必须在同一位置明确分派：

```python
if self._graph_transport_active():
    _graph_hccl_send(send_tensor, dst=dst, group=group)
    return
dist.send(send_tensor, dst=dst, group=group)
```

修改点一在当前混合 DAG 中进一步拆成逻辑事件和物理 stream plan。首层仍从 parent fork；
后续层直接等待上一层同 stage 的 `recv_done`：

```python
ready_event = events.ready
if wait_for_receive_layer_idx is None:
    events.ready.record(parent_stream)
else:
    ready_event = previous_stage_events.recv_done

with torch.npu.stream(compute_stream):
    ready_event.wait(compute_stream)
    for tensor in tensors:
        self._record_stream(tensor, compute_stream)
    yield
    events.compute_done.record(compute_stream)

events.compute_done.wait(send_stream)
send(...)
events.send_done.record(send_stream)

events.send_done.wait(recv_stream)
recv(...)
events.recv_done.record(recv_stream)
```

当前 layer-major 主机调度先为两个 stage 排首层 Attention compute，再按 stage 构造
join/send/receive。整层 exchange 的 Graph 节点构造后，再构造下一层 compute 节点；设备是否
可并发由 event DAG 决定，不由 Python 调用先后决定：

```python
for stage_idx in stage_ids:
    enqueue_attention_compute(layer=first_layer, stage_idx=stage_idx)

for layer_offset, layer in enumerate(layers):
    for stage_idx in stage_ids:
        wait_for_compute(layer_idx=layer.layer_idx, stage_idx=stage_idx, ...)
        transfer = layer.dispatch_remote_ffn(hidden[stage_idx])
        hidden[stage_idx] = layer.receive_remote_ffn(transfer)

    if layer_offset + 1 < len(layers):
        for stage_idx in stage_ids:
            enqueue_attention_compute(
                layer=layers[layer_offset + 1],
                stage_idx=stage_idx,
                wait_for_receive_layer_idx=layer.layer_idx,
            )
```

FFN 侧 parent 每收到一个 stage 就把 MoE 排到 compute stream，并在进入下一 stage receive之前等待 compute event、发送当前 stage：

```python
for stage_idx in stage_ids:
    payload = self.connector.recv_attn_output(...)
    recv_event.record(torch.npu.current_stream())
    with torch.npu.stream(self.ffn_compute_stream):
        recv_event.wait(self.ffn_compute_stream)
        output = self.model.compute_ffn_output(...)
        compute_event.record(self.ffn_compute_stream)
    compute_event.wait(torch.npu.current_stream())
    _send_ffn_output(self.connector, output, context, stage_idx=stage_idx)
```

该顺序是当前 parent-stream Graph HCCL 下的无环版本。恢复第一版 `recv0, compute0, recv1, compute1, send0, send1` 会与 Attention 的`send0, recv0, send1` 形成环：Attention 等待 FFN `send0`，FFN Graph 又在 `send0` 前排入尚无对端 `send1` 的 `recv1`。实际试验在 capture 期等待 pending HCCL work 超过 60 秒。

外部 `cann-recipes-infer` 提交[`c6c7315f`](https://gitcode.com/yijie19/cann-recipes-infer/commit/c6c7315f4bc0cd2dd1646540bdd1a4799e36a561?ref=dsv4-asyn)可参考的是按 microbatch 分 event、`record_stream` 和 recv/compute/send DAG；它使用自己运行时中的普通 `dist.send/recv`，不能直接替换本插件的 graph-visible `_send/_recv`。原[CAMP2P U2 指南](CAM_P2P_CONNECTOR_USER_GUIDE.md) 可参考 stage 独立 communicator 和消息身份，但当前能力门禁明确拒绝 CAMP2P Graph/U2。本次实现仅修改 afd-plugin，复用 torch-npu 已有的NPUGraph 多 stream/event 与 `_send/_recv`，不需要修改 vLLM-Ascend。失败实验、接口差异和设备时间线详见[Graph/U2 专项报告](DEEPSEEK_V4_AFD_HCCL_P2P_GRAPH_U2_VALIDATION_REPORT_ZH.md#8-2026-08-29-graphu2-多流重叠增量)。

修改点二：Graph key 不只保存 FFN 聚合 token 总数，而是按 stage 保存每个 Attention peer 的精确 token layout：

```python
for stage_idx, metadata in sorted(dp_metadata_list.items()):
    values_tuple = _metadata_values_tuple(
        metadata.num_tokens_across_dp_cpu
    )
    if _use_ffn_peer_layout_key(attention_size, ffn_size):
        values_tuple = _expand_attention_values_tuple(
            values_tuple,
            attention_size=int(attention_size),
            fallback=int(fallback),
        )
    key_parts.append((int(stage_idx), values_tuple))
return tuple(key_parts)
```

修改点三：`graph_run_mode` 将 warmup、capture、replay 和 eager 收敛为同一个状态机；没有命中已缓存 key 时直接返回 eager，不在线单边 capture：

```python
if is_warmup:
    return AFDGraphRunMode.WARMUP
if is_graph_capturing:
    return AFDGraphRunMode.CAPTURE
if graph_enabled and graph_exists:
    return AFDGraphRunMode.REPLAY
return AFDGraphRunMode.EAGER
```

修改点四：FFN runner 根据相同 key 选择 replay 或 eager；target 完成后再进入独立 MTP phase：

```python
graph_info = self._acl_graphs.get(graph_key)
run_mode = graph_run_mode(
    is_warmup=is_warmup and graph_enabled,
    is_graph_capturing=is_graph_capturing and graph_enabled,
    graph_enabled=graph_enabled,
    graph_exists=graph_info is not None,
)
if run_mode is AFDGraphRunMode.REPLAY:
    graph_info["graph"].replay()
else:
    self._ffn_forward(
        dp_metadata_list=dp_metadata_list,
        input_ids_by_stage=input_ids_by_stage,
        update_connector_state=False,
    )
```

修改点五：full-draft Graph 为 target decoder 和 MTP draft 建立不同 key namespace 与 cache；即使 peer layout 相同，也不会跨 phase 复用图：

```python
decoder_key = make_ffn_graph_key(...)
graph_key = ("decoder", *self._speculative_graph_signature(), *decoder_key)

peer_layout = make_mtp_ffn_graph_key(...)
mtp_graph_key = ("mtp", *self._speculative_graph_signature(), peer_layout)
graph_info = self._mtp_acl_graphs.get(mtp_graph_key)
```

### 6.5 修改意义

- 把 AF 通信和模型计算作为同一个可 replay 执行单元，降低 host 下发成本。
- 用精确 layout key 和整步 fallback 保证动态请求下的数值与通信正确性。
- NPU Graph 与 Microbatch、MTP、非等量拓扑建立了可组合但可独立门禁的实现边界。
- full-draft Graph 消除了 M2 阶段 draft 只能 eager 的限制，同时保留 target/draft 独立状态。
- 严格 ubatch 闭环消除了 U0 receive 对 U1 send 的直接依赖，但同时暴露 FFN stage-local
  compute/join 的端到端代价；调度正确性和性能收益必须分开验收。
- P8F 把最终目标收敛为同 layer 跨 stage：FFN parent 接收 U2 时 side stream 执行 U1，
  同时用 current-layer 双 recv 之后的 Attention `ready` event 禁止跨 layer 对角计算。
- 当前混合 DAG 候选取消整层双 recv barrier，改成 `recv_done(L,S) -> compute(L+1,S)`；
  stream plan 将逻辑依赖与 V1 的 side/parent/parent 物理映射分开，为后续三流实验保留接口。

### 6.6 验证结果与支持边界

- Graph/U1：30/30 golden，batch 1/8/32、capture/replay、两次冷启动、退出和清理通过。
- Graph/U2：两次独立冷启动各 30/30，真实双 stage；P1 为 `107.189 token/s`，仅是单轮候选信号。
- Graph/U2 多流增量：A8F8 组件 16 进程通过，实模 F0 为 30/30，batch 1/8/32、双 stage、
  fatal log、双侧 rc=0 和 NPU cleanup 通过；单轮 C32 profile 为 128/128、151.655 token/s，
  只作带 profiler 功能 guard。
- CANN 时间线：Attention 20 个 step 每步 43 个 send 与 U1 计算重叠；FFN 稳态 13 个 step
  每步 43 个 receive 和 43 个 send 与计算重叠。说明目标 DAG 已实现，不等于正式吞吐收益。
- FFN 全 rank 诊断：8 个 rank 均为 `Model ID=46`、`OP State=static`，确认 FFN Graph 已开启；
  63 个首层 stage0 dispatch 样本中，预计 rank 等待与 duration 相关系数为 `0.999997`，各 rank
  结束时间偏差中位数 `3 us`。稳态 Graph replay API 发出时间已相差 `3.947-6.026 ms`，因此
  `MoeDistributeDispatchV2` 的毫秒级耗时主要是 MC2 等待 host/rank 到达，不是 MoE 内核持续
  计算。Graph 前最后一次 input-ID receive 完成时间与 replay API 的相关系数为
  `0.974-0.998`，下一轮定位应转向 Attention metadata/input-ID 发送准备及 FFN host 控制路径。
  证据见 Graph/U2 专项报告 8.5.4；直接增加 barrier 不会缩短关键路径。
- 同提交 C32 三轮：Graph/U1 `134.871 token/s`、CV `5.705%`；优化前 Graph/U2
  `125.251 token/s`、CV `10.628%`；优化后 Graph/U2 `139.300 token/s`、CV `5.635%`。
  优化后相对 pre-U2 均值 `+11.217%`、相对 U1 `+3.284%`；三组 1152/1152 请求成功。
  pre-U2 只因 CV 超过 10% 未通过，因此前一个百分比仍是候选证据，不是冻结收益。
- 严格闭环增量：connector 79 项、DeepSeek-V4 构造 18 项、NPU runtime 181 项，共 278 项
  相关回归通过；定向调度用例覆盖 eager、Graph overlap 和 Graph baseline。A8F8 C32 profile
  128/128、0 failed，profile、fatal log、NPU monitor 和 cleanup gate 全部通过。
- 严格闭环 Attention 时间线：20 个稳定 step、每 step 43 层，共 860/860 层精确满足
  `send0 -> recv0 -> send1 -> recv1`，0 mismatch。`send0` 结束到 `recv0` 开始的 p50/p90
  从约 404/510 us 降为 3.5/4.25 us；stage1（第二个 ubatch）send 860/860 次、recv
  840/860 次与下一层 stage0 compute 重叠，840 正好是 20 step x 42 个非末层。
- Attention 稳定窗口中，未重叠通信从 339.654 降至 81.961 ms，下降 75.870%；
  `Computing + Communication(Not Overlapped) + Free` 从 1681.749 降至 1357.006 ms，
  下降 19.310%。但 Bubble 从 316.332 增至 646.229 ms，不能只依据 overlap 分类宣称
  端到端性能收益。
- 严格闭环单轮 profile 吞吐为 119.810 token/s、p50 TPOT 189.514 ms；相对第一版多流
  profile 的 151.655 token/s、157.069 ms，吞吐回退 20.998%。该点只冻结依赖正确性，
  不替代上一版三轮对照，不创建性能 tag。
- 恢复 FFN 跨 stage 重叠的实验在 Graph capture 期等待 pending HCCL work 超过 60 秒；
  原因是 Attention 要等 `recv0` 才 `send1`，而 FFN 把 `recv1` 排在 `send0` 前，形成环依赖。
  实验中止后 cleanup gate 通过，该 P8E 控制版本保留 stage-local 无环顺序。
- 上一条描述的是要求 `recv0` **完成**后才 `send1` 的 P8E 负实验。P8F 改用 issue/post
  语义：Attention parent 依次排入 `send0, recv0, send1, recv1`，FFN side send 等待各自
  compute event，FFN parent 不等 send0 完成便排入 recv1，因此没有该完成态环依赖。
- P8F 相关三个测试文件 279 项通过；最终 A8F8 C32 profile 为 128/128、0 failed、
  140.116 token/s、p50 TPOT 159.936 ms，profile/log/NPU monitor/shutdown/cleanup 全通过。
  相对 P8E +16.948%，相对第一版跨 layer 多流 -7.609%，仍只是单轮 profiler guard。
- P8F FFN step 69/70 共 86 个同 layer 配对：70/86 满足
  `recv U1 < recv U2 < send U1`，86/86 满足 `send U1 < send U2`；U2 recv 在 U1 FFN
  执行窗口中的覆盖为 80.477%，排除直接收发 AICPU 后真实 kernel 覆盖为 80.327%，
  纯 AI/MIX Core 覆盖为 7.268%。step 69 layer 0 的 439.428 us U2 recv 全部与 U1
  `hcom_alltoallv__703_347_1` 重叠。
- 16/86 未重叠配对表示远端 U2 send 当时尚未 ready，不是 FFN 本地先算 U2。P8F 的原始
  结论是不重新引入跨 layer 对角；当前混合 DAG 候选则有意恢复受 stage-local `recv_done`
  约束的对角重叠，必须重新执行死锁、golden、capture/replay 和性能门禁。
- full-draft Graph M7：U1/U2 各 30/30，A4F2 组件 capture/replay，P1 128/128、`27.510 token/s`，仅作功能 guard。
- 仅支持 `FULL_DECODE_ONLY`；Graph/U3 fail-fast。

严格闭环 profile 的整轮运行时间为 2026-08-29 17:28:40-17:39:29（UTC+8），服务启动
耗时 456.248 秒。Attention DP0 实际采集窗口为 17:37:01.679-17:37:37.523，FFN DP0
为 17:36:25.649-17:37:30.595；两侧 schedule 均为 `skip_first=64, wait=2,
warmup=1, active=20, repeat=1`，`with_stack=false`。raw profile 的 `cann_version` 和离线
parser 都是 9.0.0；daemon 期提示改用 offline parser，随后 `ASCEND_PROFILER_OUTPUT` 已成功
生成 `kernel_details.csv`、`step_trace_time.csv`、`trace_view.json` 和数据库。

Attention 20 个稳定 step 的完整时间分解如下。`wall` 按
`Computing + Communication(Not Overlapped) + Free` 计算；Overlap 不重复计入 wall：

| Attention 20-step aggregate | 第一版多流 | 严格闭环 |
|---|---:|---:|
| Computing | 780.187 ms | 778.236 ms |
| Communication(Not Overlapped) | 339.654 ms | 81.961 ms |
| Overlapped | 5.020 ms | 588.615 ms |
| Communication | 344.674 ms | 670.576 ms |
| Free | 561.909 ms | 496.810 ms |
| Bubble | 316.332 ms | 646.229 ms |
| wall | 1681.749 ms | 1357.006 ms |

FFN raw profile 和离线产物有效，但本轮 DP0 step schedule 与 Graph capture/角色启动窗口错位：
step 67-73 才有完整计算，后续 step 主要只剩通信。因此不能把 FFN 的 20-step 汇总与
Attention 或第一版 profile 直接做公平 aggregate；当前只用 FFN trace 验证 Graph 已执行、
stage0 receive 后 MoE 被立即排入 compute stream，以及全服务通信因果。该不确定性必须在
下一轮重新对齐的双侧 profile 中消除；无 profiler 三点三轮已完成，但未通过整体性能门禁。

详细证据：

- [Graph/U1 报告](DEEPSEEK_V4_AFD_HCCL_P2P_GRAPH_U1_VALIDATION_REPORT_ZH.md)
- [Graph/U2 报告](DEEPSEEK_V4_AFD_HCCL_P2P_GRAPH_U2_VALIDATION_REPORT_ZH.md)
- [Full Draft Graph 报告](DEEPSEEK_V4_AFD_HCCL_P2P_MTP_FULL_DRAFT_GRAPH_VALIDATION_REPORT_ZH.md)

严格闭环增量证据目录：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_multistream_profile_c32_20260829
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_stage_interleave_profile_c32_20260829
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_wavefront_profile_c32_20260829
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_stage_interleave_profile_c32_20260829/profiles/attention/4ff038c993bf40219c03182f786a9def_2913927_20260829173735418_ascend_pt
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_stage_interleave_profile_c32_20260829/profiles/ffn/4ff038c993bf40219c03182f786a9def_2913870_20260829173644048_ascend_pt
```

P8F 最终同 layer 跨 stage 证据目录：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_same_layer_cross_stage_profile_c32_20260831
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_same_layer_cross_stage_profile_c32_20260831/profiles/attention/4ff038c993bf40219c03182f786a9def_3220405_20260831122635409_ascend_pt
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_same_layer_cross_stage_profile_c32_20260831/profiles/ffn/4ff038c993bf40219c03182f786a9def_3220188_20260831122557508_ascend_pt
```

开发期间的 capture 负样本和仍含跨 layer 对角调度的中间 side-send 样本分别保存在：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_cross_stage_profile_c32_20260831
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_cross_stage_side_send_profile_c32_20260831
```

最终两侧 raw profile 均由采集时相同的 CANN 9.0.0 离线解析，`with_stack=false`，已生成
`kernel_details.csv`、`step_trace_time.csv`、`trace_view.json`、communication 文件和数据库。
Attention 与 FFN active window 仍未完全对齐，因此 FFN 的同 role layer 配对可用于证明
目标重叠，但不能把双侧 aggregate 直接相减或升级为正式性能结论。

## 7. 特性四：MTP

### 7.1 特性原理

MTP 使用一个 draft layer 根据 target hidden 提议下一个 token，再由 target 模型验证和接受/拒绝。AFD 下 target decoder 和 MTP draft 中的 MoE 都归 FFN role，因此一个在线 step 在普通 43 层 AF 往返之后，还要增加一次独立的 MTP AF phase。

```text
target decoder（U1 或 U2）
  -> 合并 target hidden
  -> Attention proposer 准备 draft 输入
  -> MTP header + post-HC hidden 发往 FFN
  -> FFN 执行唯一 MTP MoE
  -> MTP output 返回 Attention
  -> proposer / verify / rejection sampler
```

MTP 不是简单删除 `speculative_config` 门禁：必须重新定义 draft 权重归属、target hidden 生命周期、动态 header、Graph cache 以及与 Microbatch 的组合顺序。

### 7.2 端到端流程

1. 启动时按 `mtp.*` checkpoint key 将 Attention/FFN 权重分别加载到对应角色。
2. target decoder 正常执行 AF U1/U2；U2 的两个 stage 完成后合并 target hidden。
3. Attention 生成包含 speculative step、local token count 和全局 FFN token count 的 header。
4. FFN 先校验所有 peer header，再接收 post-HC hidden，执行一个 MTP layer 的远端 MoE。
5. Attention 接收 output，完成剩余 draft 计算与 target verify/rejection sampling。
6. target Graph + draft eager 时，两阶段分别执行；full-draft Graph 时使用独立 target/draft Graph cache。
7. step 结束或异常时清理 MTP layout、header buffer 和 target hidden，禁止旧状态跨 step 复用。

### 7.3 适配演进

| 阶段 | 主要修改 | 状态 |
|---|---|---|
| M0 | 冻结原生 MTP key、权重、target hidden、acceptance 和 HBM 契约 | 原生基线完成 |
| M1 | 角色化 MTP loader；eager/U1 的 header、hidden、output phase | 功能基线完成 |
| M2 | target Graph/U1 + draft eager；target/draft 分阶段 | 功能基线完成 |
| M3 | eager/U2 按请求边界切分；全 DP U1/U2 决策；一次 MTP phase | 功能完成，性能缺口保留 |
| M4 | target Graph/U2 + draft eager；重复 key 双侧 replay；在线 miss 整步 fallback | 功能基线完成 |
| M5 | MTP header 和 peer layout 泛化到 `A = k x F` | 组件闭环 |
| M6 | 非等量 target Graph + eager draft MTP | 组件闭环 |
| M7 | target 和 draft 均使用 ACL Graph | 功能基线完成 |

### 7.4 为支持该特性需要的适配

- MTP checkpoint 的 Attention/FFN key 必须和普通 decoder 分开处理，否则会漏载或双载参数。
- header 先冻结 step、local token count 和全局 FFN count，FFN 才能安全接收变长 hidden。
- target U2 后只执行一次 draft，保持上游 proposer/verify 语义。
- Graph capture 不能解析动态 header；M7 在 Graph 外准备稳定 header buffer，图内只执行固定通信。
- 所有未验证组合由 feature validator 拒绝，避免在 HCCL recv 深处挂死。

### 7.5 修改点与代码对应

| 适配点 | 代码位置 | 具体修改 | 作用 |
|---|---|---|---|
| MTP 权重归属 | `afd_plugin/model_executor/models/deepseek_v4.py::_mtp_checkpoint_weight_roles`、`_iter_mtp_role_weights` | 按 `mtp.<layer>.ffn` 与非 FFN key 分配角色 | 保证 draft 参数与普通 decoder 一样只加载一次 |
| 角色化 draft layer | `deepseek_v4.py::AFDDeepSeekMultiTokenPredictorLayer` | Attention 构造 draft 非 MoE 路径，FFN 构造 draft MoE | 把 MTP 计算沿同一远端 MoE 边界拆开 |
| target hidden 生命周期 | `deepseek_v4.py::get_mtp_target_hidden_states`、Attention runner 的 MTP buffer 管理 | 收集/合并 target hidden 并在 step 后清理 | 为 proposer 提供当前 step 的正确输入 |
| phase metadata | `afd_plugin/connectors/metadata.py::AFDTransferMetadata` | 增加 `phase='mtp'`、layer/stage/speculative step | 区分 decoder 与 MTP 消息 |
| MTP header 发送 | `p2p_hccl.py::send_mtp_header`、`prepare_mtp_header_for_graph` | 发送或预写 token layout/header | 让 FFN 在接收变长 hidden 前获得确定 shape |
| MTP header 聚合 | `p2p_hccl.py::recv_mtp_header`、`recv_mtp_header_for_graph` | 校验多 peer header 并构造 `peer_slices` | 支持等量/非等量与 eager/Graph |
| FFN draft phase | `ffn_model_runner.py::_execute_mtp_after_target`、`_mtp_ffn_forward` | target 完成后只执行一次远端 draft MoE | 保持上游 proposer/verify 契约 |
| MTP 控制握手 | `p2p_hccl.py::P2pHcclAFDControlPlane.send_mtp_phase_ready`、`recv_mtp_phase_ready` | 同步两侧进入 draft eager/replay 的决策 | 防止 target 后 A/F 执行模式分叉 |
| draft Graph key/cache | `cuda_graph.py::make_mtp_ffn_graph_key`、`ffn_model_runner.py::_capture_mtp_graphs` | 为 draft 建立独立 layout key 和 Graph cache | 支持 full-draft Graph 且不复用 target 图 |
| U2 请求边界 | `ubatch_utils.py::create_request_boundary_ubatch_slices` | 保持同一请求的 target/draft token 不被拆散 | 修复机械 token 切分导致的 golden 偏差 |
| MTP 能力门禁 | `feature_validation.py::_fail_if_unsupported_deepseek_v4_features` | 限定 method、layer 数、speculative token 数和组合 | 将未验证范围变为启动期错误 |

#### 7.5.1 关键代码串讲

修改点一：MTP 权重不能复用普通 decoder 的 key 规则。`_mtp_checkpoint_weight_roles` 明确 `mtp.<layer>.ffn.*` 属于 FFN，其余合法 MTP key 属于 Attention：

```python
def _mtp_checkpoint_weight_roles(name: str) -> frozenset[str]:
    normalized = name.removeprefix("model.")
    parts = normalized.split(".")
    if len(parts) < 3 or parts[0] != "mtp" or not parts[1].isdigit():
        return _NO_ROLE
    return _FFN_ROLE if parts[2] == "ffn" else _ATTENTION_ROLE

def _iter_mtp_role_weights(weights, *, role):
    for name, loaded_weight in weights:
        if role in _mtp_checkpoint_weight_roles(name):
            yield name, loaded_weight
```

修改点二：Attention 在发送变长 hidden 之前先冻结 header。header 同时携带本地 token 数和每个 FFN rank 的聚合 token 数：

```python
counts = self._mtp_ffn_token_counts(num_tokens_across_dp)
counts = counts.reshape(-1).to(
    device=buffer.device,
    dtype=torch.int32,
)
buffer[0] = _MTP_HEADER_MAGIC
buffer[1] = speculative_step
buffer[2] = num_tokens
buffer[3] = self.ffn_size
buffer[_MTP_HEADER_PREFIX_SIZE:].copy_(counts, non_blocking=False)
self._send_tensor(
    buffer,
    dst=self.mapping.subgroup_index,
    group=group,
)
```

修改点三：FFN 先逐 peer 接收并交叉校验 header，再构造聚合 layout；peer 对 speculative step 或 FFN count 的理解不一致时立即失败：

```python
for source_rank in peer_ranks:
    dist.recv(buffer, src=source_rank, group=group)
    values = [int(value) for value in buffer.cpu().tolist()]
    counts_tuple = tuple(values[_MTP_HEADER_PREFIX_SIZE:])
    if speculative_step is None:
        speculative_step = values[1]
        expected_counts = counts_tuple
    elif values[1] != speculative_step or counts_tuple != expected_counts:
        raise RuntimeError(
            "DSV4 MTP peer headers disagree on speculative step or FFN token counts"
        )
    seq_lens.append(values[2])
```

修改点四：target decoder 可以是 U1 或 U2，但 draft 始终合并为一次 stage 0 远端 MoE，避免错误地对两个 target stage 各执行一次 draft：

```python
if stage_ids not in ([0], [0, 1]):
    raise RuntimeError(
        "DSV4 AFD MTP requires target decoder stages [0] or [0, 1]"
    )

stage_idx = 0
header = self.connector.recv_mtp_header(stage_idx=stage_idx)
payload = self.connector.recv_attn_output(
    ubatch_idx=stage_idx,
    layer_idx=0,
    phase="mtp",
    speculative_step=header.speculative_step,
    num_tokens=header.num_tokens,
)
```

MTP header 由 `p2p_hccl.py::send_mtp_header/recv_mtp_header` 管理，逻辑格式为：

```text
[magic, speculative_step, local_num_tokens,
 ffn_size, ffn_count_0, ..., ffn_count_n]
-> post-HC hidden [T, H]
-> remote MTP MoE output [T, H]
```

当前实现显式限制：

```python
if metadata.layer_idx != 0 or metadata.speculative_step != 0:
    raise RuntimeError(
        "DSV4 HCCL P2P MTP supports only layer 0/speculative step 0"
    )
```

### 7.6 修改意义

- 保留上游 proposer/verify/rejection sampler，只把 MoE 与权重所有权扩展到 AF，减少语义偏移。
- 将 MTP 建模为独立 phase，使 eager/Graph、U1/U2 和非等量可以分别组合、分别验收。
- header/layout 协议使 FFN 在没有 scheduler 请求对象的情况下仍能正确执行 draft MoE。
- 独立 Graph cache 和控制握手解决 target/draft 状态串用与 A/F 模式分叉问题。

### 7.7 验证结果与支持边界

| 阶段 | 核心结果 | 性能口径 |
|---|---|---|
| M0 | 30/30 与 MTP-off 一致，accepted/drafted `198/264`，75.0% | 原生协议基线 |
| M1 | 30/30，batch 1/8/32，五次冷启动，30 分钟空闲恢复 | P1 `28.280 token/s`，guard |
| M2 | 30/30，target Graph + eager draft | P1 `22.835 token/s`，较 M1 回退 19.253% |
| M3 | 30/30，batch 32 真实 U2 | P1 `16.238 token/s`，较 M1 回退 42.583% |
| M4 | 30/30，Graph/U2，P1 128/128，acceptance 84.51% | `31.473 token/s`，guard |
| M7 | U1/U2 各 30/30，full-draft Graph | `27.510 token/s`，guard |

支持范围固定为 1 个 MTP layer、`method=mtp`、`num_speculative_tokens=1`。更多 speculative token 继续 fail-fast。

详细证据见 M0、M1、M2、M3、M4 和 full-draft Graph 六份专项报告。

## 8. 特性五：AF 非等量拓扑

### 8.1 特性原理

等量 A8F8 是一个 Attention rank 对一个 FFN rank。非等量拓扑支持 `A = k x F`：多个 Attention rank 共享一个 FFN rank。FFN 按固定顺序聚合多个 peer 的 hidden，执行一次合并 MoE，再按原 slice 返回各自 output。

```text
A0 ----\                         /---- A0 output
A1 -----\-> F0: concat -> MoE --+----- A1 output
A2 -----\                        +----- A2 output
A3 ------\                       \---- A3 output
```

### 8.2 端到端流程

1. 根据 A/F 数量计算整数 `ratio`，为每个 worker 构造确定性 subgroup 和 peer 列表。
2. 每个 Attention peer 发送自己的 input IDs/hidden 和本地 token count。
3. FFN 按 Attention role rank 顺序接收，生成 `peer_slices` 并拼接连续 input。
4. FFN 对聚合 tensor 执行一次 MoE。
5. FFN 按同一 `peer_slices` 拆分 output，逐 peer 返回。
6. MTP 同时校验各 peer header 的 speculative step 和 FFN count；Graph key 保存每个 peer 的精确 shape。

### 8.3 拓扑规则

确定性映射为：

```text
ratio = A / F
Attention rank a -> FFN rank floor(a / ratio)
FFN rank f -> Attention ranks [f * ratio, (f + 1) * ratio)
```

- 构建每个 FFN subgroup 的多 peer process group 和 `peer_slices`。
- FFN 按 Attention role rank 顺序接收并聚合，发送时按同一 slice 拆分。
- MTP 要求同 subgroup 的 header 对 speculative step 和 FFN count 完全一致。
- Graph key 保存展开后的每个 Attention peer token count。
- 继续拒绝 `A < F` 和非整数比例。

### 8.4 为支持该特性需要的适配

- 拓扑适配：从一对一 rank mapping 泛化为 subgroup mapping，并区分 global rank、role rank、subgroup index。
- 数据面适配：FFN 多 peer recv、聚合 buffer、slice 状态和多 peer send。
- metadata 适配：每个 peer token count 与 FFN 聚合 count 都必须进入协议。
- MTP 适配：多个 header 必须对全局 layout 和 speculative step 达成一致。
- Graph 适配：Graph key 不能只保存聚合总数，必须保留精确 peer layout。
- 资源门禁：A8F4 实模在加载前检查每 FFN rank 的专家权重和 HBM 峰值。

### 8.5 修改点与代码对应

| 适配点 | 代码位置 | 具体修改 | 作用 |
|---|---|---|---|
| rank 解析 | `afd_plugin/distributed/topology.py::resolve_role_rank` | 从 vLLM rank 解析 AFD role rank | 建立统一 rank 坐标系 |
| subgroup mapping | `topology.py::AFDRankMapping`、`build_rank_mapping` | 计算 ratio、peer ranks、subgroup index | 将 `A = k x F` 固化为确定性映射 |
| peer token layout | `p2p_hccl.py::_peer_token_counts_for_stage`、`_attention_peer_world_ranks` | 获取当前 FFN 对应的所有 Attention peer 和长度 | 为聚合 recv 准备 shape |
| slice 构造 | `p2p_hccl.py::_make_peer_slices`、`HCCLP2PTransferState.peer_slices` | 保存 `(peer, start, end)` | 保证聚合后能按原 peer 返回 |
| 多 peer A2F | `P2pHcclAFDConnector.recv_attn_output`、`recv_attn_output_streamed` | 按 role rank 顺序 recv 并写入聚合 buffer | FFN runner 仍看到连续 tensor |
| 多 peer F2A | `P2pHcclAFDConnector.send_ffn_output`、`send_ffn_output_streamed` | 按 `peer_slices` 拆分并发送 | 将计算结果正确路由回各 Attention peer |
| 非等量 MTP | `recv_mtp_header`、`_validate_mtp_header_values`、`_mtp_ffn_token_counts` | 聚合并交叉校验多个 peer header | 扩展 draft phase 到 `A = k x F` |
| 非等量 Graph key | `cuda_graph.py::make_ffn_graph_key`、`make_mtp_ffn_graph_key` | 保存每个 Attention peer 的 token layout | 防止相同总数的不同布局复用错误图 |
| 非法拓扑门禁 | `feature_validation.py::_fail_if_unsupported_deepseek_v4_features` 与 connector extra-info 校验 | 拒绝 `A < F`、非整数 ratio 和未验证 TP 组合 | 启动期暴露配置错误 |

#### 8.5.1 关键代码串讲

修改点一：`build_rank_mapping` 把 `A = k x F` 固化为确定性 subgroup。每个 subgroup 的 rank 0 是 FFN，后面是它负责的连续 Attention ranks：

```python
ratio = attention_size // ffn_size
ffn_ranks = list(range(ffn_size))
attention_ranks = list(range(ffn_size, ffn_size + attention_size))
subgroup_ranks = tuple(
    [ffn_ranks[subgroup_index]]
    + [
        attention_ranks[subgroup_index * ratio + offset]
        for offset in range(ratio)
    ],
)
rank_in_subgroup = subgroup_ranks.index(world_rank)
```

修改点二：FFN 根据控制面的 per-peer token count 构造连续聚合 buffer 及可逆的 `peer_slices`：

```python
first_attention_rank = self.mapping.subgroup_index * self.ratio
seq_lens = tuple(
    max(1, int(attention_counts[first_attention_rank + offset]))
    for offset in range(self.ratio)
)
peer_ranks = self._attention_peer_world_ranks()
layout = HCCLP2PStageLayout(
    peer_ranks=peer_ranks,
    seq_lens=seq_lens,
    peer_slices=_make_peer_slices(peer_ranks, seq_lens),
    num_tokens=sum(seq_lens),
)
```

修改点三：A2F 按 slice 把多个 peer 写入一个连续 tensor；MoE 完成后，F2A 使用同一组 slice 原路拆回：

```python
# A2F: multi-peer -> aggregate buffer
for source_rank, start, end in layout.peer_slices:
    self._recv_tensor(
        hidden_states[start:end],
        src=source_rank,
        group=group,
        stream=stream,
    )

# F2A: aggregate output -> original peers
for destination_rank, start, end in peer_slices:
    self._send_tensor(
        ffn_output[start:end],
        dst=destination_rank,
        group=group,
        stream=stream,
    )
```

修改点四：非法数量关系在 process group 初始化前拒绝，通信层只处理 `A >= F` 且整除的拓扑：

```python
if attention_size < ffn_size:
    raise ValueError("num_attention_ranks must be >= num_ffn_ranks")
if attention_size % ffn_size != 0:
    raise ValueError("num_attention_ranks must be a multiple of num_ffn_ranks")
```

### 8.6 修改意义

该协议把拓扑扩展限制在 connector 和 metadata 层，模型 runner 仍看到一个连续 tensor；因此模型计算、MTP 和 Graph 可以复用等量路径。确定性 rank 顺序也使 A/F 两侧 HCCL op 序列可验证、可 capture。

### 8.7 验证结果与支持边界

- eager 普通/MTP：A1F1、A2F1、A4F2 的两 stage、两 step、不同 token count 和 close 通过。
- Graph：A2F1/A4F2 两 stage capture/replay；A4F2 target Graph + eager MTP 组件通过。
- 这些结果是 component functional snapshot。
- A3 64 GiB 上 A8F4 的 FFN EP4 模型构造 HBM 不足，实模 golden、生命周期和性能转到 A5；不得写成 A8F4 产品级支持。

详细证据：

- [非等量 MTP 组件报告](DEEPSEEK_V4_AFD_HCCL_P2P_MTP_UNEQUAL_COMPONENT_REPORT_ZH.md)
- [非等量 Graph 组件报告](DEEPSEEK_V4_AFD_HCCL_P2P_GRAPH_UNEQUAL_COMPONENT_REPORT_ZH.md)

## 9. 特性六：TP2

### 9.1 特性原理

TP2 将一个模型角色的权重和计算拆到两个 tensor-parallel worker。AFD 的 Attention/FFN 数量配置统计 role worker，而调度 token count 是 DP 级，因此 TP2 下必须同时维护 DP rank、TP rank、role rank 和 AF peer rank。

```text
Attention DP0: A-role rank 0/1 (TP0/TP1)  <->  FFN DP0: F-role rank 0/1
Attention DP1: A-role rank 2/3 (TP0/TP1)  <->  FFN DP1: F-role rank 2/3
...
```

同一 DP 内的两个 TP worker 处理同一批 token，所以 DP 级 token count 要复制到对应的两个 role rank；但每个 TP worker 仍使用自己的 HCCL peer 和模型 shard。

### 9.2 端到端流程

1. 启动期校验 `role_ranks == DP x TP`，并解析每个进程的 DP/TP/role rank。
2. Attention 控制面发送 DP metadata 和 `tensor_parallel_size`。
3. FFN 校验两侧 TP 一致，把 DP 级 token count 展开到 role-rank layout。
4. 每个 Attention TP worker 与匹配的 FFN TP worker 交换 IDs/hidden/output。
5. Graph key 和 MTP token count 使用展开后的 role-rank layout。
6. TP2 原生 golden 与 AFD golden 使用同一 DP4/TP2 目标栈独立生成和对照。

### 9.3 为支持该特性需要的适配

- rank 适配：明确 DP、TP、role rank 和 AF peer 的转换关系。
- control payload 适配：携带 `tensor_parallel_size` 并在 FFN 侧校验。
- token layout 适配：将 DP count 复制到同 DP 的 TP workers。
- connector 适配：按 role rank 建立 peer group，不把 DP rank 直接当通信 rank。
- Graph/MTP 适配：key、header 和 count 均使用 TP 展开后的 layout。
- feature gate：第一版只开放等量 A8F8、DP4/TP2、eager/U1。

### 9.4 修改点与代码对应

| 适配点 | 代码位置 | 具体修改 | 作用 |
|---|---|---|---|
| role rank 解析 | `afd_plugin/distributed/topology.py::resolve_role_rank`、`build_rank_mapping` | 将全局/本地 rank 映射为 DP/TP 下的 AFD role rank | 为 peer 选择提供统一坐标 |
| 启动期 TP 门禁 | `feature_validation.py::_fail_if_unsupported_deepseek_v4_features` | 校验 TP 只能为 1/2、TP2 connector、等量 A/F、`role_ranks == DP x TP` | 防止错误拓扑进入通信初始化 |
| 控制 payload | `afd_plugin/connectors/metadata.py::AFDControlPayload` | 增加并序列化 `tensor_parallel_size` | 让 FFN 校验 Attention 侧并行契约 |
| 控制面校验 | `p2p_hccl.py::P2pHcclAFDControlPlane.update_state_from_dp_metadata` | 对比 payload 与本地 TP 值 | 避免两侧 layout 解释不同 |
| DP count 转 role count | `ffn_model_runner.py::_to_dp_level_token_counts`、`_ffn_token_counts_across_ranks` | 规范 DP metadata 并生成 FFN 所需 count | 为 FFN stage shape 提供正确输入 |
| Graph layout 展开 | `cuda_graph.py::_expand_attention_values_tuple` | 每个 DP token count 重复到所属 TP workers | Graph key 与真实 HCCL peer shape 一致 |
| MTP count 展开 | `p2p_hccl.py::_mtp_ffn_token_counts` | 在 TP2 下把 DP count 展开到 Attention role ranks | 保持 MTP header/hidden shape 一致 |
| recipe/验证 | `recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_attention.sh`、`afd_ffn.sh`、`run_validation.py` | 增加 DP4/TP2 设备、端口、golden 和 F0 配置 | 可复现启动与验收 TP2 |

#### 9.4.1 关键代码串讲

修改点一：role rank 不再直接等于 DP rank，而是由 DP、PCP 和 TP 三个局部坐标线性化。当前 PCP 被 feature gate 限制为 1，但公式为后续扩展保留了维度：

```python
dp_rank = int(parallel_config.data_parallel_rank) if dp_size > 1 else 0
pcp_rank = int(get_pcp_group().rank_in_group) if pcp_size > 1 else 0
tp_rank = int(get_tensor_model_parallel_rank()) if tp_size > 1 else 0
role_rank = (dp_rank * pcp_size + pcp_rank) * tp_size + tp_rank
```

修改点二：TP 大小进入 Attention -> FFN 控制 payload，FFN 在准备任何数据 buffer 前校验双方一致：

```python
@dataclass(slots=True)
class AFDControlPayload:
    dp_metadata_list: dict[int, AFDDPMetadata]
    is_graph_capturing: bool
    is_warmup: bool
    tensor_parallel_size: int = 1

if int(payload.tensor_parallel_size) != connector.tensor_parallel_size:
    raise RuntimeError(
        "DeepSeek-V4 AFD requires matching Attention/FFN tensor parallel sizes"
    )
```

修改点三：Graph key 将一个 DP token count 复制到该 DP 的所有 TP workers，使 key 的长度与真实 Attention role ranks 一致：

```python
if len(values) < attention_size and attention_size % len(values) == 0:
    tp_size = attention_size // len(values)
    expanded = tuple(values[i // tp_size] for i in range(attention_size))
```

修改点四：首版 TP2 的支持范围由启动门禁明确限定为 P2P HCCL、等量 A/F 和 `role_ranks == DP x TP`：

```python
if tensor_parallel_size == 2:
    if afd_config.connector != "P2pHcclAFDConnector":
        raise RuntimeError("DeepSeek-V4 AFD TP2 supports only P2pHcclAFDConnector")
    if afd_config.num_attention_ranks != afd_config.num_ffn_ranks:
        raise RuntimeError("DeepSeek-V4 AFD TP2 currently requires equal ranks")
    expected_role_ranks = (
        int(parallel_config.data_parallel_size) * tensor_parallel_size
    )
    if role_ranks != expected_role_ranks:
        raise RuntimeError("DeepSeek-V4 AFD TP2 requires role ranks to equal DP x TP")
```

### 9.5 修改意义

TP2 证明 AF 拓扑不再隐含“一个 DP rank 等于一个 worker”。这套 rank/metadata 契约同时服务于 Graph key、MTP header 和后续 Mooncake PD 的 Decode DP/TP 配置。

### 9.6 验证结果与支持边界

- 原生 DP4/TP2 golden 稳定。
- A8F8 DP4/TP2 eager/U1 实模 F0 达到 30/30，生命周期和清理通过。
- A2F1/A4F2/等量 TP2 组件用于验证 role-rank 和 peer 映射。
- 功能 tag：`dsv4-afd-v023-hccl-tp2-v1`。
- TP2 full-draft Graph U2 最大组合出现 FFN AICore 异常，当前显式 fail-fast；CAMP2P TP2、非等量 TP2 和 TP3 也不支持。

详细证据见 [TP2 验证报告](DEEPSEEK_V4_AFD_HCCL_P2P_TP2_VALIDATION_REPORT_ZH.md)。

## 10. 特性七：PD 分离

### 10.1 特性原理

PD 分离把 Prefill 与 Decode 部署为独立服务：Prefill 只负责 prompt 阶段并把 KV cache 传给 Decode；Decode 接管后续 token 生成。与 AF 组合后，Decode 本身又拆为 Attention 和 FFN 两个角色。

```text
请求 -> Proxy -> Prefill
                 |
                 | Mooncake KV transfer
                 v
             Decode Attention <---- HCCL hidden/output ----> Decode FFN
                 |
                 v
            后续 decode tokens
```

两条数据面必须解耦：Mooncake 只连接 Prefill 与 Decode Attention；HCCL P2P 只连接 Decode Attention 与 Decode FFN。FFN 不持有 KV cache，也不配置 KV connector。

### 10.2 端到端流程

1. Prefill、Decode 和 Proxy 分别读取角色配置，校验源码 commit、CANN/ATB、venv、NIC、端口和 Mooncake 动态库。
2. Prefill 以 `kv_producer` 启动；Decode Attention 以 `kv_consumer` 启动；Decode FFN 只启动 AFD HCCL backend。
3. Decode 节点紧邻启动 FFN 和 Attention，等待 Attention HTTP 与全部 FFN worker loop 就绪。
4. Proxy 接收请求并路由 Prefill；Mooncake 将 KV 传给 Decode Attention。
5. Decode Attention 在生成阶段通过既有 AF HCCL 协议调用 Decode FFN。
6. 验证经 Proxy 执行 smoke、golden、batch 和取消恢复，并在 Decode 日志核对真实 KV remote session。
7. 按 Proxy、Decode、Prefill 逆序停止，三个角色分别收集小型验收包。

### 10.3 为支持该特性需要的适配

- KV connector 适配：固定 `MooncakeHybridConnector` 的 producer/consumer 和 DP/TP metadata。
- EngineCore 适配：Decode Attention 保留 KV cache，FFN 使用 no-op KV/scheduler 路径。
- 配置适配：结构化生成 `engine_id`、KV port、Prefill/Decode DP/TP 与 host mapping。
- recipe 适配：Prefill、Decode Attention、Decode FFN、Proxy 四类启动命令和顺序。
- 运行时适配：Mooncake wheel/existing 安装、jemalloc preload、CANN/ATB/动态库一致性检查。
- 验证适配：本地两进程 NPU round-trip、双机 F0、取消恢复、KV 日志和角色清理。
- 组合门禁：首版只接受 TP1 eager/U1、MTP off；其他组合后续独立解除。

### 10.4 修改点与代码对应

| 适配点 | 代码位置 | 具体修改 | 作用 |
|---|---|---|---|
| PD 能力门禁 | `afd_plugin/compat/npu/feature_validation.py::_fail_if_unsupported_deepseek_v4_pd` | 校验 Attention-only KV、Mooncake connector/role、TP1/U1/eager、端口和拓扑 | 固定首个可验收 M9 边界 |
| Mooncake 拓扑校验 | `feature_validation.py::_validate_mooncake_parallel_config` | 解析 Prefill/Decode DP/TP/PP 并与 Attention 配置比对 | 防止 KV rank metadata 与 Decode 实际拓扑不一致 |
| 结构化配置 | `tools/dsv4/mooncake_pd_config.py::build_mooncake_pd_config` | 生成 vLLM `kv_transfer_config` | 避免 shell 拼接 JSON 造成字段漂移 |
| FFN EngineCore 隔离 | `engine_core.py::_initialize_ffn_engine_core`、`_AFDFFNKVCacheConfig`、`_AFDFFNNoopScheduler` | FFN 不创建 KV connector 和普通 scheduler | 保持 FFN 仅作为内部 MoE 后端 |
| Prefill recipe | `recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_pd/prefill.sh` | 启动 Mooncake producer | 提供 Prefill 服务和 KV 生产端 |
| Decode recipe | `recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_attention.sh`、`afd_ffn.sh` | Attention 启用 consumer，FFN 保持无 KV | 组合 PD 与已有 AF 数据面 |
| Proxy recipe | `recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_pd/proxy.sh` | 注册 Prefill/Decode backend | 提供统一请求入口 |
| Mooncake runtime 门禁 | `tools/dsv4/check_mooncake_runtime.sh` | 检查 import、动态库、CANN 泄漏和 jemalloc | 在模型加载前发现运行栈问题 |
| NPU 传输组件 | `tools/dsv4/check_mooncake_npu_roundtrip.py::_producer`、`_consumer` | 两进程注册 buffer 并执行两轮真实 NPU 传输 | 独立证明 Mooncake Ascend 数据面 |
| 双机角色管理 | `tools/dsv4/mooncake_pd_manual/pd.sh::install_action/check_action/start_action/status_action/validate_action/stop_action/collect_action` | 固定安装、启停、验收和证据收集流程 | 降低双机部署差异并生成可回传产物 |

#### 10.4.1 关键代码串讲

修改点一：`build_mooncake_pd_config` 生成结构化 KV 契约。Prefill/Decode 共用 `engine_id` 和 KV port，并显式携带两端 DP/TP metadata：

```python
return {
    "kv_connector": "MooncakeHybridConnector",
    "kv_role": role,
    "kv_port": kv_port,
    "engine_id": engine_id,
    "kv_parallel_size": 1,
    "kv_connector_extra_config": {
        "prefill": {
            "dp_size": prefill_dp_size,
            "tp_size": prefill_tp_size,
        },
        "decode": {
            "dp_size": decode_dp_size,
            "tp_size": decode_tp_size,
        },
    },
}
```

修改点二：KV connector 只允许挂在 Decode Attention，FFN 不会误建 Mooncake session：

```python
if afd_config.role != "attention":
    raise RuntimeError(
        "DeepSeek-V4 AFD Mooncake PD attaches KV transfer only to Attention"
    )
```

修改点三：M9 首版组合门禁在启动期固定为 eager/U1、MTP off、TP1；这些条件后续完成 F0 后逐项解除：

```python
if not bool(vllm_config.model_config.enforce_eager):
    raise RuntimeError("Mooncake PD M9 baseline supports only eager execution")
if bool(parallel_config.use_ubatching):
    raise RuntimeError("Mooncake PD M9 baseline supports only U1")
if vllm_config.speculative_config is not None:
    raise RuntimeError("Mooncake PD M9 baseline does not support MTP")
if int(parallel_config.tensor_parallel_size) != 1:
    raise RuntimeError("Mooncake PD M9 baseline supports only TP1")
```

修改点四：FFN EngineCore 使用空 KV cache 配置和 no-op scheduler，从对象构造层保证 FFN 不进入普通 KV/scheduler 路径：

```python
class _AFDFFNKVCacheConfig:
    kv_cache_groups: list[Any] = []

class _AFDFFNNoopScheduler:
    connector = None

    def get_kv_connector(self) -> None:
        return None

    def has_requests(self) -> bool:
        return False
```

### 10.5 修改意义

- 把生产链路拆成可独立诊断的 KV 通道与 AF hidden 通道，故障归因更清楚。
- FFN 不参与 KV transfer，避免多余 KV cache、错误 connector 初始化和角色混淆。
- 双机管理工具把环境、启动顺序、健康检查、取消恢复和证据收集固化为可重复流程。
- PD 首版采用最小 TP1 eager/U1 边界，先验证正确性和生命周期，再逐项组合 Graph/U2/MTP/TP2。

### 10.6 当前验证结果

阶段汇总：`/mnt/workspace/validation/dsv4_afd_v023_mooncake_pd_m9_contract_20260821_181148/summary.json`。

| 门禁 | 结果 |
|---|---|
| Mooncake runtime/metadata contract | passed |
| plugin Mooncake feature gate | 13/13 passed |
| recipe/config/EngineCore | 71/71 passed |
| vLLM-Ascend `MooncakeHybridConnector` | 5/5 passed |
| 通用 Mooncake connector | 92/92 passed |
| 两进程真实 NPU round-trip | 2/2，2 MiB 逐字节一致 |
| NPU cleanup | passed |
| M9 实模 F0 | pending |

汇总状态是 `real_transfer_component_passed_f0_pending`，不是 M9 功能完成证明。

### 10.7 当前进行中和支持边界

截止 2026-08-25，未提交改动集中在手工安装与部署健壮性：支持已有安装或 wheel 两种模式、CANN 9.0.0/9.0.1 与外置 ATB、动态库指纹、离线镜像的网络/端口探测回退、目标 venv 路径优先级和支持包收集。这些改动仍属于进行中，不能写成冻结能力。

待完成：

- TP1 eager/U1 双机 PD + A8F8 实模 golden、batch、取消恢复、生命周期 F0。
- Decode 日志中的真实 KV remote session 证据。
- TP1 通过后扩展 DP4/TP2。
- 再分别组合 U2、Graph 和 MTP；当前全部 fail-fast。

## 11. 验证体系和性能结论

### 11.1 分层验证

```text
CPU/Mock 单测
  -> 真实 NPU connector/component round-trip
  -> F0：smoke + 30/30 serial golden + batch 1/8/32
  -> 生命周期：冷启动、二次启动、空闲恢复、停止、fatal、cleanup
  -> P1：单配置单轮灾难性回退 guard
  -> P2：固定矩阵三轮、稳定性、公平资源对照、profile
  -> 功能 tag / 性能 tag
```

F0 是功能门禁；P1 只判断是否出现明显回退；只有 P2 才能支持正式性能结论。

### 11.2 正确性口径

- golden 必须来自相同 vLLM/vLLM-Ascend/CANN/模型配置的非 AFD 服务。
- 串行 10 条 prompt x 3 轮做逐 token exact；并发 batch 主要验证结构、长度、错误和 stage 行为。
- U2 必须从所有 Attention rank 日志中观察真实双 stage，不能只检查配置值。
- Graph 必须同时看到 capture 和 replay，且 A/F 两侧 HCCL 顺序一致。
- 服务停止后检查 fatal marker、return code 和 NPU process table。

### 11.3 Profile 口径

- 历史目标栈 profile 使用 CANN 9.0.1 采集并由同版本解析；2026-08-29 Graph/U2 多流、
  严格闭环及 2026-08-31 P8F profile 使用 CANN 9.0.0 采集并由 9.0.0 解析。不同 CANN
  的 raw profile 不交叉
  解析或直接比较。
- 性能 profile 固定 `TORCH_PROFILER_WITH_STACK=0`，避免 Python stack 事件改变 eager 调度。
- `Free` 表示既无计算也无非重叠通信的设备区间，不等同于 `aclrtFree`。
- Attention 和 FFN 必须使用同一采集窗口对照，分别报告 Computing、Communication、Free 和 Bubble。

### 11.4 `P8D-PERF-001`

当前正式性能问题仍为 Open：

| 对比 | 结果 | 结论 |
|---|---:|---|
| v0.23 eager U1 vs U2 三轮 | 17.082 vs 12.582 token/s，U2 -26.342% | 功能可用，U2 无收益 |
| async scheduling off 的 eager U1 vs U2 | 30.615 vs 16.631 token/s，U2 -45.676% | host 调度改善不等于 U2 改善 |
| P8D 相对 P8C | +10.099% | layer-major 有改善 |
| P8D 相对 U1 | -46.197% | 仍未关闭性能缺口 |
| MTP M3 U2 相对 M1 U1 | -42.583% | MTP 组合再次确认缺口 |
| Graph/U2 P1 | 107.189 token/s | 单轮且执行模式改变，只是候选信号 |
| Graph/U2 多流 C32 profile guard | 151.655 token/s；设备时间线出现预期 43 层重叠 | 单轮且含 profiler 开销，只证明功能和重叠存在 |
| Graph/U1 C32 三轮 | 134.871 token/s，CV 5.705% | 稳定参照点 |
| 优化前 Graph/U2 C32 三轮 | 125.251 token/s，CV 10.628%，相对 U1 -7.133% | 功能全过但稳定性门禁失败 |
| 优化后 Graph/U2 C32 三轮 | 139.300 token/s，CV 5.635%；相对 pre-U2 +11.217%，相对 U1 +3.284% | 优化后点稳定；pre-U2 波动使净收益不能冻结 |
| 严格闭环 Graph/U2 C32 profile guard | 119.810 token/s、p50 TPOT 189.514 ms；相对第一版多流 profile 吞吐 -20.998% | 860/860 层依赖正确，但端到端性能候选失败 |
| 严格闭环 Attention 稳定窗口 | non-overlap communication -75.870%，wall -19.310%，Bubble +104.288% | 局部 Attention 改善不能替代全服务吞吐门禁 |
| P8F 同 layer 跨 stage C32 profile guard | 140.116 token/s、p50 TPOT 159.936 ms；相对 P8E +16.948%，相对第一版跨 layer 多流 -7.609% | FFN `compute U1` / `recv U2` 重叠已恢复，Attention 无跨 layer 对角；仍是单轮 profiler guard |
| P8F 无 profiler C16/C32/C64 三轮 | 均值 117.824 / 126.270 / 40.403 token/s；CV 13.051% / 5.491% / 1.581% | 功能与生命周期门禁通过；整体性能门禁失败，C64 相对 C32 -68.003% |
| 混合 DAG C32 三轮 | off 126.272 token/s、CV 9.611%；on 134.309 token/s、CV 4.845%；on/off `+6.365%` | 两组稳定性与功能门禁通过；3 轮 C32 候选收益，尚非跨负载 baseline |
| 混合 DAG Attention profile | non-overlap communication `-50.118%`，overlap ratio `43.939% -> 75.646%`，stage `-18.958%` | 机制与吞吐方向一致；FFN step 窗口错位，不能直接做 20-step aggregate |

双侧 profile 已证明目标 stream/event DAG 确实产生重叠，三轮对照也证明优化后 U2 点通过稳定性门槛。但 pre-U2 对照波动超限，优化后相对稳定 U1 仅 `+3.284%`，因此不能宣称 `+11.217%` 是可发布净收益。后续正式关闭条件是：稳定复测 pre-U2、同预算 native/AFD、MTP on/off、延迟、HBM、`tokens/s/NPU` 和必要的双侧同窗口 profile 全部完成。

P8E 进一步表明：若把 `recv0 -> send1` 解释成 recv0 **完成**后才允许 send1，就会取消
FFN 原有的跨 stage overlap；再把 FFN `recv1` 放到 `send0` 完成前会形成完成态环依赖。
P8F 不要求这个互斥的完成顺序，而是保持 Attention parent 的
`send0, post recv0, send1, post recv1` issue 顺序，并让 FFN compute/send 使用 event 连接的
side stream。最终 FFN 70/86 层已出现同 layer U1 执行与 U2 recv 重叠，同时 Attention
下一层 compute 必须等待当前层双 recv 后记录的 `ready` event，不存在跨 layer 对角流水。
该版本收回了 P8E 大部分性能损失，但单轮结果仍不能关闭 `P8D-PERF-001`。

当前混合 DAG 候选在 P8F 之上将 Attention 下一层计算改为等待同 stage 的 `recv_done`，
有意恢复受数据依赖约束的跨 layer 对角重叠。A8F8 单轮功能 smoke、CANN 9.0.0 C32
on/off 三轮和双侧 profile 已完成；三轮吞吐候选提升 `6.365%`，Attention 时间线显示
non-overlap communication 降低 `50.118%`。该结果可以证明调度机制生效并支持继续推进，
但仍缺完整 F0、更多/交错轮次、native 同预算、MTP 与多并发点，不能关闭
`P8D-PERF-001` 或创建性能 tag。

详细证据见 [A3-P8 同步 HCCL 性能报告](DEEPSEEK_V4_AFD_A3_P8_SYNC_HCCL_PERFORMANCE_REPORT_ZH.md)。

## 12. 已知限制和后续更新清单

### 12.1 功能待办

- [ ] M9 TP1 eager/U1 双机 PD + A8F8 实模 F0。
- [ ] M9 DP4/TP2 PD 扩展。
- [ ] PD + U2、Graph、MTP 的独立组合门禁。
- [ ] A5 上完成 A8F4 非等量实模 golden、生命周期与性能。
- [ ] SP/CP/DCP 和 PP。
- [ ] 多 MTP layer 或多个 speculative token。
- [ ] TP2 full-draft Graph U2 最大组合重新定位并通过后解除 fail-fast。

### 12.2 性能待办

- [ ] 关闭 `P8D-PERF-001`。
- [x] 执行 Graph/U1、优化前 U2、优化后 U2 同提交 C32 三轮公平对照。
- [x] 验证严格闭环 860/860 层 `send0 -> recv0 -> send1 -> recv1`，并记录单轮性能回退。
- [x] 验证 P8F 同 layer `FFN compute U1` / `recv U2` 重叠，并移除 Attention 跨 layer 对角流水。
- [ ] 交错执行或增加轮数，使 pre-U2 CV 不超过 10% 后复核多流净收益。
- [x] P8F 同 layer 流水完成无 profiler C16/C32/C64 各三轮；C16 CV 超限且 C64 绝对性能退化，不能关闭性能问题。
- [ ] 第一版多流和 P8E 严格闭环执行同源码无 profiler C16/C32/C64 各三轮，补齐控制组。
- [ ] P8F 执行 C64 单波次对照，并在第二波补位窗口采集 Attention/FFN 同窗口 profile。
- [x] 混合 DAG 完成 A8F8 单轮 Graph capture/replay、10/10 serial golden、batch32、
  双 stage、生命周期和 NPU 清理 smoke。
- [x] 混合 DAG 完成同源码 `--graph-u2-hybrid-dag on/off` C32 三轮对照和 Attention/FFN
  DP0 双侧 profile；两组 CV 通过，Attention 时间线验证目标 overlap。
- [ ] 混合 DAG 继续完成最小 Graph component、A1F1 连续 capture/replay 和 A8F8 完整 F0。
- [ ] 在最小 HCCL Graph component 中验证独立 send stream + compute event；连续
  capture/replay 100 次无 pending work 后再接入 connector。
- [ ] 若 side-stream HCCL 仍不能稳定 capture，将 U2 拆为两个 compute Graph segment，
  HCCL 留在 Graph 外调度。
- [ ] MTP on/off 三轮公平对照。
- [ ] 同预算 native Graph 对照和 `tokens/s/NPU`。
- [ ] A/F 双侧同窗口 profile，报告 compute、non-overlap communication、Free、Bubble 和等待分布。
- [ ] A5 独立建立运行栈、golden、profile 和性能数字，不复用 A3 二进制或结论。

### 12.3 当前明确不支持

- U3。
- Attention-side gate。
- `A < F` 或 A/F 非整数比例。
- CAMP2P TP2、非等量 TP2、TP3。
- TP2 full-draft Graph U2 最大组合。
- Mooncake PD 下的 Graph、U2、MTP 和 TP2，直到各自 F0 完成。

### 12.4 报告更新规则

后续每完成一个阶段，只在具备以下证据后更新状态：

1. 固定三个源码 commit、运行栈、模型、拓扑和参数。
2. 保存原始 `runtime.json`、golden/batch、日志门禁和 cleanup 结果。
3. 功能组合通过 F0 后才能从“进行中”改为“已冻结功能基线”。
4. 组件测试不能升级为实模 E2E 结论。
5. 单轮 P1 不能升级为性能收益；正式性能结论必须满足 P2。
6. 新支持组合必须同步删除对应 fail-fast、补单测/E2E、更新 recipe 和本文支持矩阵。

## 13. 关键提交和证据索引

### 13.1 关键提交

| 提交 | 特性 |
|---|---|
| `99ee0ef6` | vLLM 0.23 + `rfc/vllm_cann` 迁移 |
| `3b0fd32b` | HCCL Graph/U1 |
| `afc5d289` | 原生 MTP M0 契约 |
| `8f2e7c80` | MTP eager/U1 |
| `f0385c78` | target Graph/U1 + eager draft MTP |
| `1d8db03f` | U2 P8D、同步性能定位 |
| `7f098d58` | HCCL Graph/U2 |
| `7ba3baca` | MTP eager/U2 |
| `694d9262` | target Graph/U2 + eager draft MTP |
| `4aafb101` | 非等量 MTP |
| `2860d09a` | 非等量 Graph |
| `a4699d5b` | full-draft MTP ACL Graph |
| `436eed65` | TP2 |
| `e7c6da77` | Mooncake PD 首个实现 |
| `49bb4a1d` | HCCL P2P recipe 隔离 |

### 13.2 冻结 tag

当前 v0.23/plugin 关键功能 tag 包括：

```text
dsv4-afd-v023-vllm-cann-eager-u2-functional-v1
dsv4-afd-v023-hccl-graph-u1-v1
dsv4-afd-v023-native-mtp-m0-v1
dsv4-afd-v023-hccl-mtp-m1-v1
dsv4-afd-v023-hccl-mtp-m2-v1
dsv4-afd-v023-hccl-u2-p8d-functional-v1
dsv4-afd-v023-hccl-graph-u2-v1
dsv4-afd-v023-hccl-mtp-u2-v1
dsv4-afd-v023-hccl-mtp-graph-u2-v1
dsv4-afd-v023-hccl-mtp-unequal-component-v1
dsv4-afd-v023-hccl-graph-unequal-component-v1
dsv4-afd-v023-hccl-mtp-full-draft-graph-v1
dsv4-afd-v023-hccl-tp2-v1
```

M9 目前没有功能 tag；A3 目标栈也没有可发布的性能 tag。

### 13.3 进一步阅读

- [A3 性能、非等量与 A5 路线](DEEPSEEK_V4_AFD_A3_PERFORMANCE_A5_PORTING_ROADMAP_ZH.md)
- [DeepSeek-V4 AFD HCCL P2P 安装部署指南](DEEPSEEK_V4_AFD_HCCL_P2P_INSTALL_DEPLOYMENT_GUIDE_ZH.md)
- [MTP M0 原生基线](DEEPSEEK_V4_AFD_MTP_M0_NATIVE_BASELINE_REPORT_ZH.md)
- [MTP M1](DEEPSEEK_V4_AFD_HCCL_P2P_MTP_M1_VALIDATION_REPORT_ZH.md)
- [MTP M2](DEEPSEEK_V4_AFD_HCCL_P2P_MTP_M2_VALIDATION_REPORT_ZH.md)
- [MTP M3](DEEPSEEK_V4_AFD_HCCL_P2P_MTP_M3_VALIDATION_REPORT_ZH.md)
- [MTP M4](DEEPSEEK_V4_AFD_HCCL_P2P_MTP_M4_VALIDATION_REPORT_ZH.md)

## 14. 一句话结论

v0.23/plugin 形态已经完成 DeepSeek-V4 AFD 的 HCCL P2P、eager/Graph U1/U2、MTP M0-M7、非等量组件和 TP2 功能主线；当前真正未闭环的是 Mooncake PD 双机实模、复杂并行与多 speculative token，以及能够证明 AFD 收益的正式性能验收，因此现阶段应表述为“功能组合大部分完成，生产 PD 与性能基线仍在推进”。
