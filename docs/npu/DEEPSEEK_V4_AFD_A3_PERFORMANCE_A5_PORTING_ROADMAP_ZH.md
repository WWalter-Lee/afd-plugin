# DeepSeek-V4 AFD A3 性能、非等量拓扑与 A5 适配路线

## 1. 文档定位

本文用于固化 DeepSeek-V4 AFD 在完成 eager/U1、Graph/U1 以及标准 HCCL P2P eager/U1、U2 正确性基线后的目标、开发顺序和验收门禁，供后续开发、验证、性能分析和 A5 迁移时直接使用。

文档状态：`2026-08-17`。CAMP2P eager/U2 已冻结为 `dsv4-afd-a3-eager-u2-v1`；标准 HCCL send/recv connector 已在提交 `9578dd2cb70f9f8db54673a70e8f45fde6479245` 完成 A3 A8F8 eager/U1、U2 正确性闭环。A3-P4 的 A8F8 未调优性能参照与 U1/U2 双侧 profile 已完成：三轮重复性通过，但 U2 在 C32 比 U1 回退 37.570%，因此当前只冻结参照协议，不冻结 U2 性能基线。

A3-P5 的 `A = k x F` 非等量协议和 A2F1/A4F2 NPU 组件验证已经完成。A3-P6 的 A8F4 实模加载在 64 GiB A3 上因 FFN EP4 专家权重峰值 HBM 不足而停止；A10F5 容量代理又被固定 vLLM-Ascend 的 256 experts/EP5 非均匀放置检查拒绝。该结论是当前硬件与固定栈组合的 E2E 门禁，不否定 connector 的非等量语义。A8F4 E2E 移到高 HBM 的 A5 实机验证，A3 当前继续以 A8F8 做同步 HCCL 性能优化和公平对照。

当前性能阶段明确不引入异步 HCCL：不使用 `isend/irecv`、后台通信线程或自定义异步传输 op，保留标准阻塞式 `torch.distributed.send/recv` 和现有 NPU 同步边界。第一轮同步优化已经减少重复 host 解析、device-to-host 标量读取和每层 forward-context 构造；A8F8/U1/C32 正式三轮均值达到 57.724 output token/s，相对 P4 均值提升 17.521%，CV 为 0.689%。该结果是 C32 候选收益，不代表 C1/C8、非 AFD 公平对照或整个 P7 已冻结。

本文不替代以下文档：

- `DEEPSEEK_V4_AFD_ADAPTATION_GUIDE_ZH.md`：完整适配背景和早期里程碑；
- `DEEPSEEK_V4_AFD_BASELINE_TAGS_SUMMARY_ZH.md`：已冻结 tag 的关键改动、原因和意义；
- `DEEPSEEK_V4_AFD_HCCL_P2P_VALIDATION_REPORT_ZH.md`：新 connector 的实现边界和 A8F8 正确性证据。

本文回答后续最关键的五个问题：

1. 当前只有 A3 环境时，哪些工作可以继续完成；
2. 如何证明开启 AFD 后有真实性能收益；
3. 为最终支持 A5，现在的代码需要保持哪些可迁移边界；
4. 标准 HCCL connector 如何支持 Attention/FFN 数量不相等；
5. A5 到位后，还必须完成哪些硬件相关适配和重新验收。

## 2. 最终目标和阶段性结论

### 2.1 最终目标

在昇腾服务器上为 DeepSeek-V4 建立可部署、可复现、可回退的 Attention/FFN 分离能力，并在相同模型、相同请求和可解释的资源口径下证明 AFD 的性能收益；最终在目标 A5 服务器上完成独立的正确性、稳定性和性能验收。

“能够运行”不是最终完成条件。正式验收必须同时满足：

- 正确性：输出 token IDs 与同平台非 AFD golden 一致；
- 生命周期：冷启动、二次启动、空闲恢复、异常退出和正常停止无残留；
- 性能：吞吐收益超过运行波动，尾延迟不出现不可接受的回退；
- 资源效率：同时报告总吞吐和 `tokens/s/NPU`，不能只用更多 NPU 与单实例比较；
- 可复现性：固定源码、运行栈、参数、拓扑、数据集和 profiling 解析版本。

### 2.2 当前阶段结论

当前可以且应该继续在 A3 上开发，先完成 AFD 的通用语义和 A3 性能闭环，再到 A5 上开发硬件差异部分。

```text
A3 当前阶段
  CAMP2P U1/U2 correctness 已完成并冻结
  -> 标准 HCCL P2P connector U1/U2 correctness 已完成
  -> 锁定 A8F8 U1/U2 性能参照和请求矩阵
  -> HCCL P2P A=kF 非等量 fan-in/fan-out 组件闭环（已完成）
  -> A8F4 实模容量预检（A3 EP4 HBM 不足，转 A5）
  -> A8F8 阻塞式 HCCL profiling、同步调度优化和公平性能验收
  -> 冻结 A3 性能基线

A5 硬件到位后
  平台审计和独立运行栈
  -> 标准 HCCL send/recv 组件验证
  -> A=F 与 A=kF 的 U1/U2 eager 回归
  -> 重新选择 A/F 比例并完成独立性能验收
```

A3 验收通过只说明实现语义和 A3 性能成立，不等于 A5 已支持，也不能将 A3 性能数字直接外推到 A5。

## 3. 已冻结基线

### 3.1 Tag 和定位

| Tag | Commit | 定位 |
|---|---|---|
| `dsv4-afd-eager-u1-v1` | `40981475a9270c9b79ebf5cfe46d375472ee0a06` | A8F8、eager、U1 正确性基线 |
| `dsv4-afd-graph-u1-v1` | `2ed98442351d4be96edbb315a6b6c8d00805bbc4` | A8F8、`FULL_DECODE_ONLY`、U1 Graph 与生命周期基线 |
| `dsv4-afd-a3-eager-u2-v1` | `1b5d011c830d66a2516ed647064fa571667761a3` | A8F8、eager、U2 正确性、生命周期与提交态 smoke 基线 |

以上 tag 均不改写。HCCL 后续阶段以提交 `9578dd2` 为开发起点，每个阶段独立提交并通过全部门禁后再打新 tag，失败阶段不打 tag。

标准 HCCL P2P connector 当前位于分支 `feat/dsv4-afd-hccl-p2p`、提交 `9578dd2cb70f9f8db54673a70e8f45fde6479245`。它是后续开发起点，但在性能门禁通过前不创建性能 tag。

### 3.2 A3 固定运行栈

| 项目 | 固定值 |
|---|---|
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| Python venv | `/mnt/workspace/code/.venvs/afd-v026` |
| vLLM | `568afb3a13806beb53bb2e6bd518269357b237c0` |
| vLLM-Ascend | `80d8c194f7584b17fe08065ea99a130916f6b0e7` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| 插件 | `ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd` |
| A8F8 参照拓扑 | Attention NPU 0-7，FFN NPU 8-15 |
| A8F4 候选拓扑 | Attention NPU 0-7，FFN NPU 8-11；NPU 12-15 不计入服务资源 |
| 并行 | A8F8 为 Attention DP8、FFN DP8/EP8；A8F4 为 Attention DP8、FFN DP4/EP4；TP1、PP1、CP1、DCP1 |
| 确定性 | seed 1024、temperature 0 |

A3 后续开发继续遵守：

- 只修改 `afd-plugin`；
- 不修改固定 vLLM 和 vLLM-Ascend 源码；
- 不混入默认 CANN 9.1.0 或 vLLM 0.22.1 工作树；
- 每次验证前运行 `tools/dsv4/check_runtime.sh`；
- 每次验证后保存清理完成后的 `npu-smi info`。

### 3.3 当前 profiling 观察基线

现有 Graph/U1 profile：

`/mnt/workspace/validation/dsv4_afd_graph_u1_dp0_profile_ceaf4f1_20260811_205830/profile_validation.json`

| Role | 平均计算 | 平均未重叠通信 | 平均 free | 平均 stage |
|---|---:|---:|---:|---:|
| Attention DP0 | 233.670 ms | 0.226 ms | 2.852 ms | 236.749 ms |
| FFN DP0 | 80.871 ms | 2.120 ms | 5.051 ms | 85.922 ms |

FFN 的 `free` 最大值为 50.261 ms，属于需要在后续稳定采集中确认的离群点。

这些数据只用于定位优化方向和与后续 trace 对比，不构成“AFD 已有性能收益”的证据。现有采集没有完成非 AFD 同口径对照，也没有完成 U2 重叠收益验证。

## 4. 当前架构边界

### 4.1 Connector 决策

DeepSeek-V4 现在有两条明确分开的 NPU 数据通路：

- `CAMP2pAFDConnector`：已有 A3 基线，hidden 路径使用 afd-plugin 自定义 A2E/E2A 算子；
- `P2pHcclAFDConnector`：当前性能主线，hidden、FFN output 和 DSV4 input IDs 全部使用 `torch.distributed.send/recv` 的 HCCL process group。

新增 connector 的原因不是模型语义不同，而是通信实现和调度约束不同。标准 HCCL 路径不加载、不调用 `torch.ops.vllm.afd_camp2p_send_attn_output()` 或 afd-plugin A2E/E2A 自定义算子，因而更贴近 A5 目标接口，也可以独立衡量 HCCL send/recv 下的 AFD 收益。

`P2pHcclAFDConnector` 的关键边界为：

- `AFDA2FTransferPayload.input_ids`；
- 每个 stage 独立的 `afd_ids` HCCL group；
- 每个 stage 独立的 hidden/output HCCL group；
- 预分配的 NPU `int32` IDs buffer；
- Attention 到 FFN 的一次性 IDs side channel；
- FFN hash layer 的 stage 级 IDs cache 生命周期；
- Gloo 控制面先传 DP token count，FFN 再按精确 shape 投递 HCCL receive；
- 阻塞式 send/recv 的 DBO 调度在 FFN output receive 后切换 stage，和 FFN 的 layer-major 顺序保持一致，避免两个 stage 交叉阻塞。

后续性能开发以 `P2pHcclAFDConnector` 为主线。CAMP2P 保留为已冻结回归基线，不把两条数据面实现在同一个 connector 内用条件分支混合。

### 4.2 A/F 非等量支持范围

当前标准 HCCL P2P DSV4 适配已经支持以下拓扑契约；等量门禁只继续保留在 CAMP2P 路径：

非等量首版支持范围明确限定为：

```text
A >= F
A % F == 0
ratio = A / F
```

即一个 FFN rank 对应连续的 `ratio` 个 Attention rank。A=F 继续作为 ratio=1 的兼容路径。A2F1/A4F2 已完成真实 NPU 组件闭环；A8F4 仍是高 HBM A5 的首个非等量 E2E 目标。

确定性映射为：

```text
Attention rank a -> FFN rank floor(a / ratio)
FFN rank f       -> Attention ranks [f * ratio, (f + 1) * ratio)
AFD world        -> [F0 ... F(F-1), A0 ... A(A-1)]
```

每个 FFN rank 必须按 Attention role rank 升序接收每个 peer 的 IDs 和 hidden，将它们按同一顺序拼接；FFN 计算后再按原始 `seq_lens` 切分 output 并发回对应 Attention。IDs、hidden 和 output 必须共享同一 peer 顺序，不能分别推导。

这个范围是 DSV4 HCCL P2P 实现的阶段性产品契约，不是 HCCL 或 A3/A5 的底层限制。首版继续显式拒绝：

- `A < F`：需要把一个 Attention rank 的 token scatter 到多个 FFN，再按原顺序 gather output；
- `A % F != 0`：需要非均匀 peer group、负载分配和更复杂的退出协议；
- Graph 下的非等量拓扑：需要稳定 buffer 地址和固定消息图，放在 eager 性能闭环之后。

因此文档中的“支持 A/F 非等量”均特指 eager 下的 `A = k x F`、`k >= 2`，不得扩展解读为任意 A/F 组合。

### 4.3 当前未验证能力

标准 HCCL P2P eager/U1、U2 已完成当前正确性门禁，但还没有形成性能 tag。以下能力仍不属于 HCCL 主线基线：

- Graph/`FULL_DECODE_ONLY`；
- Attention 侧 gate；
- MTP/speculative decoding；
- Mooncake PD；
- sequence parallel；
- A/F 非等量实模 E2E（connector 和 A2F1/A4F2 组件已通过，A3 A8F4 受 HBM 阻塞）；
- TP、PP、CP 或 DCP 大于 1；
- A5 实机 HCCL P2P 验证与调优。

## 5. A3 后续开发阶段

每阶段独立提交、独立验证。上一阶段未通过时，不进入下一阶段。

| 阶段 | 主要交付 | 进入下一阶段的门禁 |
|---|---|---|
| A3-P0 | 固定非 AFD、AFD eager/U1、AFD Graph/U1 的性能实验协议 | 三种部署使用同一模型、请求和统计口径；结果可重复 |
| A3-P1 | DSV4 eager/U2 的 stage IDs 与执行语义 | 已通过：相关回归 116 项，真实双 stage 执行通过 |
| A3-P2 | eager/U2 A8F8 E2E | 已通过：golden、batch、双冷启动、30 分钟空闲恢复、profile 和清理通过 |
| A3-P3 | 新增标准 HCCL P2P connector | 已通过：A1F1/U2 组件 round-trip，A8F8 U1/U2 各 30/30 golden，batch 1/8/32、退出和清理通过 |
| A3-P4 | 锁定 A8F8 性能协议和未调优参照 | 已完成：U1/U2 三轮 CV 均低于 10%，双侧 profile 已由 CANN 9.0.1 解析；U2 C32 回退 37.570%，保留为调优对象 |
| A3-P5 | HCCL P2P `A = k x F` 非等量实现 | 已通过：A2F1/A4F2 真实 NPU 组件、两 stage/两 step、不同 peer token count、聚合/切分和 close 均通过 |
| A3-P6 | A8F4 eager/U1、U2 E2E 正确性 | A3 停止：EP4 模型构造 HBM 不足；A10F5 被固定栈 EP5 专家放置拒绝；A8F4 E2E 转 A5 |
| A3-P7 | A8F8 同步 HCCL profiling、调优和公平性能验收 | 进行中：C32 三轮均值 57.724 token/s、CV 0.689%，较 P4 +17.521%；待补 C1/C8、冷服务重复与非 AFD 公平对照 |
| A3-P8 | 冻结 A3 HCCL 性能基线 | tag、报告、原始日志、trace、配置与清理证据齐全 |

### 5.1 A3-P0：固定性能实验协议

先锁定实验协议，再开始调优或改变拓扑。至少固定：

- prompt 集合和输入/输出长度；短请求、常规长上下文与 128K 能力点分开统计；
- batch/concurrency 阶梯，至少保留 batch 1/8/32；
- 请求到达方式和预热请求数；
- seed、temperature 和最大输出 token 数；
- HBM 利用率、最大序列数和最大 batched tokens；
- U2 threshold、HCCL buffer 和调度参数；
- 每个点至少 3 轮稳定运行；
- 冷启动数据与稳态数据分开报告；
- 服务端吞吐、客户端延迟和 NPU 资源指标使用相同测量窗口。

必须保留以下三类初始对照：

1. 非 AFD eager/Graph 基线；
2. AFD eager/U1；
3. AFD Graph/U1。

后续只逐项加入 HCCL P2P eager/U1 和 U2，避免一次改变多个变量。

建议至少冻结以下长度类型，具体 token 数写入实验 manifest，不在跑数后调整：

| 类型 | 建议用途 | 约束 |
|---|---|---|
| 短输入 + 128/512 输出 token | decode 吞吐、TPOT 和 U2 稳态收益 | 输出必须足够长，避免只测启动和 TTFT |
| 8K/32K 输入 + 128 输出 token | 常规长上下文 TTFT、HBM 与吞吐 | 各 concurrency 独立记录 |
| 128K 输入 + 32/128 输出 token | 最大上下文能力、稳定性和 HBM 边界 | 先做 batch 1；不能作为唯一性能代表点 |

128K 只有在模型 `max_model_len`、KV cache 容量和固定运行栈共同允许时才进入正式矩阵；若因容量无法运行，应记录最大可稳定长度和失败原因，不能缩短输入后仍标记为 128K。

### 5.2 A3-P1：实现 eager/U2

建议分支：

```text
feat/dsv4-afd-eager-u2
```

本阶段已经完成。DSV4 只在 eager 下解除 U2 门禁，Graph/U2 继续显式拒绝；Attention 按 stage 的 token slice 发送 IDs，FFN 在单主线程中按 stage 预接收并限定 cache 生命周期。

本阶段需要完成：

关键交付包括：

1. 仅对已验证配置解除 DSV4 U2 门禁，继续拒绝 ubatch 数不等于 2；
2. 使用 `ubatch_slices[*].token_slice` 对 `input_ids` 做与 hidden states 完全相同的切分；
3. stage 0 和 stage 1 分别向自己的 `afd_ids` group 发送一次 IDs；
4. 保证每个 step/stage 的 IDs 消息数和 hidden 消息数严格对应；
5. FFN layer 0 分 stage 接收，layer 1/2 引用对应 cache，layer 3 起传 `None`；
6. step 完成、异常、取消和 shutdown 都在 `finally` 中清空 cache；
7. 请求不足以触发 U2 时保留 U1 fallback；
8. U1 原有路径和两个已冻结 tag 的行为不得回退。

实现时不硬编码 NPU 0-15 或 A8F8。角色数、rank、stage 数和 token slice 继续来自配置或运行上下文，为 A5 不同卡数保留空间。

### 5.3 A3-P1 测试门禁

CPU/Mock 测试至少覆盖：

- stage 0/1 使用刻意不同的 IDs 和 token 数；
- 每 step/stage 只发送一次 IDs；
- layer 0 接收，layer 1/2 复用，layer 3 后不可见；
- 连续两个 step 使用不同 IDs，无旧缓存污染；
- `-1` padding、词表上下界、空 tensor 和超 buffer token 数；
- send/recv 消息顺序和数量不匹配时显式失败；
- 异常、取消、connector close 后 cache/buffer 状态可重新使用；
- 未触发 U2 时 U1 fallback 行为不变；
- DSV4 以外模型和 connector 的配置验证不回退。

connector 组件测试至少覆盖：

- A1F1 的 stage 0/1 `int32` IDs round-trip；
- 两个 stage 使用不同 token count；
- 连续两个 step；
- hidden 和 IDs 的顺序一致；
- 异常取消与 connector close；
- 测试完成后 HCCL group 和进程均无残留。

### 5.4 A3-P2：eager/U2 E2E

部署继续使用 A8F8、DP8/TP1/EP8 和固定插件列表，关闭 MTP、Graph 和 PD。

硬门禁：

- 复用 Milestone 0 的 10 条 golden prompt；
- 连续 3 轮，共 30/30 请求逐 token 一致；
- batch 1/8/32 请求结构和 token 结果正确；
- 两轮冷启动均通过；
- 空闲 30 分钟后恢复；
- U1 fallback 和实际 U2 都被请求覆盖；
- Attention 先停，FFN 后退出；
- 两侧进程返回码为 0；
- fatal marker 为空；
- 端口、共享内存、HCCL 进程和 NPU 占用清理完成。

通过后建议冻结：

```text
dsv4-afd-a3-eager-u2-v1
```

### 5.5 A3-P3：标准 HCCL P2P 等量基线

本阶段已经在提交 `9578dd2cb70f9f8db54673a70e8f45fde6479245` 完成。验证产物为：

```text
/mnt/workspace/validation/dsv4_afd_hccl_p2p_component_fix_20260813
/mnt/workspace/validation/dsv4_afd_hccl_p2p_u1_correctness_20260813
/mnt/workspace/validation/dsv4_afd_hccl_p2p_u2_correctness_20260813
```

A8F8 eager/U1、U2 均达到 30/30 golden，batch 1/8/32、真实双 stage、退出和 NPU 清理通过。该提交是非等量开发的回归基准，不是性能收益结论。

### 5.6 A3-P4：锁定 A8F8 性能参照

本阶段已经完成。完整报告见
`DEEPSEEK_V4_AFD_HCCL_P2P_A3_P4_PERFORMANCE_REPORT_ZH.md`。

未调优参照的主要结论为：U1/U2 在 concurrency 1/8/32 的三轮 output
throughput CV 均低于 10%；U2 在 C1/C8 的差值没有超过波动，在 C32 从 U1
的 49.118 output tokens/s 降到 30.664 output tokens/s，回退 37.570%。双侧
20-step profile 表明 FFN U2 computing 没有退化，但 free 从 20.864 ms 墑到
304.625 ms，当前阻塞式 HCCL send/recv 与共享 compute stream 没有形成有效
stage 重叠。

这一阶段不改变 connector 拓扑语义，使用提交 `9578dd2` 固定性能实验 manifest，并取得未针对结果调参的 A8F8 U1/U2 参照。目的有两个：确认 benchmark 波动范围；为后续判断 A8F4 的收益、回退和新增通信开销提供同 connector 对照。

需要完成：

1. 以同一请求矩阵分别运行 A8F8 eager/U1、U2，每点至少 3 轮；
2. benchmark 稳态窗口与 profiler 窗口分开，不能把 profiler 开销计入正式吞吐；
3. 保存 output tokens/s、TTFT、TPOT、HBM、利用率和启动参数；
4. 采集 Attention role-local DP0 与 FFN role-local DP0，确认 trace 可由 CANN 9.0.1 解析；
5. 记录 3 轮波动区间；若波动超过预先写入 manifest 的阈值，先处理负载、日志、温度或 host 抖动，不进入拓扑对比；
6. 该阶段只产生“参照报告”，不因 U2 暂未获益而阻塞 A8F4 实现。

A3-P4 的参照重复性门禁固定为每个 concurrency 的 output throughput
CV 不超过 10%。这个阈值只判断三轮数据是否可作为未调优参照；后续声称
U2、A8F4 或其他优化有收益时，收益仍必须大于对应三轮的实际波动区间，
不能只因为通过 10% CV 门禁就判定性能提升。

### 5.7 A3-P5：实现 `A = k x F` 非等量 HCCL P2P

实现从 A2F1、A4F2 组件拓扑开始，再进入 A8F4。不能只删除等量门禁，必须作为一个完整的 rank、消息和 buffer 协议修改。

#### 5.7.1 Rank 与控制面

- 将公共 P2P topology 校验语义明确为 `A >= F` 且 `A % F == 0`，错误信息不再绑定 NCCL connector 名称；
- HCCL connector 复用 `AFDRankMapping.subgroup_index`、`subgroup_ranks` 和 `ratio`，但保持数据面只使用标准 `torch.distributed.send/recv`；
- Attention `a` 的数据目标为 `floor(a / ratio)`，不能继续使用 `role_rank` 直接作为 FFN rank；
- 每个 FFN 只接收一个完整 DP metadata payload；由确定的 Attention metadata sender 发送，其他 Attention rank 不发送控制消息；
- 每个 subgroup 的第一个 Attention peer（role rank `f * ratio`）作为 FFN rank `f` 的 metadata sender，FFN 只从这个固定 source 接收；
- group 创建顺序在所有 rank 上完全一致；U1/U2 仍各 stage 独立 data group 和 IDs group；
- 非法比例、rank 越界、组初始化失败和 connector 重建必须 fail-fast 并清理已创建 group。

#### 5.7.2 数据面与 IDs side channel

每个 FFN rank 对其 Attention peers 按 role rank 升序执行固定协议：

```text
layer 0, each peer:
  recv input IDs
  recv hidden

layer 1+, each peer:
  recv hidden

after FFN compute, each peer:
  send matching output slice
```

- 控制面的 per-Attention token count 生成 `seq_lens`，同时决定每个 peer 的 receive slice；
- FFN 预分配每 stage 的聚合 IDs/hidden buffer，直接向不重叠 slice 接收，避免每层创建 peer tensor 后再 `cat`；
- FFN 的聚合 buffer 容量按本 rank 所有 peer 的 token 上限校验；A8F4 recipe 必须相应设置 FFN `max_num_batched_tokens`，容量不足时在接收前失败，不能截断或临时超配；
- 聚合 IDs 与聚合 hidden 使用完全相同的 peer 顺序和 `seq_lens`；layer 0-2 的 FFN IDs cache 保存聚合视图，layer 3 起为 `None`；
- FFN output 按 `seq_lens` 切分并发回原 Attention peer，Attention 只从其映射 FFN 接收；
- `HCCLP2PTransferState` 保存 stage、peer ranks 和 `seq_lens`，send 端不重新猜测切分；
- U2 继续在 Attention 收到匹配 output 后 yield，并增加多 Attention peer 发送时序偏斜测试，证明不会因阻塞 send/recv 交叉等待；
- step 成功、异常、取消和 close 都清空 stage cache；peer 丢失必须在配置的 HCCL 超时内失败，不能无限等待。

#### 5.7.3 实现测试门禁

CPU/Mock 至少覆盖：

- A1F1 兼容不回退，A2F1、A4F2 映射正确；
- A<F、A%F!=0、零/负 role 数与越界 rank 明确拒绝；
- peer token count 刻意不同，IDs/hidden 聚合顺序和 output split 精确一致；
- layer 0 IDs 每 peer 每 stage 仅一条消息，layer 1/2 复用，layer 3 后为空；
- U1、U2、连续 step、不同 token count、`-1` padding、词表边界和 buffer 上限；
- 任一 peer 发送、接收或 FFN compute 异常后无旧 cache，close 可重入；
- HCCL connector 源码仍不引用 CAMP2P A2E/E2A 自定义 op。

NPU 组件至少覆盖 A2F1 和 A4F2：

- BF16 hidden/output 与 int32 IDs round-trip；
- U1 和两个 stage，连续两个不同 step；
- 每个 Attention peer 使用不同 token count 和可识别 tensor 内容；
- 人为延迟一个 Attention peer，验证固定接收顺序不互锁；
- 正常 close、异常取消、二次创建和 `npu-smi` 清理。

只有以上门禁通过后，才删除 DSV4、配置层和 connector 构造层的等量 fail-fast；仍保留 A<F 和非整数比例的 fail-fast。

#### 5.7.4 配置与部署脚本

- HCCL recipe 增加独立的 Attention/FFN role 数和 device list，启动前验证数量、重复 device 和越界 device；
- 两侧 `additional_config.afd` 必须写入相同的 `num_attention_ranks`、`num_ffn_ranks` 和 connector，只有 `role` 不同；
- Attention 服务的 DP size 等于 A，FFN 服务的 DP/EP size 等于 F；
- FFN `max_num_batched_tokens` 和 receive buffer 上限必须覆盖一个 subgroup 的聚合 token 数；
- readiness、PID、日志和清理循环分别按 A/F 数量生成，不再假定两侧都是 8 rank；
- manifest 保存逻辑 rank 到物理 NPU 的完整映射，并明确记录未参与服务的 NPU；
- 默认 A8F8 recipe 行为保持不变，A8F4 使用单独配置或显式参数，不能静默改变已有基线。

#### 5.7.5 完成状态与证据

本阶段实现和组件门禁已完成：

- 公共 topology 与 HCCL connector 接受 `A >= F`、`A % F == 0`，CAMP2P 仍保持 A=F；
- FFN 按 subgroup 聚合 IDs/hidden、保留同一 `peer_slices`，并按原 peer 顺序切分 output；
- 每个 subgroup 只由第一个 Attention peer 发送控制 metadata；
- role device list、DP/EP 和 FFN 聚合容量已参数化，默认 A8F8 行为不变；
- CPU/Mock 回归及 A2F1、A4F2 真实 HCCL round-trip 通过。

主要 NPU 证据：

```text
/mnt/workspace/validation/dsv4_afd_a3_p5_hccl_a2f1_20260817_1055/summary.json
/mnt/workspace/validation/dsv4_afd_a3_p5_hccl_a4f2_20260817_1105/summary.json
```

### 5.8 A3-P6：A8F4 eager/U1、U2 E2E

A8F4 的计划拓扑为 Attention DP8、FFN DP4/EP4、TP1、PP1、CP1、DCP1。进入 E2E 前先完成两侧 model load 和峰值 HBM 预检：FFN rank 数减少后，每 rank 的专家权重、量化 scale/offset 和运行 workspace 会增加；若模型加载或安全 HBM 余量不通过，不得靠提高超卖比例强行进入请求测试。

正确性和生命周期门禁与 A8F8 相同，并增加：

- 参数所有权、每 FFN rank 专家分片和 peak HBM 报告；
- 10 条 golden prompt x 3 轮，30/30 逐 token 与同平台非 AFD/M0 golden 一致；
- batch 1/8/32 覆盖不同 Attention peer token count；
- U1 fallback 和真实 U2 均有 runtime evidence；
- 至少两轮冷启动、30 分钟空闲恢复和二次启动；
- Attention 全部先停、FFN 后退出；单 peer 异常时在超时内退出且无 HCCL/NPU 残留；
- A8F8 回归用同一提交重跑，确认 ratio=1 路径未回退。

A8F2 不是本阶段硬门禁。只有 A8F4 正确性通过且 FFN HBM/profile 表明仍有余量时，才以独立实验评估 A8F2。

#### 5.8.1 A3 实机预检结论

2026-08-17 的 A3 预检没有进入 golden 请求阶段：

- A8F4 eager/U1 在 FFN 模型构造时 OOM；EP4 下每个 FFN rank 持有 64 个专家，约 60.62 GiB 已激活且不足以再分配 514 MiB。降低运行时 token buffer 或 `gpu_memory_utilization` 不能解决权重构造容量；
- A10F5 用作 2:1 且更低单 rank 专家数的容量代理时，固定 vLLM-Ascend 因 256 experts 无法均匀分配到 EP5 而 fail-fast：`allocated=52, placement=51`；
- 不使用冗余专家/EPLB 绕过，因为这会改变专家放置、内存和性能语义，超出当前 topology-only 验证范围。

证据目录：

```text
/mnt/workspace/validation/dsv4_afd_a3_p6_hccl_a8f4_u1_smoke_20260817_1120
/mnt/workspace/validation/dsv4_afd_a3_p6_hccl_a10f5_u1_smoke_20260817_1125
```

因此 A3-P6 的结论是“受硬件/固定栈容量阻塞”，不是 connector 正确性失败。A3 不再尝试用超卖或改变专家放置强行完成 A8F4；该 E2E 门禁保留给高 HBM A5。

### 5.9 A3-P7：等量/非等量联合 profiling 和调优

当前 A3 实际执行范围收敛为 A8F8；A8F4 联合 profile 等 A5 完成实模加载后再恢复。第一阶段仅做同步 HCCL 优化：

- 保留阻塞式 `torch.distributed.send/recv`；
- 保留 connector 当前的 `torch.npu.synchronize()` 完成语义；
- 不使用 `isend/irecv`、额外通信 stream、后台通信线程或自定义异步 op；
- 每个 step/stage 只解析一次 token counts 和 peer slices，43 层复用同一 stage runtime；
- NPU input IDs 热路径不再通过 `min().item()`/`max().item()`做两次 device-to-host 标量读取；CPU/Mock 边界仍保留值域检查；
- 当前 step 的控制 metadata 必须先更新，再接收 IDs；close 和每次 metadata 更新都清空旧 stage layout。

第一项同步优化只缓存 stage layout/token metadata，并移除 NPU input IDs 的 `min().item()`/`max().item()`。它已通过单测和 A4F2 NPU round-trip，但两个非 profiler C32 护栏分别为 45.656 和 46.800 token/s，低于 P4 均值，不能单独形成收益结论。对应 profile 确认 Attention 侧 `aten::item/_local_scalar_dense` 总耗时从约 1211.681 ms 降到 5.498 ms，说明昂贵回读确实被消除；同时 profile 暴露 FFN 每个 layer 仍重复构造相同 stage forward context，并重复读取相同 token-count 最大值。

第二项同步优化将 FFN Ascend forward context 改为每 step/stage 构造一次、43 层复用；每层只更新 input IDs、AFD metadata 和 MoE layer index。该实现不改变任何 HCCL 消息、顺序、stream 或同步边界。10/10 golden 逐 token 通过后，最终 A8F8 eager/U1、C32、1024 输入/128 精确输出、每轮 128 请求的正式三轮结果为：

| 指标 | P4 U1 C32 | P7 同步优化 | 变化 |
|---|---:|---:|---:|
| output throughput | 49.118 token/s | 57.724 token/s | +17.521% |
| output throughput CV | 0.260% | 0.689% | 均通过 10% 门禁 |
| output token/s/NPU | 3.070 | 3.608 | +17.521% |
| p50 TTFT | 3751.984 ms | 3736.812 ms | -0.404% |
| p50 TPOT | 622.147 ms | 530.345 ms | -14.756% |
| p90 TPOT | 667.083 ms | 574.315 ms | -13.907% |
| p99 TPOT | 670.795 ms | 579.259 ms | -13.646% |

三轮原始 throughput 为 57.277、57.653、58.243 token/s；每轮均为 128/128、无请求失败，fatal、shutdown 和 NPU cleanup 门禁全部通过。该结论只覆盖高负载 C32；P7 仍需补 C1/C8、冷服务重复和同资源非 AFD 对照，未创建性能 tag。

```text
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_a4f2_20260817_1135/summary.json
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_u1_guard_c32_retry_20260817_1150/performance_summary.json
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_u1_profile_20260817_120056/performance_summary.json
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_context_cache_golden_20260817_1315/validation_summary.json
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_context_cache_c32_formal3_20260817_1345/performance_summary.json
```

部署、采集、解析必须分开执行：

1. 先确认目标服务已使用预期 commit 和参数启动；
2. 再启用 profiler 采集 Attention role-local DP0 和 FFN role-local DP0；
3. 最后使用与采集匹配的 CANN 9.0.1 工具解析原始目录；
4. 解析异常时保留原始产物，不覆盖或删除失败现场。

固定：

```bash
export TORCH_PROFILER_WITH_STACK=0
```

建议补充稳定的 `record_function` 区段：

- Attention compute；
- A2F send/receive；
- FFN compute；
- F2A send/receive；
- ubatch split、wait 和 merge；
- per-peer receive、aggregate view、output split 和 per-peer send。

需要统计：

- Attention/FFN 计算时间；
- A2F、F2A 延迟及未重叠部分；
- Attention/FFN overlap；
- FFN wait、free/bubble；
- stage 0/1 不均衡程度；
- 同一 FFN subgroup 内各 Attention peer 的到达偏斜、聚合等待和 FFN batch 放大收益；
- host `.item()`、同步 memcpy 和控制面开销；
- 每侧峰值 HBM、NPU 利用率和启动时间。

第一轮调优矩阵可以从以下值开始，但它们不是产品固定值：

```text
Topology: A8F8（A3）；A8F4（高 HBM A5）
Mode: U1, U2
U2 threshold: 16, 32, 48, 64, 96, 128
HCCL_BUFFSIZE: 在固定请求矩阵下做小范围扫描
```

异步 HCCL 不在当前矩阵中。若未来启用，必须另立里程碑并重新验证消息生命周期、异常取消、stream 同步、buffer 复用和 shutdown，不能把同步路径的正确性结论直接外推到异步实现。

调参顺序固定为 topology -> U1/U2 -> threshold -> HCCL buffer。每轮只改变一个维度；A8F8 和 A8F4 可以有不同最优 threshold，但必须共享同一请求、预热、测量窗口和正确性门禁。

性能热路径中的高频 warning、tensor `.tolist()` 和仅用于调试的 key 构造应降到 debug 或显式开关下。任何日志调整都必须保留错误和生命周期门禁所需信息。

### 5.10 Graph 后续门禁

`P2pHcclAFDConnector` 首版显式只支持 eager。只有 HCCL eager/U2 的正确性、稳定性和性能收益已经解释清楚后，才评估 Graph；Graph 不是当前性能验收的前置条件。

本阶段重点：

- U1 fallback 和 U2 使用不同且稳定的 graph key；
- stage 0/1 使用稳定地址的独立 IDs/hidden buffer；
- capture 期间和 replay 期间消息数量一致；
- 每次 replay 使用当前 step 的 IDs，不能固化首次 capture 的 IDs；
- 不同 token count、capture size 和 U1/U2 切换不串图；
- graph capture 失败、replay 异常和 shutdown 均能清理；
- 冷启动、二次启动和空闲恢复继续通过。

若未来进入 Graph，必须作为独立里程碑重新实现和验收，不能沿用 CAMP2P Graph 的结论。

## 6. A3 性能验收方法

### 6.1 比较矩阵

最终至少比较：

| 方案 | 作用 |
|---|---|
| 非 AFD eager | 基础执行对照 |
| 非 AFD Graph | 非 AFD 最佳稳态对照 |
| AFD eager/U1 | 分离本身的成本和收益 |
| CAMP2P AFD eager/U1、U2 | 已冻结功能对照，不代表 HCCL 主线性能 |
| HCCL P2P AFD eager/U1 | 分离通信成本与 U2 对照 |
| HCCL P2P AFD eager/U2 | 当前候选交付形态，验证双阶段重叠收益 |
| HCCL P2P A8F4 eager/U1 | 非等量聚合/切分成本与 U2 对照 |
| HCCL P2P A8F4 eager/U2 | 候选节省 FFN 资源形态，验证吞吐保持和资源效率 |

### 6.2 公平资源口径

A8F8 使用 16 个 NPU，A8F4 使用 12 个 NPU。只把 A8F8 与一个 8-NPU 非 AFD 实例比较，会把资源增加带来的收益误认为 AFD 架构收益；把 A8F4 直接称为“同资源非 AFD 对照”同样不成立，因为固定 DSV4 非 AFD 基线没有 12-NPU 等价布局。

因此必须同时报告以下口径：

| 对比 | 说明 |
|---|---|
| AFD A8F8 vs 非 AFD 单个 8-NPU 实例 | 反映单服务扩展能力和延迟变化 |
| AFD A8F8 vs 两个非 AFD 8-NPU 实例的总和 | 反映相同 16-NPU 总预算下的资源效率 |
| HCCL A8F4 vs HCCL A8F8 | 反映减少 4 个 FFN rank 后的吞吐保持、延迟和 HBM 变化 |
| HCCL A8F4 的 tokens/s/NPU vs A8F8/非 AFD | 反映归一化资源效率；必须同时给出绝对吞吐 |

如果后续有至少 24 张同型 NPU，可增加两个 A8F4 实例与三个非 AFD 8-NPU 实例的精确 24-NPU 总预算对照。当前 16-NPU A3 环境不能伪造该对照，也不能只按比例外推总吞吐。

所有结果必须同时给出：

- request/s；
- output tokens/s；
- output tokens/s/NPU；
- TTFT；
- TPOT p50/p90/p99；
- 峰值 HBM；
- NPU 利用率；
- A2F/F2A 和 overlap/bubble 指标。

### 6.3 建议门禁

以下是开始正式实验前应确认的建议门禁，不是适用于所有产品场景的永久阈值：

1. 正确性、生命周期和清理门禁必须 100% 通过；
2. HCCL P2P eager/U2 相对同参数 HCCL P2P eager/U1 的 output tokens/s 提升不低于 10%；
3. p99 TPOT 相对选定基线的回退不超过 5%；
4. AFD 相对同总 NPU 预算非 AFD 的收益必须大于 3 轮稳定运行的波动区间；
5. 不允许通过减少输出 token、改变 batch、降低 golden 覆盖或放宽错误检查获得收益；
6. 若吞吐增加但 `tokens/s/NPU` 明显下降，必须明确记录为扩容收益，不能表述为资源效率收益；
7. A8F4 eager/U2 相对 A8F8 eager/U2 的绝对 output tokens/s 建议保持不低于 95%，同时 `tokens/s/NPU` 建议提升不低于 20%；
8. A8F4 的 FFN 峰值 HBM 必须保留预先定义的安全余量，不能以临界 OOM 状态通过吞吐门禁。

第 7 项是首轮建议值，不是已证实结论。最终阈值应在 A3-P4 的 A8F8 参照完成后、A8F4 正式性能跑数前写入验证脚本或实验 manifest，跑完后不根据结果反向调整门禁。

### 6.4 A3 性能完成定义

A3 性能阶段只有在以下材料齐全时才完成：

- 全部对照方案的启动参数；
- 每个点 3 轮原始客户端结果；
- 服务端日志和返回码；
- 环境与精确 git commit；
- Attention/FFN profile 原始产物和解析结果；
- HBM、利用率和清理后的 `npu-smi info`；
- 一份结论明确区分“吞吐扩展”“资源效率”和“延迟”的报告。

通过后建议冻结：

```text
dsv4-afd-a3-hccl-p2p-perf-v1
```

tag message 和报告必须写明最终选择的 A/F 比例。若 A8F4 正确性通过但性能门禁未通过，可以冻结仅用于回归的 `dsv4-afd-a3-hccl-p2p-unequal-v1`，但不得使用 `perf` 命名。

## 7. 面向 A5：现在就要保持的设计边界

本文暂按“A5 指 Atlas A5 / Ascend 950 系列”规划。实际服务器 SKU、SoC 字符串和单机 NPU 数以目标机器审计结果为准。

### 7.1 已知的软件基础和当前缺口

固定 vLLM-Ascend commit 已包含：

- `A5DeviceAdaptor`；
- `SOC_VERSION` 以 `ascend950` 开头时的构建识别；
- 多个 Attention/MoE 算子的 `ascend950` 注册。

但这只说明固定上游存在 A5 基础，不说明 afd-plugin 已支持 A5。当前主线选择标准 HCCL send/recv，A5 不再以前置移植 afd-plugin 自定义 A2E/E2A kernel 为目标；这减少了以下 A3 专用实现对 A5 的阻塞：

- `csrc/npu/build_aclnn.sh` 只接受 `910c`/`ascend910_93*`；
- `a2e_def.cpp` 和 `e2a_def.cpp` 只注册 `ascend910_93`；
- `tools/dsv4/install_plugin.sh` 默认 `SOC_VERSION=ascend910_9362`；
- kernel 中存在 192 KB UB、48 core、window offset 等 A3 假设；
- ACLNN host 侧 HCCL server type 对 910B 和其他 SoC 走不同分支，A5 行为尚未实测。

这些限制仍适用于 CAMP2P 备选路径，但不进入 HCCL P2P 主线。另一方面，也不能因为代码只调用公共 `torch.distributed.send/recv` 就宣称 A5 已支持；A5 的驱动、固件、CANN、torch-npu、HCCL P2P 能力和目标拓扑仍必须实机验证。

### 7.2 A3 开发期间必须做到

- U2 stage、rank 和 token slice 逻辑保持硬件无关；
- 不在模型/connector 核心路径硬编码 A8F8、NPU 0-15 或单机卡数；
- role 数、buffer token 上限、U2 threshold 和 HCCL 配置保持可配置；
- 非等量映射只依赖逻辑 role rank，不依赖物理 device ordinal；`ratio`、peer ranks 和 `seq_lens` 由单一 topology 对象派生；
- HCCL backend、P2P send/recv 或目标拓扑不支持时显式 fail-fast；
- A3/A5 使用独立 venv、CANN 根目录、构建输出、启动 recipe 和验证目录；
- CPU/Mock 测试不依赖 A3 核数或物理 device ordinal；
- performance manifest 记录 SoC、驱动、固件、CANN、torch-npu、拓扑和 NUMA；
- connector 核心只依赖 PyTorch distributed HCCL 公共接口，不引用 CAMP2P 自定义 op；
- 上游 patch 继续标注固定 commit 和 AFD patch marker，便于 A5 栈变化时重放差异。

### 7.3 当前不要提前做的 A5 修改

没有 A5 硬件和匹配工具链时，不应凭 A3 结果猜测并提交以下变更：

- CAMP2P A2E/E2A kernel 的 UB 分配、核数和通信 window；
- A5 特定私有 IPC/SDMA 路径；
- U2 threshold、HCCL buffer 和 Graph capture size；
- CPU/NUMA 绑核和物理 rank 映射。

这些修改需要在 A5 上通过最小组件测试和 profile 驱动。

## 8. A5 到位后的独立适配阶段

### 8.1 A5-H0：硬件与运行栈审计

先记录实际机器，不沿用“A5 应该是什么”的假设。至少保存：

```bash
npu-smi info
npu-smi info -t board -i 0
npu-smi info -t topo
uname -m
```

若目标驱动不支持某个 `npu-smi` 子命令，保存等价的板卡和拓扑查询结果。

同时记录：

- 服务器完整 SKU；
- 实际 SoC 名称和 `SOC_VERSION`；
- 单机 NPU 数、每卡 HBM 和健康状态；
- 驱动、固件、CANN 和 torch-npu 版本；
- A5 对应 ops 包；
- CPU 架构、socket、NUMA 与 NIC/NPU 亲和关系；
- 单机还是跨机 A/F 拓扑。

A5 使用独立运行栈，并按目标产品支持矩阵固定版本。不要直接把 A3 的 CANN 根目录、venv 或已编译 `ascend910_93` 产物复制过去。

### 8.2 A5-H1：标准 HCCL P2P 运行栈门禁

需要完成：

1. 按 A5 产品支持矩阵建立独立 CANN、torch-npu、vLLM 和 vLLM-Ascend 环境；
2. 验证 `torch.distributed` HCCL process group 可创建、销毁和二次创建；
3. 验证 `send/recv` 支持 BF16 hidden/output 和 int32 input IDs；
4. 验证一个 FFN 与多个 Attention peer 的 communicator 建立、固定消息顺序和超时恢复；
5. 确认单机或跨机目标拓扑的 rank、NIC、NUMA 与链路能力；
6. unsupported backend、dtype、拓扑或栈版本明确失败。

只有产品决定重新启用 CAMP2P 备选路径时，才单独建立 A5 自定义算子移植里程碑；它不阻塞 HCCL P2P 主线。

### 8.3 A5-H2：kernel 和通信最小验证

按由小到大的顺序验证：

1. A1F1 IDs `int32` round-trip；
2. A1F1 hidden HCCL send/recv round-trip；
3. A2F1/A4F2 多 peer IDs、hidden 聚合和 output split round-trip；
4. 不同 peer token count、`-1` padding 和 buffer 边界；
5. 连续两个 step 和两个 stage；
6. 多 rank 单机；
7. 若产品拓扑跨机，再增加跨节点 HCCL 测试；
8. 异常取消、connector close 和重新启动。

本阶段重点实测：

- BF16/int32 send/recv 正确性和消息顺序；
- HCCL group 创建、销毁和错误恢复；
- 单链路带宽、时延与多 stage 并发行为；
- HCCL buffer、超时和网络接口选择；
- rank 到物理 NPU/NIC 的映射。

### 8.4 A5-H3：模型正确性回归

A5 必须先生成同平台非 AFD golden，不能只拿 A3 token 文件代替 A5 基线。随后依次验证：

1. Attention/FFN 角色构造和权重所有权；
2. layer 0、2、3、42 的单层/loopback 等价；
3. eager/U1；
4. eager/U2；
5. `A = k x F` 非等量 eager/U1、U2；
6. 冷启动、二次启动、batch、空闲恢复和严格关闭；
7. Graph 仅在另立里程碑并实现后回归。

若 A5 单机有 16 个 NPU，先验证 A8F8，再在 HBM 允许时验证 A8F4；若只有 8 个 NPU，先验证 A4F4，再评估 A4F2。非等量候选必须满足 `A >= F` 且 `A % F == 0`。实际角色映射必须根据 `npu-smi` 拓扑和 NUMA/NIC 关系决定，不能只按 device ordinal 对半切分，也不能在未测 HBM 前假定更少 FFN rank 一定可行。

### 8.5 A5-H4：重新调优和性能验收

A5 需要重新扫描：

- U2 threshold；
- HCCL buffer；
- Attention/FFN 角色数与 `A/F ratio`；
- CPU/NUMA 绑核；
- 单机/跨机 rank 布局。

profiling 仍遵守：部署、采集、解析分离；`TORCH_PROFILER_WITH_STACK=0`；Attention/FFN role-local DP0；使用与 A5 采集环境匹配的 CANN parser。

A5 使用与 A3 相同的公平资源口径，但重新生成全部数字和门禁结论。通过后再创建带 A5 标识的独立 tag，不能复用 A3 性能 tag 代表 A5。

## 9. PD 集成顺序

Mooncake PD 不进入当前 A3 standalone AF 性能开发的关键路径。

进入 PD 的门禁：

- A3 或目标 A5 上 standalone AF 的 HCCL P2P eager/U2 正确性和性能门禁通过；
- AFD 相对非 AFD 的性能收益已经按公平资源口径得到解释；
- A2F/F2A、FFN wait 和 bubble 的主要瓶颈已有 profile 证据；
- 生命周期和自动清理稳定。

进入 PD 后，把 PD 拓扑作为新的独立变量重新验证，不用 standalone AF 的结果直接替代生产拓扑结论。U3 不纳入当前路线。

## 10. 分支、Tag 和产物规范

### 10.1 建议分支和 Tag

| 阶段 | 建议分支/Tag |
|---|---|
| eager/U2 开发 | `feat/dsv4-afd-eager-u2` |
| A3 eager/U2 基线 | `dsv4-afd-a3-eager-u2-v1` |
| HCCL P2P connector 开发 | `feat/dsv4-afd-hccl-p2p` |
| HCCL P2P 非等量开发 | `feat/dsv4-afd-hccl-p2p-unequal` |
| A3 HCCL P2P 非等量正确性基线 | `dsv4-afd-a3-hccl-p2p-unequal-v1` |
| A3 HCCL P2P 性能验收 | `dsv4-afd-a3-hccl-p2p-perf-v1` |
| A5 基线 | 在实际硬件和版本确认后使用 `dsv4-afd-a5-*` 命名 |

每个 tag 应为 annotated tag，tag message 至少包含：

- 基线用途；
- 精确 commit；
- SoC/拓扑；
- CANN、vLLM、vLLM-Ascend；
- U1/U2、eager/Graph 和 connector；
- 主验证产物路径。

### 10.2 验证产物目录

统一使用：

```text
/mnt/workspace/validation/dsv4_afd_a3_<stage>_<commit>_<timestamp>
/mnt/workspace/validation/dsv4_afd_a5_<stage>_<commit>_<timestamp>
```

每个目录至少包含：

- `environment.txt`：环境变量、CANN 和 Python 包；
- `git.txt`：afd-plugin/vLLM/vLLM-Ascend commit 和 worktree 状态；
- `npu_before.txt`、`npu_after.txt`；
- `command.txt` 或结构化启动参数；
- Attention/FFN 日志和返回码；
- golden/batch/性能原始请求结果；
- `validation_summary.json`；
- profiler 原始目录、解析目录和 `profile_validation.json`；
- 清理检查结果。

## 11. 风险和停止条件

遇到以下情况时停止扩大测试规模，先修复当前阶段：

- token 与 golden 不一致；
- IDs/hidden 消息计数或 stage 对应关系不确定；
- 非等量拓扑下 peer 顺序、`seq_lens` 或 output 回传目标不唯一；
- 当前 eager 路径跨 step 使用旧 IDs；
- 出现 NaN/Inf、shape 或 dtype 不一致；
- Attention/FFN 任一侧非 0 退出；
- 冷启动后存在残留进程、端口或 NPU 占用；
- CANN 路径混入其他版本；
- profiler 采集栈和 parser 版本不匹配；
- 性能收益只在单轮出现，或小于稳定运行波动；
- A8F4 FFN rank 的模型加载、专家 workspace 或聚合通信 buffer 接近 OOM；
- A5 上只完成 HCCL 组件验证，没有完成模型和 E2E 验证。

## 12. 后续恢复工作时的最短检查清单

1. 确认当前分支、HEAD、已冻结 tag 和 worktree 状态；HCCL 开发起点应为 `9578dd2`；
2. 激活 A3 固定运行栈并运行 `tools/dsv4/check_runtime.sh`；
3. 确认最近一个已通过阶段及其 `validation_summary.json`；
4. A3-P4 已完成；先阅读 `DEEPSEEK_V4_AFD_HCCL_P2P_A3_P4_PERFORMANCE_REPORT_ZH.md`，不要把当前 U2 当作性能基线；
5. A3-P5 已完成；恢复时先核对 A2F1/A4F2 `summary.json` 和相关回归，不重复改写 topology 协议；
6. A3-P6 的 A8F4 已确认受 EP4 HBM 阻塞，A10F5 受固定栈 EP5 放置阻塞；不要通过超卖或 EPLB 改变语义绕过；
7. 当前执行 A3-P7 的 A8F8 同步 HCCL profile 和优化，明确禁止把异步 HCCL 混入本阶段；
8. C32 同步优化三轮已通过；下一步补 C1/C8、冷服务重复和非 AFD 公平对照，不把 C32 结果外推到全部负载；Graph 不与本阶段同时开发；
9. A3 性能 tag 冻结后再进入 PD 或 A5 硬件差异开发；
10. A5 到位后从硬件审计和独立工具链开始，不复用 A3 二进制；
11. 每次阶段完成都保存日志、原始数据、解析结果和清理证据。

## 13. 一句话路线

先在 A3 上完成标准 HCCL send/recv 的 A8F8 基线、`A = k x F` 组件语义和阻塞式同步优化；A8F4 因 A3 EP4 HBM 不足转到高 HBM A5 完成 E2E、资源效率和公平性能验收。当前阶段不引入异步 HCCL，任何未来异步方案都必须作为独立里程碑重新验收。
