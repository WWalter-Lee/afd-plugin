# DeepSeek-V4 AFD 标准 HCCL P2P A3-P4 性能参照报告

## 1. 结论

A3-P4 已完成 A8F8、eager、标准 HCCL `send/recv` connector 的 U1/U2未调优性能参照、双侧 profile 和生命周期验证。

本阶段可以冻结实验协议和参照数据，但不能把 U2 冻结为性能基线：

- U1、U2 在 concurrency 1/8/32 的三轮 output throughput CV 均小于 10%；
- U2 在 C1 与 C8 没有超过波动范围的稳定收益；
- U2 在 C32 的 output throughput 比 U1 低 `37.570%`，p50 TPOT 高  `68.692%`；
- CANN 9.0.1 profile 表明主要问题是两 stage 的等待/free 放大，而不是单个 MoE dispatch/combine kernel 整体变慢；
- U2 profile 的 128/128 请求、Attention 先停、FFN 正常退出、fatal 日志和 NPU 清理全部通过。

因此当前默认性能参照仍为 eager/U1。U2 保留为已验证的正确性能力和后续调优对象，不用于声明 AFD 性能收益。

## 2. 背景与范围

本阶段使用 `P2pHcclAFDConnector`，hidden、FFN output 和 DeepSeek-V4 input IDs 均经过 `torch.distributed.send/recv` HCCL process group。没有调用CAMP2P A2E/E2A 自定义算子。

本报告只回答 A8F8 下 U1/U2 的未调优参照与瓶颈位置，不回答以下问题：

- AFD 相对同资源非 AFD 部署是否已有最终性能收益；
- A8F4 等非等量拓扑是否更优；
- 128K 上下文、Graph、PD、MTP 或 A5 实机性能。

这些项目继续按路线图的独立门禁执行。

## 3. 固定环境与实验协议

| 项目 | 固定值 |
|---|---|
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| venv | `/mnt/workspace/code/.venvs/afd-v026` |
| vLLM | `568afb3a13806beb53bb2e6bd518269357b237c0` |
| vLLM-Ascend | `80d8c194f7584b17fe08065ea99a130916f6b0e7` |
| afd-plugin 起点 | `9578dd2cb70f9f8db54673a70e8f45fde6479245` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| 拓扑 | Attention NPU 0-7，FFN NPU 8-15，DP8/EP8/TP1 |
| 模式 | eager，A8F8，U1 或 U2 |
| workload | input 1024，output 128，ignore EOS |
| concurrency | 1、8、32 |
| 重复 | 每点 3 轮 |
| 请求数 | `max(8, concurrency x 4)` |
| 预热 | 16 请求，input 256，output 16，concurrency 8 |
| 确定性 | seed 1024，temperature 0 |
| 稳定性门禁 | output throughput CV 不超过 10% |
| profile | C32、128 请求、DP0 only、active 20 step、with stack 关闭 |

正式吞吐与 profile 分开运行。profile 数字包含采集开销，只用于定位，不能替代正式吞吐结果。

## 4. 三轮性能参照

### 4.1 Output throughput

单位为 output tokens/s。

| Concurrency | U1 mean | U1 CV | U2 mean | U2 CV | U2 相对 U1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.508 | 0.555% | 1.511 | 0.561% | +0.177% |
| 8 | 13.249 | 4.395% | 13.702 | 5.648% | +3.421% |
| 32 | 49.118 | 0.260% | 30.664 | 5.185% | -37.570% |

C1/C8 的差值小于对应运行波动，不能判断 U2 获益。C32 的负向差值远大于两侧波动，属于确定性回退。

### 4.2 C32 延迟

| 指标 | U1 | U2 | U2 相对 U1 |
|---|---:|---:|---:|
| p50 TTFT | 3751.984 ms | 4275.868 ms | +13.963% |
| p90 TTFT | 5166.035 ms | 6504.914 ms | +25.917% |
| p99 TTFT | 5768.081 ms | 7366.359 ms | +27.709% |
| p50 TPOT | 622.147 ms | 1049.513 ms | +68.692% |
| p90 TPOT | 667.083 ms | 1110.495 ms | +66.470% |
| p99 TPOT | 670.795 ms | 1117.073 ms | +66.530% |

### 4.3 HBM

| 角色 | U1 峰值范围 | U2 峰值范围 |
|---|---:|---:|
| Attention NPU 0-7 | 59498-59731 MB | 59656-60328 MB |
| FFN NPU 8-15 | 43034-43291 MB | 43188-43443 MB |

U2 没有造成 OOM，但 Attention 侧最高峰比 U1 高约 597 MB，后续非等量拓扑和更长上下文必须继续保留 HBM 门禁。

## 5. Profile 对比

四份 profile 均由 CANN 9.0.1 离线导出并生成完整`ASCEND_PROFILER_OUTPUT`。U1/U2 每个角色均取相同的 20 个 step，step ID为 67-86。

### 5.1 Step 时间

| 角色 | 模式 | Computing mean | Free mean | Stage mean | 非 receive 通信 mean |
|---|---|---:|---:|---:|---:|
| Attention | U1 | 22.327 ms | 143.333 ms | 167.020 ms | 1.359 ms |
| Attention | U2 | 40.632 ms | 515.467 ms | 571.991 ms | 15.893 ms |
| FFN | U1 | 844.574 ms | 20.864 ms | 865.891 ms | 0.452 ms |
| FFN | U2 | 729.934 ms | 304.625 ms | 1038.099 ms | 3.540 ms |

关键观察：

- FFN U2 的 computing mean 比 U1 低 `13.6%`，但 free 增加约 283.8 ms，最终 stage mean 增加约 172.2 ms；
- Attention U2 的 computing 接近 U1 的两倍，且 free 增加约 372.1 ms；
- 当前 Ascend ubatch wrapper 用两个 CPU thread 交替执行，但两个 stage 共享  compute stream；HCCL connector 又使用阻塞式 `send/recv`，切换发生在  receive 返回之后，因此没有形成足以覆盖通信等待的有效重叠；
- 这与 C32 TPOT 回退一致。

### 5.2 FFN MoE 算子

U2 对两个半 batch 各执行一次 FFN，因此 20-step 窗口内每层算子调用数从`860` 增至 `1720`，这是预期的两 stage 语义。设备总时长并未按调用数翻倍：

| 算子 | U1 调用/总设备时间 | U2 调用/总设备时间 | 观察 |
|---|---:|---:|---|
| `aclnnMoeDistributeDispatchV4` | 860 / 13956.575 ms | 1720 / 12607.753 ms | U2 更低 |
| `aclnnMoeDistributeCombineV4` | 860 / 2671.628 ms | 1720 / 1595.141 ms | U2 更低 |
| `aclnnGroupedMatmulSwigluQuantWeightNZ` | 860 / 132.006 ms | 1720 / 178.838 ms | 增幅小于 2 倍 |
| `aclnnGroupedMatmulWeightNz` | 860 / 65.344 ms | 1720 / 94.867 ms | 增幅小于 2 倍 |

因此不能把 U2 回退归因于单个 MoE kernel 退化。半 batch 对通信/计算 kernel有收益，但收益被 stage 调度、host 发射和等待成本抵消。

### 5.3 HCCL 消息

| 角色/模式 | P2P op 数 | HCCS 传输量 | 累计 wait |
|---|---:|---:|---:|
| Attention U1 | 1740 | 28.180 MB | 20336.979 ms |
| Attention U2 | 5200 | 28.180 MB | 24880.469 ms |
| FFN U1 | 1739 | 28.148 MB | 759.148 ms |
| FFN U2 | 3479 | 28.164 MB | 949.309 ms |

U2 的有效 hidden 数据总量没有增加，但消息和同步次数增加。两个 data group分别承载半量 hidden，两个 IDs group 每 step 各传一次 int32 IDs。Attention侧还承担两个 stage thread 的交替和等待，因此消息碎片化与阻塞同步成本集中体现在 Attention free/bubble。

## 6. 生命周期修复

U1 profile 首次采集在全部请求和 raw trace 已完成后，暴露了 Attention 退出与FFN Gloo body receive 并发发生的关闭竞态：Gloo 可能返回来源 rank，但只写入部分 JSON，原逻辑会把零填充尾部交给 JSON decoder。

修复后，控制帧中出现 JSON 不可能包含的 `NUL` 即转换为`AFDControlPlaneClosedError`；FFN 已有关闭逻辑将该异常识别为正常 peer close。非空且无 `NUL` 的错误 JSON 仍然失败，不会掩盖协议损坏。

U2 profile 随后完成严格验证：

- 128/128 正式请求，131072 input tokens、16384 output tokens；
- Attention 与 FFN raw trace gate 通过；
- `log_gate`、`shutdown`、`npu_cleanup_gate` 全部通过；
- 无 `JSONDecodeError`、`AFD NPU FFN worker loop failed` 或 EngineCore fatal；
- 清理后 16 张 NPU 无服务进程。

## 7. 证据目录

| 内容 | 目录 |
|---|---|
| U1 三轮正式参照 | `/mnt/workspace/validation/dsv4_afd_a3_p4_hccl_u1_perf_guarded_log_20260814_182150` |
| U2 三轮正式参照 | `/mnt/workspace/validation/dsv4_afd_a3_p4_hccl_u2_perf_no_debug_log_20260814_163232` |
| U2 双 stage 独立证据 | `/mnt/workspace/validation/dsv4_afd_a3_p4_hccl_u2_stage_evidence_warning_20260814_181226` |
| U1 双侧 profile | `/mnt/workspace/validation/dsv4_afd_a3_p4_hccl_u1_profile_c32_20260814_193819` |
| U2 双侧 profile 与关闭回归 | `/mnt/workspace/validation/dsv4_afd_a3_p4_hccl_u2_profile_c32_20260814_2004` |

U2 三轮目录的顶层 `passed=false` 仅因为当时高频 stage 日志已关闭，runner 无法从日志证明真实 U2；其 warmup、aggregate、log、monitor、shutdown 和 cleanup门禁均通过。独立 U2 stage 目录在随后的一次性 WARNING 方案下同时证明 8 个Attention rank 均出现 stage 1/2，且顶层 `passed=true`。

## 8. 下一步

按路线图进入 A3-P5，在同一 `P2pHcclAFDConnector` 中实现 eager 下`A = k x F` 的非等量 HCCL P2P，首批组件拓扑为 A2F1/A4F2，E2E 目标为A8F4。

实现期间保留以下性能事实作为回归边界：

- ratio=1 的 U1 吞吐不得因非等量代码回退；
- 不把当前 U2 设置为默认；
- A8F4 正确性通过后，再联合比较 A8F8/A8F4 的 U1/U2；
- 后续 U2 优化优先处理独立 comm stream、事件交接、消息碎片化和 stage  到达偏斜，仍只使用标准 HCCL send/recv 接口。
