# DeepSeek-V4 AFD HCCL P2P A3-P5/P7 实施报告

## 1. 结论

本轮完成了标准阻塞式 HCCL `send/recv` 主线的两个交付：

1. A3-P5 `A = k x F` 非等量 connector 语义和 A2F1/A4F2 NPU 组件闭环；
2. A3-P7 A8F8 eager/U1 同步 host 热路径优化，并在 C32 取得正式三轮候选收益。

本轮没有引入异步 HCCL。代码不使用 `isend/irecv`、后台通信线程、额外通信 stream 或自定义异步传输 op；`torch.distributed.send/recv` 和现有 `torch.npu.synchronize()` 完成边界保持不变。

A8F4 实模 E2E 没有在 A3 上强行绕过：EP4 在 64 GiB A3 上模型构造 OOM；A10F5 又被固定 vLLM-Ascend 的 EP5 专家放置检查拒绝。A8F4 完整门禁转移到高 HBM A5。

## 2. 关键实现

### 2.1 非等量 topology

- 公共 P2P topology 支持 `A >= F` 且 `A % F == 0`；CAMP2P 仍保持 A=F 门禁；
- Attention rank 映射到确定的 FFN subgroup；每个 FFN 按 peer rank 顺序接收 IDs 和 hidden；
- FFN 直接向预分配聚合 buffer 的不重叠 slice 接收，计算后按同一 `peer_slices` 回传；
- 每个 subgroup 只有第一个 Attention peer 发送控制 metadata；
- role device list、DP/EP、FFN token capacity 和验证 manifest 均已参数化。

### 2.2 同步 host 热路径

- 每个 step/stage 只解析一次 Attention token counts、peer ranks 和 slices；
- FFN 每个 stage 的 token layout 在 43 层中复用；
- NPU input IDs 不再通过 `min().item()`/`max().item()`做两次 device-to-host 标量读取；CPU/Mock 边界继续做 dtype 和值域检查；
- FFN Ascend forward context 从每 layer 构造改为每 step/stage 构造一次；每层只更新 input IDs、AFD metadata 和 MoE layer index；
- 当前 control payload 在接收 IDs 前应用；每次 step 更新和 close 都清空旧 stage layout，防止跨 step 复用。

## 3. 为什么修改

P4 profile 表明 U2 的半 batch kernel 收益被等待、host 发射和 stage 调度成本抵消。先做同步路径优化可以在不改变通信生命周期的前提下减少确定的固定开销，也为 A5 保留最简单的公共 HCCL接口。

第一项 layout/readback 优化的初始非 profiler 护栏没有获益，但 CANN 9.0.1 profile 证明 Attention 侧 `aten::item/_local_scalar_dense` 总耗时从约 1211.681 ms 降到 5.498 ms。该 profile 进一步显示 FFN 对不变 token metadata 重复执行 per-layer forward-context 初始化，因此增加第二项 per-stage context 复用。

这两步拆开验证的意义是：不把通信异步、调度和 metadata 重构同时混入一个结果；首项失败时有 profile 证据驱动下一项，而不是继续盲扫参数。

## 4. 验证结果

### 4.1 功能与组件

| 门禁 | 结果 | 证据 |
|---|---|---|
| CPU/Mock 相关回归 | 210 项通过 | topology、connector、NPU runner、recipe |
| A2F1 NPU round-trip | 通过 | `dsv4_afd_a3_p5_hccl_a2f1_20260817_1055` |
| A4F2 NPU round-trip | 通过 | `dsv4_afd_a3_p5_hccl_a4f2_20260817_1105` |
| 优化后 A4F2 round-trip | 通过 | `dsv4_afd_a3_p7_sync_hccl_a4f2_20260817_1135` |
| 最终代码 golden | 10/10 逐 token 一致 | `dsv4_afd_a3_p7_sync_hccl_context_cache_golden_20260817_1315` |
| shutdown/cleanup | 通过 | 双侧 return code 0，无 NPU 残留进程 |

### 4.2 A8F4/A10F5 预检

- A8F4 FFN EP4 每 rank 持有 64 个专家；模型构造约 60.62 GiB 已激活时仍需再分配 514 MiB，A3 OOM；
- A10F5 的 EP5 无法均匀放置 256 个专家，固定栈报 `allocated=52, placement=51`；
- 不启用冗余专家/EPLB 绕过，因为它会改变专家放置、HBM 和性能语义。

### 4.3 A8F8/U1/C32 性能

固定请求：random input 1024、exact output 128、C32、每轮 128 请求、seed 1024、temperature 0、ignore EOS。

| 指标 | P4 U1 C32 | P7 同步优化 | 变化 |
|---|---:|---:|---:|
| output throughput | 49.118 token/s | 57.724 token/s | +17.521% |
| 三轮 throughput | 48.995/49.066/49.294 | 57.277/57.653/58.243 | 全部高于旧区间 |
| throughput CV | 0.260% | 0.689% | 通过 10% 门禁 |
| output token/s/NPU | 3.070 | 3.608 | +17.521% |
| p50 TTFT | 3751.984 ms | 3736.812 ms | -0.404% |
| p50 TPOT | 622.147 ms | 530.345 ms | -14.756% |
| p90 TPOT | 667.083 ms | 574.315 ms | -13.907% |
| p99 TPOT | 670.795 ms | 579.259 ms | -13.646% |

正式三轮全部完成 128/128 请求，failed=0；fatal log、shutdown 和 NPU cleanup 均通过。

## 5. 证据目录

```text
/mnt/workspace/validation/dsv4_afd_a3_p5_hccl_a2f1_20260817_1055
/mnt/workspace/validation/dsv4_afd_a3_p5_hccl_a4f2_20260817_1105
/mnt/workspace/validation/dsv4_afd_a3_p6_hccl_a8f4_u1_smoke_20260817_1120
/mnt/workspace/validation/dsv4_afd_a3_p6_hccl_a10f5_u1_smoke_20260817_1125
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_a4f2_20260817_1135
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_u1_profile_20260817_120056
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_context_cache_golden_20260817_1315
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_context_cache_c32_formal3_20260817_1345
```

## 6. 边界和下一步

C32 已达到候选收益门禁，但 P7 尚未整体冻结：

- 补 C1/C8 同协议三轮，确认低并发没有显著回退；
- 补独立冷服务重复，解释前期 49.906 与后续稳态结果的差异；
- 完成非 AFD 同资源口径对照，区分 AFD 架构收益和增加 NPU 的扩容收益；
- 再评估 U2 threshold/HCCL buffer；当前仍不引入异步 HCCL；
- 完整 P7 门禁通过前不创建 `perf` tag。

A5 到位后，先审计独立运行栈和 HBM，再执行 A8F8 与 A8F4 实模正确性、稳定性和公平性能验收；A3 的 C32 数字不能直接外推到A5。
