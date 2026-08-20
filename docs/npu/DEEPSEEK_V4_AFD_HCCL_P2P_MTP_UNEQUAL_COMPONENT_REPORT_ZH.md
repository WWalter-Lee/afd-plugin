# DeepSeek-V4 AFD HCCL P2P 非等量 MTP 组件验证报告

## 1. 结论

在固定 vLLM 0.23 + vLLM-Ascend `rfc/vllm_cann` 目标栈上，
`P2pHcclAFDConnector` 已完成 eager `A = k x F` 与 MTP 的组件级功能闭环。
A1F1、A2F1 和 A4F2 均通过 target decoder 两个 stage、连续两个不同 step，
以及每 step 一个合并 MTP phase 的真实 NPU HCCL round-trip。

本结果冻结为 component functional snapshot，不是 A8F4 实模 E2E 或性能基线。
A3 上 A8F4 仍受 FFN EP4 专家权重峰值 HBM 限制，实模 golden、batch、生命周期
和性能验证转到高 HBM A5 完成。

## 2. 背景和目标

M1-M4 已完成 A8F8 等量拓扑的 eager/Graph、U1/U2 与 MTP 组合。普通 decoder
数据面也已支持 eager `A = k x F`：一个 FFN rank 聚合连续多个 Attention rank 的
IDs/hidden，计算后按原 token 切片返回 output。MTP 数据面此前仍有三处一对一假设：

- 功能校验、recipe 和角色脚本拒绝 MTP 下 `A != F`；
- Attention 发送的 DP token count 被要求直接等于 FFN world size；
- FFN 只接收一个 Attention header，并把全部 MTP hidden 视为一个 peer slice。

本阶段目标是在不增加新 connector、不修改上游 vLLM/vLLM-Ascend、不引入 CAMP2P
自定义 op 或异步 HCCL 的前提下，把现有标准 HCCL P2P MTP 协议泛化到整数倍拓扑。

## 3. 协议改动

拓扑仍使用已有确定性映射：

```text
ratio = A / F
Attention rank a -> FFN rank floor(a / ratio)
FFN rank f       -> Attention ranks [f * ratio, (f + 1) * ratio)
```

Attention 侧收到的 MTP token count 是 A 长度向量。发送 header 前将其 reshape 为
`[F, ratio]` 并按 subgroup 求和，得到 F 长度向量。每个 Attention peer 发送：

```text
[magic, speculative_step, local_num_tokens, ffn_size, ffn_count_0, ...]
-> local post-HC hidden [T, H]
```

FFN 按 Attention role rank 升序收齐本 subgroup 的所有 header，并执行以下检查：

- magic、FFN world size 和 speculative step 合法；
- 所有 peer 携带的 F 长度汇总向量完全一致；
- peer 的本地 token 数均在 buffer capacity 内；
- peer 本地 token 总和等于汇总向量中当前 FFN rank 的值；
- 聚合 token 总数不超过 FFN `max_num_batched_tokens`。

检查通过后生成一次性 MTP layout，按相同 peer 顺序接收 hidden。FFN runner 仍只执行
一次合并后的 draft MoE；返回时复用 `HCCLP2PTransferState.peer_slices` 拆分 output，
因此没有复制模型计算，也没有修改 runner 的 MTP 调用契约。

## 4. 生命周期和门禁

每个 stage 最多存在一个未消费的 MTP layout。以下情况显式失败：

- 未消费旧 layout 时再次接收 header；
- 没有先接收 header 就接收 MTP hidden；
- 调用方 token 总数与 header 聚合值不一致；
- 多 peer header 的 step 或汇总向量不一致；
- subgroup 本地 token 总和与 FFN 汇总值不一致。

hidden receive 在进入通信前取走 layout，异常不会把旧 layout 留给下一 step；
connector `close()` 也会清空所有 MTP layout 和 buffer 状态。

功能边界调整为：

- 支持 eager U1/U2 + MTP 的 `A >= F` 且 `A % F == 0`；
- 继续只支持 1 个 MTP layer 和 `num_speculative_tokens=1`；
- Graph 非等量继续在 feature validation、recipe 和角色脚本 fail-fast；
- full draft Graph、Graph U3、任意非整数 A/F、PD 和 sequence parallel 仍不支持。

## 5. 验证结果

### 5.1 CPU/Mock

```text
tests/unit/connectors/test_p2p_hccl_connector.py       59 passed
tests/e2e/test_dsv4_recipe.py                          58 passed
tests/unit/v1/worker/test_npu_runtime.py -k feature_validation
                                                       44 passed
ruff check                                             passed
```

覆盖 Attention A->F count 投影、FFN 多 header fan-in、hidden 聚合、output 拆分、
header 不一致、subgroup 总数不一致、一次消费、close 清理，以及 eager 接受/Graph 拒绝
非等量 MTP。

### 5.2 真实 NPU HCCL

所有拓扑均使用 `stages=2`、`steps=2`、`--enable-mtp`。每个 step 的 decoder stage
和 MTP 使用不同 token 分布，检查跨 stage、跨 step 不复用旧布局。

| 拓扑 | 设备 | 结果 | 关键覆盖 |
|---|---|---|---|
| A1F1 | A:0，F:8 | passed | 等量 MTP 回归、两 stage、两 step、正常 close |
| A2F1 | A:0,1，F:8 | passed | 双 peer header/hidden fan-in、output fan-out |
| A4F2 | A:0-3，F:8,9 | passed | 两个独立 subgroup 并行 fan-in/fan-out |

证据目录：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_equal_a1f1_20260820_203703
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_unequal_a2f1_20260820_203437
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_unequal_a4f2_20260820_203549
```

三组 summary 均为 `passed: true`，全部 worker exit code 为 0，全部 connector
`closed: true`。

### 5.3 首轮失败说明

首次 A2F1 使用 `1000 + attention_rank` 作为 BF16 hidden 识别值。`1001` 在 BF16
精度下舍入为 `1000`，FFN 已完成协议聚合，但测试按整数 `1001` 比较而失败。该问题不是
HCCL 消息错序或模型数值错误。识别值改到 BF16 可精确表示范围后 A2F1、A4F2 均通过。

同时，组件工具改为任一 worker 非零退出后立即终止其余 worker，避免失败方退出后通信
对端等待完整 timeout。

## 6. 意义和剩余工作

本阶段证明 MTP 的通信语义可以复用现有标准 HCCL P2P connector 和 decoder 的
fan-in/fan-out state，不需要增加第三条 connector 或把 MTP MoE 权重复制回 Attention。
它也保持了 A5 主线需要的公共 `torch.distributed.send/recv` 接口。

尚未证明的内容包括：

- A8F4 实模能在目标硬件加载并完成 token-exact golden；
- 非等量 MTP 相对等量拓扑或非 AFD 的吞吐收益；
- A5 上的 HCCL、HBM、NUMA/NIC 和最佳 A/F 比例；
- 多 speculative token 的 header、buffer 和 proposer/verify 生命周期。

下一功能阶段是独立审计和实现多 speculative token。非等量 A8F4 的实模 F0/P1
保留为 A5 到位后的硬件验收项；功能组合闭环前不恢复完整 P2 性能矩阵。
