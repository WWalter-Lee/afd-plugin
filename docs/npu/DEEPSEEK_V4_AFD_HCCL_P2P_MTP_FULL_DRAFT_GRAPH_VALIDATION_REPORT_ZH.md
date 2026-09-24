# DeepSeek-V4 AFD HCCL P2P MTP Full Draft Graph 验证报告

## 1. 背景与结论

A3-P7M7 解决了 M2 探索阶段 full draft ACL Graph 只有 6/30 golden token IDs
一致的问题。在固定 vLLM 0.23 + vLLM-Ascend `rfc/vllm_cann` 栈上，
`P2pHcclAFDConnector` 现在支持 target 和 MTP draft 都使用
`FULL_DECODE_ONLY` ACL Graph，并完成 A8F8 U1/U2 实模功能闭环。

最终结果：

- U1、U2 均达到 30/30 串行 golden token exact；
- batch 1/8/32 均有效，U2 batch 32 实际执行两个 target stage；
- A4F2 完成 target Graph + MTP draft Graph 的真实 NPU 组件 capture/replay；
- P1 为 128/128 成功、0 failed，MTP acceptance rate 86.75%；
- fatal log、Attention 先停/FFN 后退和 NPU cleanup 全部通过。

这是 full-draft Graph 的 functional snapshot，不是 performance baseline。P1 单轮 output
throughput 为 27.510 token/s，只用于检查灾难性回退；不得据此宣称 AFD、Graph、MTP 或
microbatch 已有正式性能收益。

## 2. 固定环境与支持范围

| 项目 | 固定值 |
|---|---|
| 日期 | 2026-08-21 |
| 基础提交 | `2860d09` |
| 开发分支 | `feat/dsv4-afd-hccl-mtp-full-draft-graph` |
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| venv | `/mnt/workspace/code/.venvs/afd-v023-vllm-cann` |
| vLLM | `releases/v0.23.0` / `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann` / `3da28f9414583d2d0b672a8f06d1fae142404bda` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| 实模拓扑 | Attention NPU0-7，FFN NPU8-15，A8F8，DP8/EP8 |
| 执行 | target/draft `FULL_DECODE_ONLY`，U1/U2 |
| MTP | 1 个 MTP layer，`num_speculative_tokens=1` |
| 确定性 | seed 1024、temperature 0 |

通信仍使用标准 HCCL P2P。没有使用 CAMP2P 自定义 op、`isend/irecv`、后台通信线程，
也没有修改目标 vLLM 或 vLLM-Ascend 源码。

## 3. 关键改动、原因和意义

### 3.1 target 与 draft 使用独立 Graph cache

FFN 的 target decoder 和 MTP virtual layer 具有不同模型、shape 和 HCCL 消息序列，不能
共用一个 Graph cache。实现新增 `_mtp_acl_graphs`，使用独立 MTP key、capture 和 replay；
target U2 结束后只执行一次合并的 MTP stage 0。

意义是 draft Graph 不再借用 target 的执行状态，避免 replay 错图或漏掉远端 MTP MoE。

### 3.2 Graph 内不解析动态 header

eager MTP 先接收 header 并复制到 CPU 解析；这个动作不能进入 NPUGraph。Attention 现在在
进入 draft Graph 前，根据 control-plane DP metadata 更新稳定的 int32 header buffer；FFN
capture 使用同一 scheduler-owned peer layout，只在图中记录 header receive、hidden receive
和 output send，不在图中执行 `.cpu()` 或 Python 分支。

意义是 header/hidden/output 仍保持原 wire 顺序，同时满足 Graph capture 的静态内存要求。

### 3.3 draft phase 必须显式配对

target step 完成所有请求时，上游可能不调用 proposer，因此 FFN 不能在每个 target 后无条件
等待 MTP 数据。control payload 新增 phase marker：只有 draft runnable 真正执行时 Attention
才发送 marker；FFN 若先收到下一 target 或 shutdown payload，会暂存该 payload并跳过 draft。

marker 还携带本轮 draft 实际走 Graph replay 还是 eager。配置启用 Graph 不等于当前 batch
命中已捕获 Graph；两侧严格按 marker 使用同一种模式，避免 eager receive 把下一轮 token IDs
误读为 MTP header。

意义是 phase 生命周期与真实 proposer 调用一致，跨 step、请求完成和 shutdown 都不会遗留
旧 MTP 消息。

### 3.4 live Graph miss 双侧回退 eager

Attention 的 `ACLGraphWrapper` 如果在 live request 发现新 batch descriptor，默认会单边动态
capture；远端 FFN 无法在同一时刻原子创建匹配 Graph。wrapper 现在只在已存在 Graph entry
时标记 replay；entry miss 时临时把本次 draft runtime mode 改为 eager，调用结束后恢复父
forward context。

意义是禁止运行期单边扩充跨进程 Graph cache，保持标准 HCCL send/recv 严格配对。

### 3.5 MTP peer layout 对齐 capture bucket

最后一个动态 batch 问题是 Attention 将 live token count `6` 分派到 capture bucket `8`，
FFN 却用原始 `6` 查 MTP Graph，导致 peer-layout miss。FFN 现在把每个 Attention peer 的
MTP token count 分别向上归一到 `[1,2,4,8]` 等实际 capture size，再生成 Graph key。

这不是把所有 peer 聚合成一个 shape。A4F2 等非等量拓扑仍逐 peer 保留 bucket，因此不同
HCCL slice 不会错误复用；超过最大 capture size 的请求继续走 eager。

### 3.6 recipe 和组件工具

recipe 增加 `MTP_DRAFT_EXECUTION=eager|graph` 和 `--mtp-draft-execution`，运行 manifest 显式
记录 draft 模式。真实 NPU round-trip 工具增加 `--mtp-graph-transport`，覆盖 A1F1/A2F1/A4F2
的 MTP header、hidden、output capture/replay 和 close。

## 4. 自动化与组件验证

最终相关回归共 316/316 通过，覆盖：

- control payload 新字段的 JSON 向后兼容和 phase marker 收发；
- proposer 未执行、实际执行、下一 target 提前到达和 shutdown；
- draft Graph hit、live miss eager fallback、runtime mode 恢复和 replay 后 stream 同步；
- target eager/Graph 与 draft eager/Graph 的独立组合分派；
- MTP Graph key 的 speculative signature、精确 peer layout 和 capture bucket；
- U1/U2、MTP-off、target Graph + eager draft、非等量 Graph 和 recipe 回归。

Ruff、`compileall`、shell `bash -n` 和 `git diff --check` 均通过。

A4F2 真实 NPU 组件使用 Attention NPU0-3、FFN NPU8-9，两个 target stage 和两个 eager
step 后完成 target Graph capture/replay，再完成一个合并 MTP Graph capture/replay。6 个
worker 返回码均为 0，全部 connector 正常 close：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_full_draft_graph_unequal_a4f2_20260821_m7
```

## 5. F0 正确性结果

golden 是同一固定运行栈、同一模型、seed 1024、temperature 0 的非 AFD 确定性 token-ID
基线。10 条 prompt 连续运行 3 轮，共 30 个完整 token 序列逐 token 比较。

| 门禁 | U1 | U2 |
|---|---:|---:|
| serial golden | 30/30 | 30/30 |
| batch 1/8/32 | 全部有效 | 全部有效 |
| batch token exact（诊断） | 1/1、3/8、13/32 | 1/1、3/8、9/32 |
| 两个 target stage | 不要求 | 已观测 |
| startup | 416.247s | 444.271s |
| fatal/shutdown/cleanup | 通过 | 通过 |

并发 batch token exact 受 DP 调度顺序影响，只作诊断；严格确定性硬门禁是串行 30/30。

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_full_draft_graph_u1_correctness_20260821_m7
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_full_draft_graph_u2_correctness_20260821_m7
```

## 6. 动态 Batch 与 P1

动态 batch smoke 使用 C32、1024/32、32 请求，32/32 成功，60.038 output token/s；它验证
了请求结束时省略 draft、Graph/eager phase marker 和 `6 -> 8` capture bucket，不作为性能点：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_full_draft_graph_u2_p1_smoke_20260821_m7_retry9
```

正式 P1 使用 16 条预热请求和一轮 128 条请求：

| 指标 | 结果 |
|---|---:|
| 请求成功 | 128/128 |
| failed | 0 |
| 输入/输出 token | 131,072 / 16,384 |
| duration | 595.572s |
| output throughput | 27.510 token/s |
| output token/s/NPU | 1.719 |
| p50/p90 TTFT | 8059.608 / 26562.671 ms |
| p50/p90 TPOT | 1339.715 / 1482.432 ms |
| MTP drafted/accepted | 8,739 / 7,581 |
| MTP acceptance rate | 86.75% |
| Attention 峰值 HBM | 61,526 MiB |
| FFN 峰值 HBM | 45,162 MiB |
| U2/log/shutdown/cleanup | 全部通过 |

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_full_draft_graph_u2_p1_20260821_m7_final
```

P1 没有三轮波动、MTP-off、Graph/U1/U2 或同预算 native 对照，不能关闭
`P8D-PERF-001`，也不能创建 performance tag。

## 7. 边界和下一步

M7 支持 A8F8 实模 U1/U2 full-draft Graph，并在 A4F2 完成非等量组件验证。A8F4 实模仍因
A3 FFN EP4 HBM 不足保留到高 HBM A5。当前还不支持：

- 多 speculative token；
- Graph U3；
- TP、PP、SP、CP 或 DCP 大于 1；
- Mooncake PD；
- `A < F` 或 `A % F != 0`；
- Attention-side gate。

下一阶段按既定功能优先顺序进入 M8：先支持 TP，并验证 rank mapping、DP count 到 TP peer
layout 的扩展和 TP collective 与 HCCL P2P 的顺序；随后依次处理 SP/CP/DCP 和 PP。M9 再进入
Mooncake PD。多 speculative token 独立实现；A8F4 full-draft Graph 实模 F0 在 A5 完成。

功能组合闭环后才恢复三轮 P2，以 Graph/U1、Graph/U2、MTP on/off 和同预算 native Graph
公平对照回答“开启 AFD 和 microbatch 是否有性能收益”。
