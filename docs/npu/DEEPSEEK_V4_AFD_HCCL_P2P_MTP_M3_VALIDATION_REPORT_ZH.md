# DeepSeek-V4 AFD HCCL P2P MTP M3 验证报告

## 1. 结论

A3-P7M3 已完成 `P2pHcclAFDConnector` 的 eager/U2 + MTP 功能闭环。支持范围严格限定为：

- A8F8 等量拓扑，Attention DP8、FFN DP8/EP8、TP1/PP1/CP1/DCP1；
- target 与 draft 都使用 eager；
- target decoder 最多两个 microbatch；
- MTP proposer 使用合并后的 target hidden，只执行一个 MTP phase；
- 1 个 MTP layer、`method=mtp`、`num_speculative_tokens=1`；
- hidden 和 output 仍使用同步 `torch.distributed.send/recv`，不引入异步 HCCL。

完整 F0 达到 30/30 串行 golden token IDs 一致，batch 1/8/32 均有效，batch 32 在全部 8 个 Attention rank 上实际执行两个 stage。独立冷启动、SpecDecoding、正常停止、fatal 日志和 NPU cleanup 均通过。

P1 的运行完整性也通过：C32、输入 1024、精确输出 128、128 请求全部成功，双 stage、HBM、日志和清理无异常；但 output throughput 只有 16.238 token/s，相对 M1 MTP/U1 的 28.280 token/s 回退 42.583%，超过路线图的 20% 暂停线。因此本阶段只能冻结功能 tag，不能创建性能 tag，也不能继续扩展 target Graph/U2 + MTP。

## 2. 固定环境

| 项目 | 固定值 |
|---|---|
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| venv | `/mnt/workspace/code/.venvs/afd-v023-vllm-cann` |
| vLLM | `releases/v0.23.0` / `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann` / `3da28f9414583d2d0b672a8f06d1fae142404bda` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| 插件 | `ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd` |
| 拓扑 | Attention NPU0-7，FFN NPU8-15，A8F8 |
| 确定性 | seed 1024、temperature 0 |

运行前环境审计输出：

```text
AFD_DBO_CONFIG_PRESERVED
DSV4_AFD_V023_VLLM_CANN_RUNTIME_OK
torch_npu 2.10.0.post2
vllm 0.23.0
vllm_ascend 0.1.dev1+g3da28f941
```

两个目标上游源码工作树均保持干净，本阶段只修改 `afd-plugin`。

## 3. 关键问题与实现

### 3.1 为什么不能直接复用普通 U2 的 token 中点切分

最初实现把 MTP target verify 与普通 decoder 一样按 token 数量近似对半切分。HCCL 消息和生命周期均能完成，但真实 smoke 的第一个 prompt 在第 12 个输出 token 开始偏离 golden：

```text
actual: ... 3072, 16, 11892, 12082, 17984, 6471
golden: ... 3072, 2148, 11, 223, 1823, 24800
```

MTP verify 的同一请求包含相关的 target/draft token。机械切分可能把它们放入两个 stage，使稀疏/量化 Attention 使用不同 kernel shape；即使通信 shape 合法，也不再保证逐 token 数值一致。

修复后，MTP target decoder 按请求边界切分，完整保留每个请求的 scheduled tokens。该规则只作用于 HCCL P2P eager/U2 + MTP，不改变普通 U2、U1 或 Graph 路径。

失败现场保留在：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m3_smoke_20260820_1540
```

### 3.2 全 DP 一致的 stage 决策

每个 DP rank 的请求数量可能不同。若部分 rank 能切成两个非空 stage、其他 rank 只能形成一个 stage，FFN peers 会进入不同的消息序列并死锁。

当前 DP 同步 payload 增加第五行 `stage0_tokens`，一次 CPU/Gloo all-reduce 后同时得到：

```text
unpadded tokens
padded tokens
graph mode
uniform decode flag
request-boundary stage 0 tokens
```

只有所有 DP rank 的 stage 0 和 stage 1 都非空时才启用 U2；否则所有 rank 全局回退 U1。启用 U2 时，Attention 保存两个精确的 per-rank token count 向量并发送给 FFN，不能再由本 rank 的 split offset 推导其他 rank 的 stage shape。

在 DP8 下，至少需要 batch 16 才能保证每个 rank 有两个可分离请求。因此：

- 串行 golden、batch 1 和 batch 8：预期全局 U1 fallback；
- batch 16、batch 32 和 C32 P1：必须观测真实 U2；
- runner 会根据最大请求 batch 自动选择相应门禁，不再把低并发 fallback 误判为失败。

### 3.3 target U2 与 MTP U1 phase

target decoder 在一个 Python host thread 中保持 `layer -> stage 0 -> stage 1` 顺序。两个 target stage 都完成后，Attention 按 stage 顺序把 pre-HC residual 合并写入当前 step 的 `_mtp_hidden_buffer`，再由上游 proposer 消费。

MTP virtual layer 不跟随 target 拆成两个 stage。FFN 先接受 target decoder stage `[0,1]`，随后只在 stage 0 接收一次 MTP header、post-HC `[T,4096]` hidden 并返回一次 output。这样保持 M1/M2 已验证的 proposer/verify 语义，也避免重复执行学习式 MTP gate。

connector 的 stream pipeline 和 `dbo_yield` 只作用于 decoder phase，MTP phase 走独立同步边界。`RemoteFFNProxy` 显式传递 `phase=mtp`，connector 会拒绝 stage 1 的重复 MTP proposal。

### 3.4 fail-fast 边界

本阶段解除 eager/U2 + MTP 门禁，但继续拒绝：

- target Graph/U2 + MTP；
- full draft ACL Graph；
- A/F 非等量 + MTP；
- 多个 MTP layer 或多个 speculative token；
- PD、sequence parallel、Attention gate；
- TP/PP/CP/DCP 大于 1。

target Graph/U1 + draft eager MTP 仍使用 M2 已验证路径。

## 4. 自动化回归

固定目标栈中以下 271 项测试全部通过：

```text
tests/unit/v1/worker/test_npu_runtime.py
tests/unit/connectors/test_p2p_hccl_connector.py
tests/unit/model_executor/models/test_deepseek_v4_construction.py
tests/unit/model_executor/models/test_deepseek_v4_proxy.py
tests/e2e/test_dsv4_recipe.py
```

覆盖点包括 request-boundary split、低并发全局 fallback、精确 per-rank stage metadata、两个 decoder stage 后单次 MTP phase、target hidden 合并顺序、proxy phase、connector stage guard、recipe gate 和 Graph/U2+MTP fail-fast。定向 Ruff、`git diff --check` 和 role shell 语法检查通过。

## 5. F0 功能结果

### 5.1 定向 smoke

请求边界修复后先完成两个独立 smoke：

| 场景 | 结果 |
|---|---|
| 单请求 | 16/16 token exact，预期 U1 fallback |
| batch 1 | 1/1 token exact |
| batch 16 | 16/16 token exact，全部 Attention rank `stage_count=2` |
| batch 16 stage shape | stage 0/1 均为 `(2,2,2,2,2,2,2,2)` |
| shutdown/cleanup | 通过 |

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m3_smoke_retry_20260820_1705
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m3_u2_batch16_20260820_1620
```

### 5.2 完整 F0

| 门禁 | 结果 |
|---|---:|
| serial golden | 30/30 token exact，0 mismatch |
| batch 1/8/32 | 全部 `valid=true` |
| batch token exact（诊断项） | 1/1、3/8、9/32 |
| batch 32 U2 | 8/8 Attention rank 观测两个 stage |
| batch 32 stage shape | stage 0/1 均为 `(4,4,4,4,4,4,4,4)` |
| startup | 404.230s |
| SpecDecoding | proposal/acceptance metrics 持续更新 |
| shutdown | Attention/FFN rc=0，停止顺序正确 |
| fatal log gate | 两侧通过 |
| NPU cleanup | 首次检查通过，无进程残留 |

并发 batch 的 token exact 只作为诊断值；不同 DP 调度顺序下不把它作为确定性硬门禁。严格确定性由 30 个串行请求逐 token 覆盖。

正式 F0 产物：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m3_f0_20260820_162944
```

## 6. P1 轻量性能 guard

P1 固定 A8F8、C32、输入 1024、精确输出 128、128 请求，16 条 256/16 请求预热，只运行一轮：

| 指标 | M3 eager/U2 + MTP |
|---|---:|
| 请求成功 | 128/128 |
| output throughput | 16.238 token/s |
| output token/s/NPU | 1.015 |
| p50 TTFT | 7724.851ms |
| p50 TPOT | 1746.781ms |
| p99 TTFT | 14013.664ms |
| p99 TPOT | 2336.163ms |
| Attention 峰值 HBM | 60,539 MiB |
| FFN 峰值 HBM | 44,819 MiB |
| 双 stage、fatal、shutdown、cleanup | 通过 |

对比关系：

| 对照 | 吞吐 | M3 差异 | 解释 |
|---|---:|---:|---|
| M1 eager/U1 + MTP | 28.280 token/s | -42.583% | 新增 U2 的直接功能对照，超过 20% 暂停线 |
| P8D eager/U2、MTP off | 16.472 token/s | -1.423% | U2 本身的最近 layer-major 对照，说明 MTP 不是主要新增退化源 |

`performance_summary.json` 的 `passed=true` 表示请求、结果、双 stage、HBM、日志和清理等运行门禁通过；它没有替代跨基线的 20% 比较。阶段结论仍是 P1 性能门禁未通过。

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m3_p1_20260820_164750
```

P1 只有一轮且没有 profiler，不能用于宣称性能收益，也不扩大为三轮 P2。

## 7. 冻结与下一步

M3 可以冻结为 `dsv4-afd-v023-hccl-mtp-u2-v1` 功能 tag。该 tag 的含义是“eager/U2 + MTP 功能正确”，不是 eager/U2 性能基线。

下一步按暂停门禁执行：

1. 对同参数 MTP/U1 和 MTP/U2 采集 Attention DP0、FFN DP0 定向 profile，固定 `TORCH_PROFILER_WITH_STACK=0`，由 CANN 9.0.1 解析。
2. 复用 `P8D-PERF-001`，重点比较每层 HCCL wait、FFN free、stage 到达偏斜、host metadata/send 发射频率和 MTP phase 额外成本。
3. 保持同步 `send/recv` API，不在该定位中引入 `isend/irecv` 或后台通信线程。
4. P1 回到相对 MTP/U1 不超过 20% 回退后，才进入 target Graph/U2 + draft eager MTP。
5. full draft Graph、非等量 + MTP 和多个 speculative token 继续保持 fail-fast。

最终性能收益仍要由三轮 P2、公平同预算 native 对照、波动门禁和 `tokens/s/NPU` 共同证明，M3 功能通过不改变该要求。
