# DeepSeek-V4 AFD HCCL P2P MTP M2 验证报告

## 1. 结论

A3-P7M2 已在目标栈完成 target Graph/U1 + eager MTP 功能闭环。交付范围严格限定为：

- `P2pHcclAFDConnector` 标准 HCCL P2P；
- A8F8 等量拓扑，DP8/TP1/EP8；
- target `FULL_DECODE_ONLY`；
- draft `enforce_eager=true`；
- U1、1 个 MTP layer、`num_speculative_tokens=1`。

10 条 prompt 连续 3 轮共 30/30 最终 token IDs 与 vLLM 0.23 目标栈原生 golden 一致，batch 1/8/32 有效性门禁通过；两次连续冷启动、正常停止、fatal 日志和 NPU cleanup 均通过。

P1 固定 C32、输入 1024、输出 128、128 请求，128/128 成功，output throughput 为 22.835 token/s。相对 M1 eager/U1 + MTP 的 28.280 token/s 回退 19.253%，未越过 20% 暂停阈值，但已接近边界。M2 是功能基线，不是性能基线，也不证明 Graph 或 MTP 有性能收益。

完整 draft ACL Graph 不在支持范围。探索实现虽能 capture 和完成请求，但正式 golden 仅 6/30，一律由 feature validation 拒绝。

## 2. 背景与取舍

M1 已完成 eager/U1 + MTP，M2 的目标是让普通 target decode 使用 Graph，降低 host 发射成本，同时保持严格 AF 参数和计算分离。原计划曾要求 target 与 draft 都真实 capture/replay；实机结果证明不能把“可捕获”当作“正确”。

完整 draft Graph 的探索结果为：

| 门禁 | 结果 |
|---|---:|
| graph capture/startup | 通过 |
| serial golden | 6/30 token exact |
| batch 1 | 1/1 token exact |
| batch 8 | 2/8 token exact |
| batch 32 | 13/32 token exact |

失败稳定复现且没有 HCCL timeout，因此不能归因于单纯通信死锁。当前选择保守的 hybrid 边界：target 使用 `FULL_DECODE_ONLY`，draft 继续 eager。该配置复用已经通过 M1 的 draft 数值路径，同时取得 target graph 的功能基础。

失败探索产物保留在：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m2_smoke_20260818_215657
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m2_golden_20260818_220543
```

## 3. 固定环境

| 项目 | 固定值 |
|---|---|
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| venv | `/mnt/workspace/code/.venvs/afd-v023-vllm-cann` |
| vLLM | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `3da28f9414583d2d0b672a8f06d1fae142404bda` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| 拓扑 | Attention NPU0-7，FFN NPU8-15，DP8/TP1/EP8 |
| 插件 | `ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd` |
| target | `FULL_DECODE_ONLY`，capture size 1/2/4/8 |
| draft | eager，MTP 1 layer，1 speculative token |

golden 固定使用同一目标栈原生结果：

```text
/mnt/workspace/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json
```

验证 runner 原默认值仍指向旧栈 Milestone 0 golden，曾造成一次假失败；本阶段已把默认 golden 修正为上述 v0.23 目标栈基线，并新增回归测试防止再次混用。

## 4. 关键改动及原因

### 4.1 target 与 draft 分阶段执行

FFN runner 不再在 `_ffn_forward()` 内把普通 decoder 与 MTP 作为一个捕获单元执行。decoder key 增加 MTP method、speculative token 数和 draft execution 标识；MTP 不建立 graph key，每个在线 step 都按 eager 协议接收新 header 并执行。在线 hybrid 路径固定为：

```text
接收当前 step target IDs
-> replay target decoder graph
-> 接收当前 step MTP header/hidden
-> eager 执行 MTP FFN
-> 返回 MTP output
```

这样可以避免仅按 target DP shape 复用 graph 时，draft token layout 和旧 buffer 被错误复用。

### 4.2 capture 期间省略 eager drafter

目标上游在一次 dummy call 中先运行 target，再调用 drafter。AFD 两侧使用标准阻塞式 HCCL；若 Attention 在 target graph context 结束后进入 eager draft，而 FFN 仍在另一个 graph context 内等待 draft 消息，两侧会在 context 同步边界互相等待。

修复后，Attention 和 FFN 在 target capture 时都只执行 target：Attention 临时从 capture-time dummy call 省略 `enforce_eager=true` 的 drafter，FFN 只捕获 decoder graph。eager draft 没有需要预捕获的 graph，因此不会漏掉 graph 状态；普通 warmup和在线请求仍完整执行 MTP。

### 4.3 保持 eager MTP HCCL 协议

支持路径不把 MTP header、hidden 或 output 纳入 draft ACL Graph capture/replay，而是继续复用 M1 已验证的固定 header buffer 和同步 HCCL 消息顺序。需要区分 ACL Graph 和 `torch.compile`：即使 draft 配置为 `enforce_eager=true`，vLLM 0.23 仍会在初始 profile 中通过 TorchDynamo 跟踪 draft model。因此编译期跳过 header count tensor 的 Python `.tolist()` 值校验，并让 header 发送复用 connector 现有的 HCCL `_send` lowering；非编译入口仍执行完整值校验，线上 draft 仍按每 step eager 执行。

最终源码 smoke 曾在这一边界发现回归：移除 full draft Graph 探索代码后，初始 profile 因 data-dependent header 校验失败。增加上述最小编译兼容处理和单测后，重试跨过 profile、target capture/replay、请求及清理全流程。独立 MTP graph key、draft ACL Graph dispatch 和 capture-only receive 路径仍未保留，避免把数值门禁失败的完整 draft Graph 死路径留在功能 tag 中。

普通 decoder IDs side channel、layer 0/1/2 复用和 layer 3 后清理语义保持不变。MTP 学习式 gate 不消费 input IDs。

### 4.4 fail-fast 与部署配置

role scripts 在 target eager 和 `FULL_DECODE_ONLY` 两种模式下都显式设置 draft `enforce_eager=true`。feature validation 接受 target Graph + eager draft，拒绝 draft Graph、U2、A/F 不等量、多个 speculative token 和多个 MTP layer。runtime/performance manifest 显式记录 `mtp_draft_execution=eager`。

该限制避免 full draft Graph 的错误结果被配置层静默接受。

### 4.5 日志门禁修正

P1 的 128 个请求全部成功，但旧日志门禁把 shutdown 之后 CANN TBE repository-manager 线程关闭 queue 时的 `EOFError` 判为通用 `Exception in thread` fatal。修复后仍保留通用线程异常门禁，只忽略同时满足以下条件的已知 shutdown traceback：

- 已出现 `[shutdown]`；
- traceback 来自 `tbe/common/repository_manager/utils/multiprocess_util.py`；
- 异常为 queue `EOFError`。

同一 TBE traceback 若发生在 shutdown 前仍判 fatal。新规则重放 P1 原始 Attention/FFN 日志后，两侧 fatal marker 均为空。

## 5. F0 功能结果

| 门禁 | 结果 |
|---|---:|
| serial golden | 30/30 token exact，mismatch 0 |
| batch 1/8/32 | 三组请求结构、输出长度和错误门禁均通过 |
| batch token exact（观测值） | 1/1、3/8、14/32 |
| lifecycle cycle 1 | token exact，startup 392.219s |
| lifecycle cycle 2 | token exact，startup 424.199s |
| 两轮 shutdown | Attention/FFN 返回码均为 0 |
| 两轮 fatal log gate | 通过 |
| 两轮 NPU cleanup | 首次检查通过，无残留进程 |
| 最终源码 smoke | 1/1 token exact，log/shutdown/cleanup 均通过 |

batch 的硬门禁是并发请求结构、完成数、输出长度和无错误；`token_exact` 作为观测值保留，不把并发调度结果冒充串行确定性门禁。串行确定性由独立 30/30 golden 覆盖。

正式产物：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m2_correctness_20260818_2310
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m2_lifecycle_20260818_2316
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m2_final_smoke_retry2_20260819
```

静态检查、格式检查、connector 单测 37/37 和 recipe 回归 51/51 通过。NPU runner 的定向 pytest 在受限测试进程导入 vLLM-Ascend 时被 Triton `get_arch()` 的设备探测阻断，未进入测试函数；新增 runner 逻辑由真实 16 卡 E2E 覆盖。该环境限制不应被记录成单测逻辑失败。

## 6. P1 单点结果

P1 与 M1 使用相同 A8F8、C32、输入 1024、精确输出 128、128 请求和单轮口径：

| 指标 | M2 target Graph + draft eager | M1 eager + draft eager | 差异 |
|---|---:|---:|---:|
| 请求成功 | 128/128 | 128/128 | 持平 |
| output throughput | 22.835 token/s | 28.280 token/s | -19.253% |
| p50 TTFT | 14,247.563 ms | 12,932.671 ms | +10.167% |
| p50 TPOT | 1,591.997 ms | 1,026.239 ms | +55.128% |
| acceptance rate | 86.297% | 85.70% | 仅观测 |
| Attention 最大 HBM | 60,030 MiB | 59,650 MiB | +380 MiB |
| FFN 最大 HBM | 44,527 MiB | 44,253 MiB | +274 MiB |

M2 无 OOM、timeout 或请求失败，最终 HCCL/shutdown/NPU cleanup 正常。吞吐回退未超过 P1 的 20% 暂停线，因此功能阶段可以冻结；但只低 0.747 个百分点，且 TPOT 明显上升，P2 必须优先分析 target graph 与 eager draft 的阶段切换、host 发射和等待成本。

P1 原始 `performance_summary.json` 由修复前日志规则生成，因 shutdown-only TBE EOF 记为 `passed=false`；其中 warmup、128 请求、aggregate、NPU monitor、shutdown、U1 和 cleanup 子门禁全部通过。用修复后的同一代码重放原始日志，Attention/FFN log gate 均通过。为保持原始证据可审计，不回写该 summary。

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m2_p1_20260818_2328
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m2_p1_20260818_2328/log_gate_recheck.json
```

## 7. 当前边界与下一步

M2 不支持 draft ACL Graph、U2 + MTP、A/F 非等量 + MTP、多个 speculative token、多个 MTP layer、PD、sequence parallel 或 Graph 非等量拓扑。full draft Graph 的失败结果不得用于性能测试。

下一步进入 A3-P8 正式性能阶段：

1. 先完成 MTP-off 的 HCCL P2P eager U1/U2 与同总 NPU 预算非 AFD 三轮公平对照，回答 AFD + microbatch 是否有收益。
2. 再把 MTP-on eager/U1 和 target Graph/U1 作为独立维度，报告 acceptance、proposal/verify 开销和每个最终 token 的通信成本。
3. 对 M2 P1 的阶段切换、host 发射和 FFN 等待做 Attention DP0/FFN DP0 定向 profile；固定 `TORCH_PROFILER_WITH_STACK=0` 并由 CANN 9.0.1 解析。
4. 只有三轮吞吐收益超过波动门禁且公平资源口径成立，才创建性能 tag。M2 当前只能创建功能 tag。
