# DeepSeek-V4 AFD HCCL P2P MTP M4 验证报告

## 1. 背景与结论

A3-P7M4 完成了 `P2pHcclAFDConnector` 的 target Graph/U2 + eager draft MTP 功能闭环。该阶段复用 M2 的 target Graph + eager draft 语义和 M3 的 request-boundary U2 切分，不改变标准 HCCL 接口：hidden/output 仍由同步 `torch.distributed.send/recv` 传输，不使用 CAMP2P 自定义 op、`isend/irecv` 或后台通信线程。

支持范围严格限定为：

- A8F8 等量拓扑，Attention DP8、FFN DP8/EP8；
- TP1、PP1、CP1、DCP1；
- target `FULL_DECODE_ONLY`，target decoder 最多两个 microbatch；
- draft `enforce_eager=true`，合并 target 后只执行一次 MTP phase；
- 1 个 MTP layer、`method=mtp`、`num_speculative_tokens=1`。

F0 达到 30/30 串行 golden token IDs 一致，batch 1/8/32、真实双 stage、ACL Graph capture/replay、正常停止、fatal log 和 NPU cleanup 门禁均通过。

P1 固定 C32、输入 1024、精确输出 128、128 请求，结果为 128/128 成功、0 failed、31.473 output token/s，MTP acceptance rate 84.51%。这是单轮功能 guard，不是性能基线，也不能据此宣称 AFD、Graph、MTP 或 microbatch 已产生正式性能收益。

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

本阶段只修改 `afd-plugin`，没有修改目标 vLLM 或 vLLM-Ascend 源码。

## 3. 关键实现与原因

### 3.1 解除组合门禁

feature validator、recipe shell 和验证 runner 解除 `target Graph/U2 + MTP` 的旧拒绝。其他边界保持不变：draft 必须 eager，MTP 只允许一个 speculative token，A/F 必须等量，Graph U3、full draft Graph、PD 和复杂并行仍 fail-fast。

target decoder 继续按 M3 的请求边界切成两个 stage。两个 stage 完成后才合并 target hidden，并执行一次 eager MTP proposal；MTP phase 不拆成 U2，不进入 target ACL Graph。

### 3.2 启动期重复 graph key 必须双侧 replay

第一次真实 smoke 在启动抓图阶段挂住。MTP request-boundary U2 对最小 capture shape 可能回落到已有 U1 graph key：

- Attention wrapper 发现 key 已存在后直接 replay；
- FFN 原实现遇到重复 key 时跳过 capture，也没有 replay；
- Attention graph 内的 HCCL send 因而没有匹配的 FFN receive。

修复后，FFN 在 capture 请求命中已有 decoder graph key 时同步 replay target graph 并返回，不进入重复 capture，也不执行 eager MTP phase。这样 A/F 两侧 graph 中的 HCCL op 顺序保持严格配对。

### 3.3 在线动态抓图不能由 Attention 单边决定

第二类问题出现在 P1 的新 stage shape。Attention 只有进入 model wrapper 后才能发现 Graph/U2 key 未命中；此时 FFN 已经收到 step metadata，并按 `is_graph_capturing=false` 进入 eager。若 Attention 在线动态抓图，会形成：

```text
Attention: Graph capture + graph-visible HCCL
FFN:       eager + ordinary HCCL
```

初版尝试仅把 Attention 的 connector-owned comm stream 切回 capture 主 stream，消除了 `capture model contains a stream that was not joined`，但无法解决 A/F 执行模式不一致，负载仍会等待。

最终策略是只允许同步 startup dummy run 创建 U2 graph key：

- 已捕获 stage-shape key：A/F 两侧 replay Graph/U2；
- 在线 key miss：Attention 在发送本 step 数据前把 target 整步降为 eager/U2，FFN 原本也因 graph cache miss 走 eager；
- connector 在真实 capture 时禁用 eager comm-stream pipeline，并使用 graph-visible HCCL op；
- graph 外 eager/U2 继续保留既有 layer-major comm stream/event 路径。

该策略避免运行时单边扩充 graph cache。代价是未预抓取 shape 的请求不会获得 Graph replay 收益，但功能和通信顺序是确定的；后续若要扩大 Graph 覆盖，应增加双方同步的启动 capture shape，而不是在线单边抓图。

## 4. 自动化回归

新增或调整的测试覆盖：

- validator 和 recipe 接受 target Graph/U2 + eager draft MTP；
- target Graph/U2 capture 省略 eager drafter，在线请求仍执行单次 MTP phase；
- FFN 重复 graph key 同步 replay target，不重复 capture、不执行 MTP；
- U2 graph key 按两个 stage token shape 和 LoRA descriptor 区分；
- 在线 graph key miss 整步回退 eager，命中时保持 FULL replay；
- 动态 capture 不进入 connector-owned Attention comm stream；
- capture 下延迟 receive wait 为 no-op，并使用 graph-visible HCCL send/recv；
- 既有 eager U1/U2、Graph U1/U2、MTP M1/M2/M3 和 DSV4 角色构造不回退。

定向 Ruff、shell 语法、`git diff --check` 和固定目标栈相关回归均作为提交门禁。

## 5. F0 功能结果

### 5.1 最终结果

| 门禁 | 结果 |
|---|---:|
| serial golden | 30/30 token exact，0 mismatch |
| batch 1/8/32 | 全部 `valid=true` |
| batch token exact（诊断项） | 1/1、3/8、9/32 |
| U2 | batch 32 观测真实双 stage |
| ACL Graph | 8 个 Attention rank 完成 capture/replay |
| startup | 432.304s |
| shutdown | Attention/FFN 正常停止 |
| fatal log gate | 两侧通过 |
| NPU cleanup | 通过，无进程残留 |

并发 batch 的 token exact 继续只作诊断；不同 DP 调度顺序下不作为确定性硬门禁。严格确定性由 30 个串行请求逐 token 覆盖。

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m4_f0_20260820_182146
```

### 5.2 失败现场与修复验证

| 现场 | 结论 |
|---|---|
| `...m4_smoke_retry_20260820_180931` | 修复 FFN 重复 key 后，prompt、batch 1/16、U2 和启动抓图通过 |
| `...m4_p1_20260820_183119` | 在线动态 capture 使用独立 stream，capture end 明确失败 |
| `...m4_p1_retry_20260820_184820` | capture 路径误等 deferred receive，启动失败 |
| `...m4_p1_retry2_20260820_185932` | 单边 capture stream 问题消失，但 A/F Graph/eager 模式不一致导致等待 |
| `...m4_p1_final_20260820_192421` | 在线 miss 整步 eager fallback 后完整通过 |

失败产物用于保留根因证据，不作为通过结果。

## 6. P1 轻量性能 guard

P1 使用 16 条预热请求和 128 条正式请求，只运行一轮：

| 指标 | M4 Graph/U2 + eager draft MTP |
|---|---:|
| 请求成功 | 128/128 |
| failed | 0 |
| 输入 token | 131,072 |
| 输出 token | 16,384 |
| output throughput | 31.473 token/s |
| output token/s/NPU | 1.967 |
| p50 TTFT | 5939.420 ms |
| p90 TTFT | 16796.600 ms |
| p50 TPOT | 1144.432 ms |
| p90 TPOT | 1464.460 ms |
| MTP acceptance rate | 84.51% |
| MTP acceptance length | 1.845 |
| Attention 峰值 HBM | 61,437 MiB |
| FFN 峰值 HBM | 45,119 MiB |
| 双 stage、log、cleanup | 通过 |

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m4_p1_final_20260820_192421
```

`performance_summary.json` 的 `passed=true` 表示请求完整性、数量、日志、HBM 采样和清理门禁通过。它不代表跨配置性能门禁通过。M4 只有一轮，且 Graph coverage、MTP 和运行期 eager fallback 比例尚未作为独立变量控制，不能与 M1/M2/M3 或 MTP-off Graph/U2 的单点直接计算收益结论。

## 7. 当前边界与下一步

M4 可冻结为 target Graph/U2 + eager draft MTP 的 functional snapshot。它不是 performance baseline。

仍未支持：

- full draft ACL Graph；
- Graph U3 和 Graph 非等量拓扑；
- A/F 非等量 + MTP；
- 多个 MTP layer 或多个 speculative token；
- PD、sequence parallel、Attention gate；
- TP/PP/CP/DCP 大于 1；
- A5 实机功能和性能验证。

按功能优先路线，下一阶段先独立设计 eager `A = k x F` + MTP 的 fan-in/fan-out 协议和 A2F1/A4F2 组件门禁；A8F4 实模 E2E 因 A3 EP4 HBM 不足转到 A5。随后再单独处理更多 speculative token。full draft Graph 和非等量 Graph 后置。

功能组合闭环后统一回到 `P8D-PERF-001`，执行 Graph/U1、Graph/U2、MTP on/off 和同预算 native Graph 的三轮 P2，并以波动、延迟、HBM、`tokens/s/NPU` 和双侧 profile 给出最终性能结论。
