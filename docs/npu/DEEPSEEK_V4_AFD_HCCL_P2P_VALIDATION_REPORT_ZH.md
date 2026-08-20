# DeepSeek-V4 AFD 标准 HCCL P2P Connector 实现与验证报告

> 状态说明（2026-08-18）：本文是首版 eager/U1、U2 验证快照。后续等量 A/F、`FULL_DECODE_ONLY`、U1 的 Graph 功能已经完成，当前状态与证据见 `DEEPSEEK_V4_AFD_HCCL_P2P_GRAPH_U1_VALIDATION_REPORT_ZH.md`；本文其余 eager 实验数据保持原样。

## 1. 结论

`P2pHcclAFDConnector` 已在固定 A3 运行栈上完成首版实现和 eager 正确性闭环：

- 数据面仅使用 `torch.distributed.send/recv` + HCCL；
- 不加载 afd-plugin CAMP2P 自定义算子环境；
- 不调用 `torch.ops.vllm.afd_camp2p_send_attn_output()`、`afd_ascend.a2e` 或 `afd_ascend.e2a`；
- A1F1、U2、连续两个 step 的 int32 IDs、BF16 hidden/output round-trip 通过；
- A8F8 eager/U1 和 eager/U2 均为 30/30 golden 请求逐 token 一致；
- batch 1/8/32、U2 双 stage 观测、Attention 先停、两侧零返回码和 NPU 清理通过。

本阶段证明标准 HCCL P2P 数据面可以正确承载 DeepSeek-V4 AFD，并且 U2 调度不会互锁。它还不证明已经获得性能收益；下一阶段必须完成同口径 U1/U2 profiling 和公平吞吐对照。

## 2. 背景与目标

此前 DeepSeek-V4 AFD 使用 `CAMP2pAFDConnector`，hidden/output 数据面依赖 afd-plugin 的 A2E/E2A 自定义算子。新的产品目标要求：

1. 在 HCCL send/receive 接口下获得 AF 分离性能收益；
2. A5 继续使用这套公共接口，不把移植 A3 专用 kernel 作为主线前置条件；
3. CAMP2P 继续保留为已冻结基线，不改变其既有行为。

因此新增独立 `P2pHcclAFDConnector`。两种数据面拥有不同的初始化资源、阻塞语义和 DBO 调度顺序，独立 connector 比在 CAMP2P 内增加条件分支更容易验证、回退和做公平性能归因。

## 3. 关键实现

### 3.1 通信拓扑

首版延续已验证的等量一一映射：

```text
Attention rank i  <->  FFN rank i

每个 U stage：
  1 个 hidden/output HCCL group
  1 个 input IDs HCCL group（DSV4）

所有 stage：
  1 个 Gloo DP metadata 控制 group
```

当前显式拒绝 A/F 非等量。该限制来自首版消息所有权和一一映射，不是 HCCL 或 A3/A5 硬件协议的永久限制。

### 3.2 数据与控制顺序

每层、每 stage 的数据顺序为：

```text
layer 0:
  Attention -- int32 input IDs --> FFN
  Attention -- BF16 hidden -----> FFN
  Attention <-- BF16 output ----- FFN

layer 1/2:
  FFN 复用对应 stage 的 IDs cache

layer 3 起:
  不再传递 IDs，step 结束或异常时清空 cache
```

Gloo 控制面先发送各 DP/stage token count。FFN 据此预分配接收 buffer，再投递 HCCL receive，避免数据面依赖固定最大 shape。

### 3.3 阻塞式 HCCL 的 U2 调度

标准 `dist.send/recv` 在当前栈上按阻塞式语义工作。原 CAMP2P 调度在 Attention send 后立即切换 stage，会形成以下交叉等待：

```text
Attention stage 1: send hidden
FFN stage 0:       send output
```

双方同时阻塞 send 时会互锁。为此公共 connector 契约增加 `yield_after_attn_send` 能力：

- CAMP2P 等原有 connector 保持默认值 `True`，行为不变；
- HCCL P2P 设置为 `False`；
- HCCL Attention 在收到当前 stage 的 FFN output 后再执行 DBO yield。

这样 Attention 的 stage 切换与 FFN 的 layer-major 顺序一致：stage 0 完成当前层 round-trip 后，再让 stage 1 进入当前层。

### 3.4 配置、recipe 与 fail-fast

- factory 注册 `P2pHcclAFDConnector`，保持现有 `--additional-config` 格式；
- DSV4 HCCL 首版允许 eager/U1 和 eager/U2；
- 显式拒绝 Graph、A/F 非等量及 DSV4 已有的未验证组合；
- HCCL recipe 复用同一验证器，但不会 source afd-plugin CAMP2P 自定义 op 环境；
- runtime manifest 记录 connector、U1/U2、threshold、固定 commit 和运行栈。

## 4. 固定验证环境

| 项目 | 固定值 |
|---|---|
| 日期 | `2026-08-13` |
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| venv | `/mnt/workspace/code/.venvs/afd-v026` |
| vLLM | `568afb3a13806beb53bb2e6bd518269357b237c0` |
| vLLM-Ascend | `80d8c194f7584b17fe08065ea99a130916f6b0e7` |
| 开发起点 | afd-plugin `1b5d011c830d66a2516ed647064fa571667761a3` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| 拓扑 | Attention NPU 0-7，FFN NPU 8-15，A8F8 |
| 并行 | DP8、TP1、EP8、PP1、CP1、DCP1 |
| 执行 | eager，MTP/Graph/PD/sequence parallel 关闭 |

## 5. 验证结果

### 5.1 HCCL 组件

产物：

`/mnt/workspace/validation/dsv4_afd_hccl_p2p_component_fix_20260813/roundtrip.json`

使用物理 NPU 2 和 10，覆盖 U2 两个 stage、连续两个 step、不同 token count。Attention 和 FFN 两个子进程返回码均为 0，4 组 IDs/hidden/output round-trip 全部通过。

### 5.2 A8F8 eager/U1

产物：

`/mnt/workspace/validation/dsv4_afd_hccl_p2p_u1_correctness_20260813`

| 门禁 | 结果 |
|---|---|
| 10 prompts x 3 rounds | 30/30，0 mismatch |
| batch 1/8/32 | 全部 `valid=True` |
| fatal log marker | 两侧为空 |
| shutdown | Attention 先停，两侧返回码 0 |
| NPU cleanup | 通过，无残留 PID |
| 冷启动 | 384.111 s |

### 5.3 A8F8 eager/U2

产物：

`/mnt/workspace/validation/dsv4_afd_hccl_p2p_u2_correctness_20260813`

| 门禁 | 结果 |
|---|---|
| 10 prompts x 3 rounds | 30/30，0 mismatch |
| batch 1/8/32 | 全部 `valid=True` |
| U2 runtime evidence | stage 0/1 均被观测，门禁通过 |
| fatal log marker | 两侧为空 |
| shutdown | Attention 先停，两侧返回码 0 |
| NPU cleanup | 通过，无残留 PID |
| 冷启动 | 386.100 s |

batch 的 `token_exact_count` 是诊断项，不是硬门禁。并发请求的 DP 调度顺序可能改变，因此 batch 门禁检查 prompt IDs、choice 数、输出长度/类型和请求完整性；temperature 0 的确定性由 30 个串行请求逐 token 0 mismatch 保证。

## 6. 测试覆盖

CPU/Mock 和 recipe 回归覆盖：

- factory/模块注册和 CPU-safe import；
- 等量拓扑门禁与 eager-only 门禁；
- IDs 先于 hidden、每 stage 独立 group、layer 0 单次传递；
- `-1` padding、词表上下界和 buffer 上限；
- FFN token count/buffer 准备与 output 回传 rank；
- HCCL connector 不含 CAMP2P 自定义 op 引用；
- U2 阻塞 connector 的 yield 顺序；
- group 销毁、buffer/cache 清理；
- recipe 选择、runtime manifest 和日志/退出/NPU 清理门禁。

## 7. 当前边界

本报告不宣称以下能力已经完成：

- AFD 相对非 AFD 或 HCCL U2 相对 U1 的性能收益；
- 本报告当时尚未完成 Graph/`FULL_DECODE_ONLY`；后续等量 A/F 的 Graph/U1 已完成，Graph/U2、U3 和 Graph 非等量仍未支持；
- A/F 非等量、TP/PP/CP/DCP 大于 1；
- Attention-side gate、MTP、PD、sequence parallel；
- A5 实机支持。

## 8. 下一阶段

下一阶段只做 A3 HCCL 性能闭环：

1. 固定 prompt 输入/输出长度、concurrency、预热和测量窗口；
2. 同一 HCCL connector 对比 eager/U1 与 eager/U2，至少 3 轮；
3. 采集 Attention/FFN role-local DP0 profile，设置 `TORCH_PROFILER_WITH_STACK=0`；
4. 统计 output tokens/s、tokens/s/NPU、TTFT、TPOT p50/p90/p99、HBM、利用率、A2F/F2A 与 bubble；
5. 扫描 U2 threshold 与 HCCL buffer，收益必须超过运行波动；
6. 再与相同 16-NPU 总预算的非 AFD 实例做资源效率对照。

A5 到位后，先建立独立运行栈并重跑 HCCL BF16/int32 组件测试；再按 U1、U2、生命周期、profile 和性能顺序重新验收。公共 HCCL 接口减少了自定义 kernel 移植工作，但不能替代 A5 实机验证。
