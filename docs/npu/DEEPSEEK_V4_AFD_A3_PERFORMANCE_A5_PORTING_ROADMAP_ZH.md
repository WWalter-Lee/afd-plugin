# DeepSeek-V4 AFD A3 性能验收与 A5 适配路线

## 1. 文档定位

本文用于固化 DeepSeek-V4 AFD 在完成 eager/U1 与 Graph/U1 正确性基线后的目标、开发顺序和验收门禁，供后续开发、验证、性能分析和 A5 迁移时直接使用。

文档状态：`2026-08-13`，eager/U2 已实现，A8F8 正确性、30 分钟空闲恢复和清理门禁已通过，等待提交与 tag 冻结。

本文不替代以下两份文档：

- `DEEPSEEK_V4_AFD_ADAPTATION_GUIDE_ZH.md`：完整适配背景和早期里程碑；
- `DEEPSEEK_V4_AFD_BASELINE_TAGS_SUMMARY_ZH.md`：两个已冻结 tag 的关键改动、原因和意义。

本文回答后续最关键的四个问题：

1. 当前只有 A3 环境时，哪些工作可以继续完成；
2. 如何证明开启 AFD 后有真实性能收益；
3. 为最终支持 A5，现在的代码需要保持哪些可迁移边界；
4. A5 到位后，还必须完成哪些硬件相关适配和重新验收。

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
  U1 correctness 已完成
  -> eager U2（实现、30/30 golden、batch、二次启动、空闲恢复和 profile 已通过）
  -> 独立提交和冻结 tag
  -> eager U2 profiling/tuning
  -> U2 + FULL_DECODE_ONLY
  -> A3 公平性能验收
  -> 冻结 A3 性能基线

A5 硬件到位后
  平台审计和独立运行栈
  -> A2E/E2A 自定义算子适配
  -> 组件/单层验证
  -> U1/U2、eager/Graph 回归
  -> A5 重新调优和独立性能验收
```

A3 验收通过只说明实现语义和 A3 性能成立，不等于 A5 已支持，也不能将 A3 性能数字直接外推到 A5。

## 3. 已冻结基线

### 3.1 Tag 和定位

| Tag | Commit | 定位 |
|---|---|---|
| `dsv4-afd-eager-u1-v1` | `40981475a9270c9b79ebf5cfe46d375472ee0a06` | A8F8、eager、U1 正确性基线 |
| `dsv4-afd-graph-u1-v1` | `2ed98442351d4be96edbb315a6b6c8d00805bbc4` | A8F8、`FULL_DECODE_ONLY`、U1 Graph 与生命周期基线 |

后续阶段从 `dsv4-afd-graph-u1-v1` 开新分支，不改写以上 tag。每个阶段通过全部门禁后再打新 tag，失败阶段不打 tag。

### 3.2 A3 固定运行栈

| 项目 | 固定值 |
|---|---|
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| Python venv | `/mnt/workspace/code/.venvs/afd-v026` |
| vLLM | `568afb3a13806beb53bb2e6bd518269357b237c0` |
| vLLM-Ascend | `80d8c194f7584b17fe08065ea99a130916f6b0e7` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| 插件 | `ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd` |
| 拓扑 | Attention NPU 0-7，FFN NPU 8-15 |
| 并行 | DP8、TP1、EP8、PP1、CP1、DCP1 |
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

DeepSeek-V4 当前复用已有的 `CAMP2pAFDConnector`，没有新增 DSV4 专用 connector。

DSV4 的改动是在现有 connector 中扩展：

- `AFDA2FTransferPayload.input_ids`；
- 每个 stage 独立的 `afd_ids` HCCL group；
- 预分配的 NPU `int32` IDs buffer；
- Attention 到 FFN 的一次性 IDs side channel；
- FFN hash layer 的 stage 级 IDs cache 生命周期。

后续 U2 仍应扩展这条路径，不应为了 U2 再复制一套 connector。

### 4.2 A/F 数量相等的含义

当前 DSV4 适配显式要求：

```python
num_attention_ranks == num_ffn_ranks
```

这个门禁位于 `afd_plugin/compat/npu/feature_validation.py`。原因是当前 IDs side channel 和 CAMP2P 数据面采用 Attention rank `i` 与 FFN rank `i` 一一配对，并按对应 stage 使用相同消息顺序。

因此：

- A/F 等量是当前 DSV4 实现和已验证拓扑的要求；
- 它不是 CAMP2P 的抽象定义，也不是 A3/A5 硬件的永久要求；
- 若将来支持 A/F 非等量，必须先定义 IDs 的 fan-out/gather 映射、消息计数、buffer 所有权和 Graph 稳定地址，再单独验收；
- 当前性能路线不并行引入非等量拓扑，避免把正确性、U2 和拓扑变化混在一个阶段。

### 4.3 当前未验证能力

以下能力仍不属于已冻结基线：

- eager DBO/U2（实现与全部正确性门禁已通过，尚未完成 tag 冻结）；
- U2 + `FULL_DECODE_ONLY`；
- Attention 侧 gate；
- MTP/speculative decoding；
- Mooncake PD；
- sequence parallel；
- A/F 非等量；
- TP、PP、CP 或 DCP 大于 1；
- A5 自定义 A2E/E2A 算子。

## 5. A3 后续开发阶段

每阶段独立提交、独立验证。上一阶段未通过时，不进入下一阶段。

| 阶段 | 主要交付 | 进入下一阶段的门禁 |
|---|---|---|
| A3-P0 | 固定非 AFD、AFD eager/U1、AFD Graph/U1 的性能实验协议 | 三种部署使用同一模型、请求和统计口径；结果可重复 |
| A3-P1 | DSV4 eager/U2 的 stage IDs 与执行语义 | 已通过：相关回归 116 项，真实双 stage 执行通过 |
| A3-P2 | eager/U2 A8F8 E2E | 已通过：golden、batch、双冷启动、30 分钟空闲恢复、profile 和清理通过 |
| A3-P3 | eager/U2 profiling 与调优 | 形成 threshold、`aiv_num` 和热点结论；收益超过波动 |
| A3-P4 | U2 + `FULL_DECODE_ONLY` | U1 fallback、U2 capture/replay、动态 IDs 和生命周期通过 |
| A3-P5 | A3 公平性能验收 | 正确性硬门禁通过；性能和资源效率达到预先锁定阈值 |
| A3-P6 | 冻结 A3 性能基线 | tag、报告、原始日志、trace 和清理证据齐全 |

### 5.1 A3-P0：固定性能实验协议

先锁定实验协议，再开始调优。至少固定：

- prompt 集合和输入/输出长度；
- batch/concurrency 阶梯，至少保留 batch 1/8/32；
- 请求到达方式和预热请求数；
- seed、temperature 和最大输出 token 数；
- HBM 利用率、最大序列数和最大 batched tokens；
- U2 threshold、`aiv_num` 和 Graph capture size；
- 每个点至少 3 轮稳定运行；
- 冷启动数据与稳态数据分开报告；
- 服务端吞吐、客户端延迟和 NPU 资源指标使用相同测量窗口。

必须保留以下三类初始对照：

1. 非 AFD eager/Graph 基线；
2. AFD eager/U1；
3. AFD Graph/U1。

后续只逐项加入 eager/U2 和 Graph/U2，避免一次改变多个变量。

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

实现时不硬编码 NPU 0-15 或 A8F8。角色数、rank、stage 数、token slice 和 `aiv_num` 继续来自配置或运行上下文，为 A5 不同卡数保留空间。

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

CAMP2P 组件测试至少覆盖：

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

### 5.5 A3-P3：profiling 和调优

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
- ubatch split、wait 和 merge。

需要统计：

- Attention/FFN 计算时间；
- A2F、F2A 延迟及未重叠部分；
- Attention/FFN overlap；
- FFN wait、free/bubble；
- stage 0/1 不均衡程度；
- host `.item()`、同步 memcpy 和控制面开销；
- 每侧峰值 HBM、NPU 利用率和启动时间。

第一轮调优矩阵可以从以下值开始，但它们不是产品固定值：

```text
U2 threshold: 16, 32, 48, 64, 96, 128
aiv_num:      结合 A3 有效核数做小范围扫描
```

性能热路径中的高频 warning、tensor `.tolist()` 和仅用于调试的 key 构造应降到 debug 或显式开关下。任何日志调整都必须保留错误和生命周期门禁所需信息。

### 5.6 A3-P4：U2 + FULL_DECODE_ONLY

只有 eager/U2 正确且 profile 已解释后，才进入 Graph/U2。

本阶段重点：

- U1 fallback 和 U2 使用不同且稳定的 graph key；
- stage 0/1 使用稳定地址的独立 IDs/hidden buffer；
- capture 期间和 replay 期间消息数量一致；
- 每次 replay 使用当前 step 的 IDs，不能固化首次 capture 的 IDs；
- 不同 token count、capture size 和 U1/U2 切换不串图；
- graph capture 失败、replay 异常和 shutdown 均能清理；
- 冷启动、二次启动和空闲恢复继续通过。

通过后建议冻结：

```text
dsv4-afd-a3-graph-u2-v1
```

## 6. A3 性能验收方法

### 6.1 比较矩阵

最终至少比较：

| 方案 | 作用 |
|---|---|
| 非 AFD eager | 基础执行对照 |
| 非 AFD Graph | 非 AFD 最佳稳态对照 |
| AFD eager/U1 | 分离本身的成本和收益 |
| AFD Graph/U1 | 当前已冻结执行基线 |
| AFD eager/U2 | 双阶段重叠收益 |
| AFD Graph/U2 | A3 当前候选交付形态 |

### 6.2 公平资源口径

A8F8 使用 16 个 NPU。只把它与一个 8-NPU 非 AFD 实例比较，会把资源翻倍带来的收益误认为 AFD 架构收益。因此必须同时报告两种口径：

| 对比 | 说明 |
|---|---|
| AFD A8F8 vs 非 AFD 单个 8-NPU 实例 | 反映单服务扩展能力和延迟变化 |
| AFD A8F8 vs 两个非 AFD 8-NPU 实例的总和 | 反映相同 16-NPU 总预算下的资源效率 |

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
2. Graph/U2 相对 Graph/U1 的 output tokens/s 提升不低于 10%；
3. p99 TPOT 相对选定基线的回退不超过 5%；
4. AFD 相对同总 NPU 预算非 AFD 的收益必须大于 3 轮稳定运行的波动区间；
5. 不允许通过减少输出 token、改变 batch、降低 golden 覆盖或放宽错误检查获得收益；
6. 若吞吐增加但 `tokens/s/NPU` 明显下降，必须明确记录为扩容收益，不能表述为资源效率收益。

正式跑数前应把最终阈值写入验证脚本或实验 manifest，跑完后不根据结果反向调整门禁。

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
dsv4-afd-a3-graph-u2-perf-v1
```

## 7. 面向 A5：现在就要保持的设计边界

本文暂按“A5 指 Atlas A5 / Ascend 950 系列”规划。实际服务器 SKU、SoC 字符串和单机 NPU 数以目标机器审计结果为准。

### 7.1 已知的软件基础和当前缺口

固定 vLLM-Ascend commit 已包含：

- `A5DeviceAdaptor`；
- `SOC_VERSION` 以 `ascend950` 开头时的构建识别；
- 多个 Attention/MoE 算子的 `ascend950` 注册。

但这只说明固定上游存在 A5 基础，不说明 afd-plugin 已支持 A5。当前 AFD 自定义 A2E/E2A 仍是 A3/910C 专用：

- `csrc/npu/build_aclnn.sh` 只接受 `910c`/`ascend910_93*`；
- `a2e_def.cpp` 和 `e2a_def.cpp` 只注册 `ascend910_93`；
- `tools/dsv4/install_plugin.sh` 默认 `SOC_VERSION=ascend910_9362`；
- kernel 中存在 192 KB UB、48 core、window offset 等 A3 假设；
- ACLNN host 侧 HCCL server type 对 910B 和其他 SoC 走不同分支，A5 行为尚未实测。

因此不能仅增加一个 `ascend950` 字符串就宣称 A5 支持。

### 7.2 A3 开发期间必须做到

- U2 stage、rank 和 token slice 逻辑保持硬件无关；
- 不在模型/connector 核心路径硬编码 A8F8、NPU 0-15 或单机卡数；
- `aiv_num`、role 数、buffer token 上限和 capture size 可配置；
- SoC 不支持时显式 fail-fast，不能静默跳过自定义算子后继续启动；
- A3/A5 使用独立 venv、CANN 根目录、构建输出、启动 recipe 和验证目录；
- CPU/Mock 测试不依赖 A3 核数或物理 device ordinal；
- performance manifest 记录 SoC、驱动、固件、CANN、torch-npu、拓扑和 NUMA；
- 上游 patch 继续标注固定 commit 和 AFD patch marker，便于 A5 栈变化时重放差异。

### 7.3 当前不要提前做的 A5 修改

没有 A5 硬件和匹配工具链时，不应凭 A3 结果猜测并提交以下变更：

- A2E/E2A kernel 的 UB 分配和核数；
- HCCL server type 和通信 window 选择；
- A5 特定 IPC/SDMA 路径；
- `aiv_num`、U2 threshold、Graph capture size；
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

### 8.2 A5-H1：自定义算子构建和注册

需要完成：

1. 在 `build_aclnn.sh` 增加精确的 `ascend950*` 识别和构建目标；
2. 为 A2E/E2A host definition 增加正确的 `ascend950` 配置；
3. 使用 A5 工具链重新编译，禁止复用 A3 二进制；
4. `ensure_afd_ascend_ops_loaded()` 在 A5 venv 中通过；
5. unsupported SoC、缺少 ops 包或产物架构不匹配时明确失败。

### 8.3 A5-H2：kernel 和通信最小验证

按由小到大的顺序验证：

1. A1F1 IDs `int32` round-trip；
2. A1F1 hidden A2E/E2A round-trip；
3. 不同 token count、`-1` padding 和 buffer 边界；
4. 连续两个 step 和两个 stage；
5. 多 rank 单机；
6. 若产品拓扑跨机，再增加跨节点 HCCL 测试；
7. 异常取消、connector close 和重新启动。

本阶段重点实测：

- UB 大小和对齐；
- 可用 AIV/core 数；
- HCCL server type；
- 通信 window/offset；
- IPC/SDMA/MTE 行为；
- rank 到物理 NPU/NIC 的映射。

### 8.4 A5-H3：模型正确性回归

A5 必须先生成同平台非 AFD golden，不能只拿 A3 token 文件代替 A5 基线。随后依次验证：

1. Attention/FFN 角色构造和权重所有权；
2. layer 0、2、3、42 的单层/loopback 等价；
3. eager/U1；
4. `FULL_DECODE_ONLY`/U1；
5. eager/U2；
6. `FULL_DECODE_ONLY`/U2；
7. 冷启动、二次启动、batch、空闲恢复和严格关闭。

若 A5 单机有 16 个 NPU，可以验证 A8F8；若只有 8 个 NPU，当前等量门禁下的起始拓扑是 A4F4。实际角色映射必须根据 `npu-smi` 拓扑和 NUMA/NIC 关系决定，不能只按 device ordinal 对半切分。

### 8.5 A5-H4：重新调优和性能验收

A5 需要重新扫描：

- `aiv_num`；
- U2 threshold；
- Graph capture size；
- HCCL buffer；
- Attention/FFN 角色数；
- CPU/NUMA 绑核；
- 单机/跨机 rank 布局。

profiling 仍遵守：部署、采集、解析分离；`TORCH_PROFILER_WITH_STACK=0`；Attention/FFN role-local DP0；使用与 A5 采集环境匹配的 CANN parser。

A5 使用与 A3 相同的公平资源口径，但重新生成全部数字和门禁结论。通过后再创建带 A5 标识的独立 tag，不能复用 A3 性能 tag 代表 A5。

## 9. PD 集成顺序

Mooncake PD 不进入当前 A3 standalone AF 性能开发的关键路径。

进入 PD 的门禁：

- A3 或目标 A5 上 standalone AF 的 U2 + Graph 正确性通过；
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
| A3 Graph/U2 基线 | `dsv4-afd-a3-graph-u2-v1` |
| A3 性能验收 | `dsv4-afd-a3-graph-u2-perf-v1` |
| A5 基线 | 在实际硬件和版本确认后使用 `dsv4-afd-a5-*` 命名 |

每个 tag 应为 annotated tag，tag message 至少包含：

- 基线用途；
- 精确 commit；
- SoC/拓扑；
- CANN、vLLM、vLLM-Ascend；
- U1/U2、eager/Graph；
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
- Graph replay 使用旧 IDs；
- 出现 NaN/Inf、shape 或 dtype 不一致；
- Attention/FFN 任一侧非 0 退出；
- 冷启动后存在残留进程、端口或 NPU 占用；
- CANN 路径混入其他版本；
- profiler 采集栈和 parser 版本不匹配；
- 性能收益只在单轮出现，或小于稳定运行波动；
- A5 上只完成编译，没有完成算子、模型和 E2E 验证。

## 12. 后续恢复工作时的最短检查清单

1. 确认当前分支、HEAD、两个已冻结 tag 和 worktree 状态；
2. 激活 A3 固定运行栈并运行 `tools/dsv4/check_runtime.sh`；
3. 确认最近一个已通过阶段及其 `validation_summary.json`；
4. 当前下一项是提交 eager/U2 并冻结 tag；
5. 随后固定非 AFD/U1/U2 的公平请求矩阵和性能门禁；
6. 先完成同口径 profiling 与 threshold 扫描，再判断是否进入 Graph/U2；
7. 正式性能跑数前锁定公平对照和门禁；
8. A3 性能 tag 冻结后再进入 PD 或 A5 硬件差异开发；
9. A5 到位后从硬件审计和独立工具链开始，不复用 A3 二进制；
10. 每次阶段完成都保存日志、原始数据、解析结果和清理证据。

## 13. 一句话路线

先在 A3 上把 U2、Graph 和公平性能收益做实，同时保持拓扑与硬件参数可配置；再在 A5 上重新构建通信算子、验证硬件假设、重跑全部正确性和性能门禁，最终形成独立的 A5 交付基线。
