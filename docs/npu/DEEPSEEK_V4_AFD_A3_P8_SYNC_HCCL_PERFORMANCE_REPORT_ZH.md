# DeepSeek-V4 AFD A3-P8 同步 HCCL 性能门禁报告

## 1. 结论

A3-P8 第一轮性能门禁未通过，当前不能冻结性能 tag，也不能宣称“开启 AFD 和 U2 microbatch 后有性能收益”。

在 vLLM 0.23 + vLLM-Ascend `rfc/vllm_cann` 目标栈上，关闭 vLLM async scheduling 后，AFD eager/U1 的 C32 单轮 output throughput 从此前三轮均值 17.082 提升到 30.615 token/s，说明 host/DP 调度是一个真实退化源。但相同配置的 eager/U2 只有 16.631 token/s，比优化后的 U1 回退 45.676%。U2 虽然正确执行了两个 stage，却没有形成通信与计算重叠，反而把同步 HCCL、MC2 和 Python stage 的调用次数近似翻倍。

本轮完成的是公平对照工具、调度开关、U1/U2 定向 profile 和失败归因。MTP-off 主门禁未通过，因此按既定顺序不启动 MTP-on 正式性能矩阵。

## 2. 固定环境和口径

| 项目 | 固定值 |
|---|---|
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| Python 环境 | `/mnt/workspace/code/.venvs/afd-v023-vllm-cann` |
| vLLM | `/mnt/workspace/code/vllm-release-v0.23.0`，`0fc695fc...` |
| vLLM-Ascend | `/mnt/workspace/code/vllm-ascend-rfc-vllm-cann`，`3da28f94...` |
| afd-plugin | `f0385c7863c9...`，分支 `feat/dsv4-afd-a3-p8-performance` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| AFD 拓扑 | A8F8，Attention DP8，FFN DP8/EP8，TP/PP/CP/DCP 均为 1 |
| 工作负载 | C32，输入 1024 token，精确输出 128 token，MTP off，eager |
| 数据面 | 标准阻塞式 `torch.distributed.send/recv`，不使用异步 HCCL |

公平资源基线使用两个相互独立的 native DP8 实例，占用 NPU0-7 和 NPU8-15。总请求按并发和请求数拆分给两个服务，结果按共同 wall-clock 窗口合并，避免把单个 8-NPU native 实例与 16-NPU A8F8 直接比较。

## 3. 工具改动

新增 `recipe/npu/P2pHcclAFDConnector/deepseek_v4/run_native_performance.py`，提供以下能力：

- 同时拉起两个不加载 afd plugin 的 native DP8 服务；
- 固定独立 device、API、DP RPC、master 和 HCCL 端口；
- 拆分负载并按真实共同测量窗口合并吞吐和延迟；
- 保存 runtime、原始 benchmark、NPU 监控、fatal log gate 和清理证据；
- 支持与 AFD runner 相同的 MTP 开关，为后续 MTP 公平对照复用。

AFD performance runner 和两侧启动脚本增加 `auto/on/off` 三态 async scheduling 控制。该开关只控制 vLLM host scheduler，不改变 HCCL send/recv 语义，也不引入异步通信。

## 4. 结果

### 4.1 同预算 native 基线

两个 native DP8 实例的三轮结果全部通过成功率、fatal 日志、CV 和 NPU 清理门禁：

| 并发 | output throughput 均值 | CV | p50 TPOT 均值 |
|---:|---:|---:|---:|
| 1 | 4.657 token/s | 1.148% | 210.416 ms |
| 8 | 34.617 token/s | 1.705% | 225.521 ms |
| 32 | 116.952 token/s | 4.058% | 247.723 ms |

证据目录：`/mnt/workspace/validation/dsv4_afd_a3_p8_native_pair_eager_mtp_off_1k128_r3_20260819_100137`。

### 4.2 host async scheduling 定位

| AFD 配置 | 调度 | 轮数 | output throughput | p50 TPOT | 说明 |
|---|---|---:|---:|---:|---|
| U1 | auto/on | 3 | 17.082 token/s | 1778.871 ms | 目标栈原参照 |
| U2 | auto/on | 3 | 12.582 token/s | 2558.760 ms | 比 U1 回退 26.342% |
| U1 | off | 1 | 30.615 token/s | 969.657 ms | 比原 U1 提升 79.228% |
| U2 | off | 1 | 16.631 token/s | 1804.228 ms | 比新的 U1 回退 45.676% |

关闭 async scheduling 后 U1 的提升远大于单轮正常波动，profile 中 Attention `Preparing` 从约 1066.969 ms 降到 1.411 ms，也消除了大量 coordinator 到达顺序告警。因此后续同步 HCCL 对照应显式使用 `--async-scheduling off`，不能继续把 `auto` 结果当作候选最优值。

但是 U2 的 P1 已超过 20% 回退门限，所以没有无意义地扩大为三轮正式矩阵。该单轮结果用于否决候选和触发 profile，不用于声明稳定性能数字。

### 4.3 U1/U2 定向 profile

profile 使用 `TORCH_PROFILER_WITH_STACK=0` 采集，并由相同 CANN 9.0.1 工具链解析。下表为 DP0 的 step 汇总：

| 角色/指标 | U1 async off | U2 async off | U2/U1 变化 |
|---|---:|---:|---:|
| Attention compute | 21.730 ms | 39.394 ms | +81.3% |
| Attention communication | 869.134 ms | 1925.313 ms | +121.5% |
| Attention bubble | 863.924 ms | 1814.378 ms | +110.0% |
| FFN compute | 788.088 ms | 1358.410 ms | +72.4% |
| FFN free | 118.672 ms | 592.080 ms | +398.9% |
| FFN stage | 907.953 ms | 1953.347 ms | +115.1% |

关键算子统计同样显示 U2 主要增加了总工作量和等待：

| FFN DP0 指标 | U1 | U2 | 变化 |
|---|---:|---:|---:|
| dispatch 调用数 | 860 | 1720 | +100% |
| dispatch 总时长 | 12426.013 ms | 22145.605 ms | +78.2% |
| combine 调用数 | 860 | 1720 | +100% |
| combine 总时长 | 3092.843 ms | 4631.533 ms | +49.8% |
| grouped matmul 调用数 | 1720 | 3440 | +100% |
| grouped matmul 总时长 | 182.203 ms | 278.553 ms | +52.9% |

U1 profile 目录：`/mnt/workspace/validation/dsv4_afd_a3_p8_hccl_u1_async_off_profile_20260819_115900`。

U2 profile 目录：`/mnt/workspace/validation/dsv4_afd_a3_p8_hccl_u2_async_off_profile_20260819_123000`。

## 5. 根因边界

当前 U2 实现创建两个 Python worker thread，但两个 microbatch 仍使用同一 NPU compute stream。connector 的 HCCL `send/recv` 是阻塞调用；已分配的 `comm_stream` 没有参与当前 stage 切换，也没有 event 建立 compute/communication 的依赖关系。因此执行效果接近“把一个 batch 拆成两半后串行执行两遍”，而不是“一个 microbatch 通信时另一个 microbatch 计算”。

这也解释了为什么半 batch 可能降低单个 dispatch/combine 的平均成本，却无法转化为端到端收益：调用数量翻倍，Attention communication/bubble 和 FFN free/stage 同时增加，节省被同步点、host 发射和跨 DP 到达偏差抵消。

目标 vLLM 0.23 与此前工作树的 `core_client.py` 负载均衡主路径基本相同，仅发现 shutdown 日志和无关 cache 字段差异。当前没有证据将 U2 回退归因于目标栈请求路由算法，也不能把 DP0 观察到的 MC2 时长简单归因于单个 MoE kernel 退化，因为 collective kernel 时长包含等待其他 EP rank 到达的时间。

## 6. 门禁判定和下一步

| 门禁 | 结果 |
|---|---|
| native 同预算三轮稳定性 | 通过 |
| AFD U1 调度退化定位 | 通过，候选配置为 async scheduling off |
| AFD U2 正确执行双 stage | 通过 |
| AFD U2 相对 AFD U1 有增益 | 未通过，回退 45.676% |
| AFD U2 相对同预算 native 有增益 | 未通过，单轮吞吐仅为 native C32 三轮均值的 14.22% |
| MTP-off P2 总门禁 | 未通过 |
| MTP-on 正式性能矩阵 | 未启动，受 MTP-off 门禁阻塞 |
| 性能 tag | 不创建 |

在“暂不考虑异步 HCCL”的约束下，本轮已经到达可证明的优化边界。继续调整 U2 threshold、加入 barrier 或重复三轮都不能建立缺失的重叠关系，只会移动或重复等待。

下一性能开发里程碑应单独立项为“标准 HCCL P2P 的 NPU 通信流重叠”，保持 `torch.distributed`/HCCL 接口不变，但为 hidden/output 收发引入专用 NPU comm stream、event 依赖和双 microbatch 状态机。该改动必须重新跑 F0、P1 和双侧 profile，先证明 trace 中存在真实 overlap，再恢复 P2 三轮、Graph 和 MTP 矩阵。若仍坚持完全阻塞式 send/recv，则应冻结同步 U1 作为功能/调试基线，并停止把 U2 作为性能候选。

## 7. P8C comm stream 跟进结果

### 7.1 实现边界

P8C 保持所有 HCCL 点对点调用为同步 `torch.distributed.send/recv`，没有使用 `isend/irecv`、后台通信线程或自定义异步 op。eager/U2 decoder 新增 Attention A2F/F2A stream、FFN receive/compute/send stream，以及逐 layer/stage event；U1、Graph U1 和 MTP 不进入该路径。

DSV4 input IDs 在 Attention ubatch 线程启动前按 stage 预传。Attention receive 使用新的 output tensor，避免 send 和原地 receive 对同一存储产生生命周期冲突；FFN 的 receive、compute 和 send buffer 均通过 `record_stream` 关联实际使用 stream。异常和 close 会清空 pending transfer、event 和 stream 引用。

### 7.2 功能门禁

以下门禁通过：

- connector 单测 41 个通过；
- connector、DBO、FFN runner 和 NPU runtime 目标回归全部通过；
- A2F1、U2、两 step、不同 token count、int32 IDs、BF16 hidden/output 和 close 组件门禁通过；
- A8F8 batch32 下 8 个 Attention rank 均为 `stage_count=2`，golden 结果一致；
- fatal、Attention 先停/FFN 后停和 NPU cleanup 通过。

主要证据：

```text
/mnt/workspace/validation/dsv4_afd_a3_comm_stream_component_a2f1_20260819_170945
/mnt/workspace/validation/dsv4_afd_a3_comm_stream_u2_batch32_20260819_173904
```

### 7.3 P1 性能结果

工作负载与旧 U1/U2 保持一致：A8F8、eager、async scheduling off、C32、输入 1024、精确输出 128、128 请求、MTP off。

| 配置 | output throughput | p50 TPOT | 相对 U1 |
|---|---:|---:|---:|
| U1 async off | 30.615 token/s | 969.657 ms | 基线 |
| 旧同步 U2 async off | 16.631 token/s | 1804.228 ms | -45.676% |
| P8C comm-stream U2 | 14.961 token/s | 2128.415 ms | -51.133% |

P8C 相对旧 U2 也回退 10.043%。该轮 128/128 成功，双 stage、fatal、shutdown、监控和 cleanup 门禁全部通过，因此结果是有效的性能否决，不是功能失败或残留进程造成的失真。

证据目录：`/mnt/workspace/validation/dsv4_afd_a3_comm_stream_u2_p1_c32_1k128_20260819_174750`。

### 7.4 双侧 profile

profile 使用 `TORCH_PROFILER_WITH_STACK=0`，Attention DP0 和 FFN DP0 各采 20 个 step，并显式由固定 CANN 9.0.1 路径解析。

| 角色/指标 | 旧 U2 | P8C U2 | 变化 |
|---|---:|---:|---:|
| Attention compute | 39.394 ms | 39.232 ms | -0.4% |
| Attention communication not overlapped | 1925.313 ms | 1669.388 ms | -13.3% |
| Attention overlapped | 0.000 ms | 35.551 ms | 新增真实 overlap |
| Attention bubble | 1814.378 ms | 1187.816 ms | -34.5% |
| FFN compute | 1358.410 ms | 682.618 ms | -49.7% |
| FFN overlapped | 0.000 ms | 13.557 ms | 新增真实 overlap |
| FFN free | 592.080 ms | 949.095 ms | +60.3% |
| FFN stage | 1953.347 ms | 1624.974 ms | -16.8% |

`Overlapped` 非零且 Attention 几乎覆盖了每 step 的 39 ms 计算，证明 stream/event 已在设备侧生效。kernel 明细中 FFN 计算由旧默认 stream 46 移至专用 stream 43；HCCL AICPU task 继续落在 process-group 内部 stream 9/171，符合底层仍使用标准 `send/recv` 的设计。

但 Attention 每 step 仍有约 1.67 秒未重叠通信，FFN free 又增至约 0.95 秒。新增 overlap 被 host DBO 交接、跨 DP 到达偏斜和 FFN 等待抵消，所以不能从“trace 有 overlap”推导“端到端有收益”。

profile 目录：`/mnt/workspace/validation/dsv4_afd_a3_comm_stream_u2_profile_20260819_181639`。

### 7.5 判定和下一步

P8C 实现目标“同步 HCCL API 下建立 NPU stream overlap”已经完成，但性能目标未通过；不运行三轮 P2、不恢复 MTP-on 矩阵、不创建性能 tag。

下一步保持同步 `send/recv`，将 Attention 的两个 DBO Python 线程和 `dbo_yield` 交接替换为 afd-plugin 内单线程 layer-major U2 入队，使 host 顺序与 FFN 的 `layer -> stage` loop 一致。该建议已由 P8D 实施，结果见第 8 章。

## 8. P8D 单线程 layer-major U2 跟进结果

### 8.1 实现边界

P8D 只改变 DeepSeek-V4、`P2pHcclAFDConnector`、eager/U2、MTP off 的 host 提交顺序。Attention 不再为两个 stage 创建 Python worker thread，也不再通过 `dbo_yield` 交接；一个插件内 host loop 按 `layer -> stage` 顺序推进两个 stage。

connector 在 F2A receive stream 上记录完成 event，并把 compute-stream wait 延迟到同一 stage 进入下一层之前。DSV4 decoder layer 拆成“Attention/HC pre/remote MoE”和“等待后 HC post”两段，避免未满足 F2A 依赖时消费 output。异常路径清空延迟依赖和 stage 状态。U1、Graph、MTP、其他 connector 和原有两线程 wrapper 路径不变。

数据面没有变化：每个传输仍调用同步 `torch.distributed.send/recv`；没有使用 `isend/irecv`、后台通信线程、自定义 CAMP2P op 或异步 HCCL op。

### 8.2 功能门禁

以下门禁通过：

- connector、DSV4 构造/单层等价、DBO、FFN runner 和 NPU runtime 目标回归；
- A2F1、两个 stage、两个 step、不同 token count、int32 IDs、BF16 hidden/output 和 close 组件测试；
- A8F8 batch32 下 8 个 Attention rank 均记录 `stage_count=2`；
- 固定栈 golden 请求逐 token 一致，fatal、shutdown 和 NPU cleanup 通过。

这是 P8D 候选的定向功能门禁，不是重新执行完整 30/30 golden 与 batch 1/8/32 生命周期矩阵。由于随后 P1 已否决性能候选，本轮没有扩大正确性运行；已有 U1、Graph 和 MTP 功能基线不因该结果重新冻结。

首次 A8F8 验证发现 HC post 在 F2A event wait 前消费 output，触发 tensor identity fail-fast；拆分 decoder layer 后问题修复。失败产物保留在 `/mnt/workspace/validation/dsv4_afd_a3_layer_major_u2_batch32_20260819_193218`，不得把该目录计入成功门禁。

成功证据：

```text
/mnt/workspace/validation/dsv4_afd_a3_layer_major_component_a2f1_20260819_193218
/mnt/workspace/validation/dsv4_afd_a3_layer_major_u2_batch32_20260819_194535
```

### 8.3 P1 性能结果

工作负载固定为 A8F8、eager、async scheduling off、C32、输入 1024、精确输出 128、128 请求、MTP off。

| 配置 | output throughput | p50 TTFT | p50 TPOT | 相对 U1 |
|---|---:|---:|---:|---:|
| U1 async off | 30.615 token/s | - | 969.657 ms | 基线 |
| 旧同步 U2 | 16.631 token/s | - | 1804.228 ms | -45.676% |
| P8C 双线程 comm stream U2 | 14.961 token/s | 13996.645 ms | 2128.415 ms | -51.133% |
| P8D 单线程 layer-major U2 | 16.472 token/s | 11587.963 ms | 1960.470 ms | -46.197% |

P8D 相对 P8C 吞吐提升 10.099%，p50 TTFT 改善 17.209%，p50 TPOT 改善 7.891%；相对旧同步 U2 仍低 0.958%。该轮 128/128 请求成功，双 stage、fatal、shutdown、监控和 cleanup 全部通过，因此它证明删除线程/GIL 交接有收益，但仍未达到 U1，更没有达到同预算 native。

证据目录：`/mnt/workspace/validation/dsv4_afd_a3_layer_major_u2_p1_c32_1k128_20260819_195445`。

### 8.4 双侧 profile

profile 继续使用 `TORCH_PROFILER_WITH_STACK=0`，Attention DP0 和 FFN DP0 各采 20 个 step，并由固定 CANN 9.0.1 在宿主环境串行解析。profile 下的 16.531 token/s 只用于定位，不纳入正式吞吐比较。

| 角色/指标 | P8C | P8D | 变化 |
|---|---:|---:|---:|
| Attention compute | 39.232 ms | 39.385 ms | +0.4% |
| Attention communication not overlapped | 1669.388 ms | 1398.570 ms | -16.2% |
| Attention overlapped | 35.551 ms | 35.690 ms | +0.4% |
| Attention bubble | 1187.816 ms | 1430.170 ms | +20.4% |
| FFN compute | 682.618 ms | 422.890 ms | -38.0% |
| FFN overlapped | 13.557 ms | 10.931 ms | -19.4% |
| FFN free | 949.095 ms | 1273.094 ms | +34.1% |
| FFN stage | 1624.974 ms | 1692.277 ms | +4.1% |

两份 trace 的 kernel 数量一致：Attention 每侧有 1760 个 send 和 1720 个 receive，FFN 为 1720 个 send 和 1760 个 receive，说明变化不是少执行了模型层。Attention HCCL send 总时长从 16784.926 ms 降到 1392.634 ms，但 receive 总时长从 23756.319 ms 增至 28603.394 ms；等待被重新分配到 F2A receive，而没有从关键路径消失。FFN 计算总时长下降，但 free 上升，表明跨侧/跨 DP 推进仍不均衡。

profile 目录：`/mnt/workspace/validation/dsv4_afd_a3_layer_major_u2_profile_20260819_202039`。

### 8.5 判定和下一步

P8D 的定向功能目标和“消除双 Python 线程交接”目标已完成，P1 相对 P8C 有改善；性能总门禁仍未通过，所以不补跑完整 30/30 功能矩阵、不运行三轮 P2、不启动 MTP-on 性能矩阵、不创建性能 tag。

下一候选应保持同步 HCCL API，优先减少每 layer/stage 的同步消息和 event wait 数量，或把多个连续层可安全合并的控制/收发推进成更粗粒度状态机，并结合 DP0-7 到达时间定位最慢 rank。任何 batching 都必须保持 IDs/hidden/output 顺序、精确 shape 和异常清理语义。只有新的 P1 至少达到 U1 后才恢复三轮；单纯扫描 DBO threshold、重复当前点或把 profiler 吞吐当正式结果均没有意义。

### 8.6 已知问题登记

| 问题 ID | 状态 | 现象 | 当前归因 | 关闭条件 |
|---|---|---|---|---|
| `P8D-PERF-001` | Open，暂缓到功能扩展完成后统一优化 | P8D C32 为 16.472 token/s，相对 U1 回退 46.197% | Attention 等待转移到 F2A receive；FFN free 为 1273.094 ms，跨侧/跨 DP 推进不均衡 | 同口径 P1 不低于 U1；随后 P2 三轮通过重复性、延迟和同预算 native 收益门禁 |

冻结 tag 只能标记为 functional snapshot，不能包含 `performance-baseline` 或暗示 P2 已通过。后续 Graph/U2、MTP/U2 等功能提交必须继续引用 `P8D-PERF-001`，不得用新功能的单轮数据覆盖本问题；统一性能优化恢复时，先重跑 DP0-7 到达时间定位，再建立新候选 A/B。
