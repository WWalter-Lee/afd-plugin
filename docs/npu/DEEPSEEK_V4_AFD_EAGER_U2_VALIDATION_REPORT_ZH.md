# DeepSeek-V4 AFD A3 eager/U2 开发与验证报告

## 1. 结论

本阶段在固定 A3 运行栈上完成了 DeepSeek-V4 AFD eager/U2 的协议实现、回归测试、A8F8 正确性验证和 role-local DP0 profiling 采集。Graph/U2 仍显式拒绝，性能收益尚未验收。

当前已确认：

- A8F8、DP8/TP1/EP8 下真实执行两个 ubatch stage；
- Milestone 0 的 10 条 golden prompt 连续 3 轮，共 30/30 请求逐 token 一致；
- batch 1/8/32 均返回合法结果，非均匀 DP token count 的 batch 32 不再死锁；
- 两次独立冷启动、Attention 先停、FFN 后停、两侧返回码 0 和 fatal marker 为空均通过；
- Attention DP0 和 FFN DP0 的原始 profiler 数据均成功采集，并由固定 venv/CANN 9.0.1 重新解析。

30 分钟空闲恢复和提交态 smoke 均已通过，冻结 tag 为 `dsv4-afd-a3-eager-u2-v1`。

本阶段不声称“AFD 已获得性能收益”。现有 profile 只用于建立 U2 调优起点；公平性能结论必须在下一阶段加入非 AFD 和 U1 同资源、同请求、同统计窗口对照后给出。

## 2. 固定环境

| 项目 | 值 |
|---|---|
| 硬件 | A3 环境，16 个逻辑 NPU，Attention 0-7，FFN 8-15 |
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| venv | `/mnt/workspace/code/.venvs/afd-v026` |
| vLLM | `568afb3a13806beb53bb2e6bd518269357b237c0` |
| vLLM-Ascend | `80d8c194f7584b17fe08065ea99a130916f6b0e7` |
| 基础 tag | `dsv4-afd-graph-u1-v1` |
| 开发分支 | `feat/dsv4-afd-eager-u2` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| 拓扑 | A8F8、DP8、TP1、EP8、PP1、CP1、DCP1 |
| 执行 | eager、U2、MTP/Graph/PD 关闭 |
| 确定性 | seed 1024、temperature 0 |

固定插件列表：

```text
ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd
```

## 3. 为什么需要本次修改

U1 中 Attention 可以在模型前一次性把整步 `input_ids` 发给 FFN。U2 会把同一步拆成 stage 0 和 stage 1，两个 stage 的 hidden states、token slice 和 CAMP2P group 都不同。如果仍按 U1 发送整步 IDs，会产生以下问题：

1. FFN 无法判断每个 stage 对应的 token IDs；
2. hash layer 可能使用另一个 stage 或上一个 step 的 IDs；
3. IDs 和 hidden 的 HCCL 消息顺序不一致时会死锁；
4. 不同 DP rank 的真实 token 数不同，若控制面重复本 rank 的局部 stage 数量，各 rank 会对同一 stage 构造不同 CAMP2P key。

因此 U2 不能只解除配置门禁，必须让 IDs、hidden、控制元数据和 cache 都具备一致的 stage 语义。

## 4. 关键实现

### 4.1 能力边界

- DSV4 允许 eager/U1、eager/U2 和 `FULL_DECODE_ONLY`/U1；
- `FULL_DECODE_ONLY`/U2 继续 fail-fast；
- 继续拒绝非 CAMP2P、A/F 非等量、TP/PP/CP/DCP 非 1、sequence parallel、Attention gate、MTP 和 PD；
- recipe 仅接受 `U_BATCHES=1` 或 `2`，U2 自动加入 DBO threshold 参数。

### 4.2 每 stage 的 IDs 协议

Attention 的 U1 路径仍在模型前预发送一次 IDs。U2 路径改为：

```text
Attention stage 0: send IDs -> DBO yield
Attention stage 1: send IDs -> DBO yield
FFN main thread:     recv stage 0 IDs -> recv stage 1 IDs
随后两个 stage 分别进入 hidden CAMP2P 通信
```

DBO yield 是必要的协议握手。FFN worker 使用单主线程同步预接收两个 stage 的 IDs，避免在额外 Python 线程中调用 HCCL 导致 AICore/通信上下文错误。每个 stage 使用已有 CAMP2P connector 中独立的 `afd_ids` group 和预分配 `int32` buffer，没有新增 DSV4 专用 connector。

### 4.3 FFN cache 生命周期

- 每个 `_ffn_forward()` 创建 stage-local IDs cache；
- layer 0 接收对应 stage IDs；
- hash layer 1/2 只引用本 step、本 stage 的 cache；
- layer 3 起传入 `None`；
- 正常完成或异常都会在函数退出时销毁局部 cache，不跨 step 保存。

### 4.4 非均匀 DP 元数据修复

真实 batch 32 首次运行暴露了关键问题：Attention 各 DP rank 把自己的局部 stage 数复制到全体 8 个 DP 位置，导致混合长度请求中各 rank 形成不同 CAMP2P key 并最终死锁。

修复后先保留 DP all-reduce 得到的真实未 padding token 数：

```text
[35, 21, 23, 22, 37, 25, 36, 21]
```

再按共同 split point 18 投影为：

```text
stage 0: [18, 18, 18, 18, 18, 18, 18, 18]
stage 1: [17,  3,  5,  4, 19,  7, 18,  3]
```

所有 Attention rank 因而向 FFN 发送完全一致的全局 per-stage 向量。若任一 DP rank 在某个请求的 stage 为空，当前实现显式失败；空 stage 的协议语义留给独立后续设计，不做静默降级。

### 4.5 DSA 元数据隔离

每个 ubatch 分配独立的 DSA ratio metadata cache，并按 ubatch 的 request slice 计算真实请求数。这样 stage 0/1 不会复用错误的 request count、block table 或 compressor ratio 元数据。

### 4.6 验证器增强

验证器新增：

- `--u-batches` 和两个 DBO threshold 参数；
- Attention 日志中的双 stage 运行证据门禁；
- role-local DP0 profiler 原始文件门禁；
- `Exception in thread` fatal marker；
- 最多等待 60 秒的 NPU 清理硬门禁。

清理门禁只读取 `npu-smi info`，不会终止环境中的其他任务。命令失败或超时后仍有 PID 都会使 cycle 失败，并把 PID 写入 `cycle_summary.json`。

## 5. 测试结果

设备可见环境中的相关回归：

```bash
source tools/dsv4/activate_runtime.sh
python -m pytest -q \
  tests/e2e/test_dsv4_recipe.py \
  tests/unit/connectors/test_camp2p_connector.py \
  tests/unit/v1/worker/test_npu_runtime.py
```

结果：`116 passed`。DBO helper 独立回归另有 `5 passed`。

覆盖范围包括：

- eager/U2 接受及 Graph/U2 拒绝；
- stage-local input IDs slice 和 forward context；
- 每 stage 一次 IDs 消息及 IDs-before-hidden 顺序；
- FFN layer 0/1/2/3 cache 语义和跨 step 清理；
- 非均匀 DP token vector 投影；
- stage-local DSA cache 和真实 request count；
- U1 fallback；
- recipe 参数、manifest、fatal marker、profile 和 NPU 清理门禁。

注意：Ascend 原生扩展相关用例需要设备可见环境。受限沙箱中会在 `torch.tensor(..., device="npu")` 处返回驱动不可见错误；把多个 Ascend 测试文件放入同一受污染 Python 进程也可能触发 `camem` 原生扩展 abort。正式门禁使用固定 venv、设备可见环境和 `python -m pytest`。

## 6. A8F8 正确性证据

主产物：

```text
/mnt/workspace/validation/dsv4_afd_a3_eager_u2_correctness_fix_20260813_1320
```

| 检查项 | 结果 |
|---|---|
| golden | 30/30 token IDs 精确一致，0 mismatch |
| batch 1 | 结构合法，1 choice |
| batch 8 | 结构合法，8 choices |
| batch 32 | 结构合法，32 choices |
| U2 证据 | stage 0/1 均实际执行 |
| 启动 | 370.102 秒 |
| 关闭顺序 | Attention -> FFN |
| 返回码 | Attention 0，FFN 0 |
| fatal marker | 两侧均为空 |
| NPU 清理 | 无运行进程 |

batch 请求的 `token_exact_count` 是额外诊断项，不是 golden 门禁。批处理调度与单请求执行路径不同，批量检查的硬门禁是 prompt IDs、choice 数、输出长度和有限结果；逐 token 确定性由 30 条串行 golden 请求负责。

失败现场也被保留：

```text
/mnt/workspace/validation/dsv4_afd_a3_eager_u2_correctness_20260813_1256
```

该现场记录了修复前 batch 32 的非均匀 DP key 不一致和超时，用于证明本次元数据修复对应真实故障，而非仅为单测构造。

## 7. DP0 profile 基线

产物：

```text
/mnt/workspace/validation/dsv4_afd_a3_eager_u2_dp0_profile_20260813_1340
```

采集配置为 `wait=2, warmup=1, active=10, repeat=1`，Attention role-local DP0 和 FFN role-local DP0 各一个 trace，`TORCH_PROFILER_WITH_STACK=0`。原始门禁：

| Role | CANN raw files | `torch.op_range` |
|---|---:|---:|
| Attention DP0 | 100 | 6,340,481 bytes |
| FFN DP0 | 72 | 22,361,721 bytes |

第一次解析误用了一个 activate 脚本内历史硬编码的 vLLM 0.22.1 venv，错误解析目录已重命名为 `ASCEND_PROFILER_OUTPUT_wrong_venv_vllm0221` 保留。随后明确激活 `/mnt/workspace/code/.venvs/afd-v026`，用 torch-npu 2.10.0.post2 和固定 CANN 9.0.1 根目录重新解析，正确结果位于标准 `ASCEND_PROFILER_OUTPUT`。

正确解析的 10 个 active step 摘要：

| Role | mean compute | mean non-overlap comm | mean free | mean stage | median stage | max free |
|---|---:|---:|---:|---:|---:|---:|
| Attention DP0 | 998.399 ms | 0.573 ms | 2063.313 ms | 3062.285 ms | 1151.993 ms | 18538.866 ms |
| FFN DP0 | 2118.069 ms | 260.336 ms | 537.390 ms | 2828.289 ms | 1067.962 ms | 2263.702 ms |

这些数据含 profiler 开销、启动后的动态 batch 和显著离群点，不能与此前 Graph/U1 的短窗口均值直接比较。它们说明下一阶段必须先固定请求形态和测量窗口，再分析 stage 不均衡、FFN 未重叠通信和 Attention free/bubble。

## 8. 30 分钟空闲恢复

产物：

```text
/mnt/workspace/validation/dsv4_afd_a3_eager_u2_idle_20260813_1405
```

状态：`PASSED`。

门禁包含：30/30 golden、batch 1/8/32、空闲 1800 秒、恢复后请求、双 stage 证据、严格关闭、fatal marker 和 NPU PID 清理。

| 检查项 | 结果 |
|---|---|
| 首轮 golden | 30/30 token IDs 精确一致 |
| 首轮 batch | 1/8/32 均合法 |
| 实际 U2 | 双 stage 证据存在 |
| 空闲时间 | 1800 秒 |
| 恢复请求 | 10/10 token IDs 精确一致，batch 1 合法 |
| 启动 | 388.109 秒 |
| 关闭 | Attention -> FFN，两侧返回码 0 |
| fatal marker | 两侧均为空 |
| NPU 清理 | 第一次查询通过，PID 列表为空 |

验证器在本次长测启动后又增加了“Process 表必须存在”的防截断检查，因此该运行中的旧进程结果尚无 `process_table_present` 字段；实际保存的 `npu_after_cleanup.txt` 包含完整 Process 表和各 NPU 的 `No running processes`。新增分支已由截断输出单测和提交态 smoke 实测覆盖。

提交态 smoke：

```text
/mnt/workspace/validation/dsv4_afd_a3_eager_u2_e65b31d_smoke_20260813_1457
```

该产物记录 `afd-plugin=e65b31d2c31b45a68757f78e2a7f28c4837ce5c0`、tracked worktree clean、batch 32 合法、双 stage 实际执行、两侧返回码 0、fatal marker 为空，并且 `process_table_present=true`、PID 列表为空。

## 9. 当前边界和下一步

eager/U2 已冻结为 `dsv4-afd-a3-eager-u2-v1`。下一阶段进入 A3-P3 profiling/调优，而不是立即解除 Graph/U2：

1. 固定非 AFD、eager/U1 和 eager/U2 的同口径请求矩阵；
2. 分离 prefill、decode 和稳态窗口；
3. 扫描 DBO threshold，并测量 throughput、TTFT、TPOT、HBM、利用率和 `tokens/s/NPU`；
4. 解释 Attention free、FFN communication 和 stage imbalance；
5. 只有 eager/U2 正确性与性能行为均可解释后，才开发 `FULL_DECODE_ONLY`/U2。

128K 不是本正确性门禁的默认长度。长上下文性能应作为独立矩阵逐级扩展到 4K/16K/32K/64K/128K，并先确认模型 `max_model_len`、KV cache 容量、`max_num_batched_tokens` 和超时配置。128K prefill 与 decode 稳态回答的是不同问题，不能替代当前 16-token deterministic golden 或混在同一个吞吐数字中。
