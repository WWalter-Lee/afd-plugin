# DeepSeek-V4 AFD HCCL P2P Graph/U1 验证报告

## 1. 结论

标准 HCCL P2P connector 已在 vLLM 0.23 + vLLM-Ascend `rfc/vllm_cann` 目标栈完成 A8F8 Graph/U1 功能闭环。当前支持范围是：

- `P2pHcclAFDConnector`；
- A8F8，Attention/FFN rank 数量相等；
- `FULL_DECODE_ONLY`；
- U1；
- AFD hidden、FFN output 和 input IDs 全部走 HCCL P2P，不调用 CAMP2P 自定义传输 op。

完整门禁达到 30/30 golden token IDs 一致，batch 1/8/32 有效，Graph compile/capture/replay、两次成功冷启动、正常退出和 NPU 清理通过。这是功能正确性结论，不是性能收益结论。

## 2. 固定环境

| 项目 | 固定值 |
|---|---|
| 日期 | 2026-08-18 |
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| venv | `/mnt/workspace/code/.venvs/afd-v023-vllm-cann` |
| vLLM | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，`releases/v0.23.0` |
| vLLM-Ascend | `3da28f9414583d2d0b672a8f06d1fae142404bda`，`rfc/vllm_cann` |
| torch-npu | `2.10.0.post2` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| golden | `/mnt/workspace/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json` |

拓扑固定为 Attention NPU 0-7、FFN NPU 8-15、DP8/TP1/EP8，seed 1024、temperature 0、MTP/PD/sequence parallel 关闭。

## 3. 关键实现和原因

### 3.1 Graph 内 HCCL send/recv

eager 路径使用的 `torch.distributed.send/recv` Python wrapper 不能直接稳定进入目标栈的编译图。实现只在 `torch.compiler.is_compiling()` 时调用 torch-npu 已注册的 `npu_define::_send/_recv` op，并复用同一 HCCL group、peer rank 和 tag；非编译路径继续使用标准 `torch.distributed.send/recv`。

这不是新增传输协议或 connector，也没有调用 `torch.ops.vllm.afd_camp2p_send_attn_output()`。lowering 只是让同一 HCCL P2P 语义能被 AscendCompiler 捕获。

### 3.2 不传动态 shape

第一次实现把 `torch.Size` 作为 shape 传给 `_send/_recv`，vLLM 编译器会把 shape 展开，导致 op schema 参数数量不匹配。改成显式 shape list 后，Dynamo 又把动态 token 维度专门化为首次请求的常量 8，违反动态 shape 约束。

目标 torch-npu 的 Meta 和 PrivateUse 实现并不依赖可选 shape 参数，收发 tensor 本身已经提供 shape。因此最终向该可选参数传 `None`，既保持 HCCL op 语义，也避免对 token 维度增加常量 guard。

### 3.3 input IDs 保持在 graph 外

DSV4 input IDs 仍通过一次性 HCCL side channel 在 hidden graph 执行前预传。这样每个 step 使用当前 IDs，不会把首次 capture 的 IDs 固化进图，也不会把动态长度控制消息放进 capture/replay。FFN layer 0 接收、layer 1/2 复用、layer 3 起清空的生命周期保持不变。

### 3.4 Fail-fast 边界

当前只解除等量 A/F、`FULL_DECODE_ONLY`、U1 的 Graph 门禁。以下配置继续在启动前拒绝：

- Graph/U2；
- Graph/U3；
- Graph 下 Attention/FFN rank 数量不相等；
- 非 `FULL_DECODE_ONLY` Graph 模式；
- MTP、PD、Attention gate、sequence parallel；
- TP、PP、CP 或 DCP 大于 1。

eager 下已经实现的 `A = k x F` 非等量拓扑不受影响，但不能据此宣称非等量 Graph 已支持。

## 4. 自动化验证

目标运行栈执行结果：

| 验证 | 结果 |
|---|---:|
| `tests/e2e/test_dsv4_recipe.py` | 34 passed |
| `tests/unit/v1/worker/test_npu_runtime.py -k dsv4_feature_validation` | 23 passed |
| `tests/unit/connectors/test_p2p_hccl_connector.py` | 33 passed |
| 角色脚本 `bash -n` | passed |
| Ruff、compileall、`git diff --check` | passed |
| `tools/dsv4/check_v023_vllm_cann_runtime.sh` | passed |

测试覆盖 Graph 等量接受、Graph 非等量拒绝、Graph/U2 拒绝、编译 lowering 不携带动态 shape guard，以及 recipe/runtime manifest 使用 HCCL P2P connector。

## 5. A8F8 E2E 结果

### 5.1 冒烟和完整门禁

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u1_smoke_20260818_112726
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u1_full_20260818
```

两次运行均独立完成模型加载、编译、Graph capture、请求、退出和清理。完整门禁结果：

| 门禁 | 结果 |
|---|---:|
| 10 prompt x 3 轮串行 golden | 30/30 token IDs 完全一致 |
| batch 1/8/32 | 返回数量、结构和生成有效性通过 |
| Attention 8 rank Graph capture | 全部完成 |
| Attention 8 rank ACL Graph replay | 全部观测到 |
| fatal log gate | passed |
| Attention 先停、FFN 后停 | passed，双方 return code 0 |
| NPU cleanup | passed，16 张 NPU 无运行进程 |

并发 batch 的 `token_exact` 是诊断项，不作为强门禁；并发 DP 调度顺序不保证逐样本 token 确定性。严格 token 一致性由串行 30 请求门禁负责。

### 5.2 Graph 运行证据

Attention 日志确认：

- `enable_npugraph_ex: True`；
- capture size 1/2/4/8 全部完成；
- 8 个 rank 的 Graph capture 耗时约 17-18 秒、额外 HBM 约 0.33-0.34 GiB/rank；
- 8 个 rank 均输出 `Replaying aclgraph`。

完整汇总文件：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u1_full_20260818/validation_summary.json
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u1_full_20260818/cycle_1/cycle_summary.json
```

## 6. 下一步

当前实现已经由 tag `dsv4-afd-v023-hccl-graph-u1-v1` 冻结为 HCCL P2P Graph/U1 功能基线。下一功能阶段是 MTP：先建立目标栈原生 MTP golden 和 AFD 权重/消息契约，再依次完成 HCCL P2P eager/U1 + MTP 与 Graph/U1 + MTP；当前 MTP 门禁继续保留。

MTP 功能闭环后再进入性能阶段。性能比较至少包含同参数非 AFD Graph、HCCL P2P eager/U1 和 HCCL P2P Graph/U1，并为每种模式增加 MTP on/off 对照及 acceptance rate。Graph/U2、Graph/U3 和 Graph 非等量拓扑均作为后续独立功能里程碑。当前阶段继续使用阻塞式同步 HCCL，不引入 `isend/irecv`、后台通信线程或异步自定义 op。
