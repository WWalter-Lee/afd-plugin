# DeepSeek-V4 AFD HCCL P2P TP2 功能基线验证报告

## 1. 结论

M8 已达到进入 M9 Mooncake PD 的既定功能门禁，可以冻结
`P2pHcclAFDConnector` 的等量 A8F8、DP4/TP2、eager/U1 基线：

- TP2 HCCL P2P 组件的 eager、MTP 和 full Graph/MTP capture/replay 通过；
- 同栈原生 DP4/TP2 的 10 条 prompt、3 轮共 30 个结果稳定，可作为 TP2 golden；
- AFD A8F8、DP4/TP2、eager/U1 实模 F0 达到 30/30 token IDs 精确一致；
- batch 1/8/32 请求结构、角色日志、Attention 先停、FFN 后停和 NPU 清理门禁通过；
- TP1 入口、非 TP2 拓扑和两个固定上游工作树未被修改。

本报告只冻结功能基线，不创建性能结论或性能 tag。补充运行的
`TP2 + FULL_DECODE_ONLY + U2 + full-draft MTP` 最大组合未通过：FFN Graph
warmup 触发 AICore 非法内存访问。该组合已增加精确 fail-fast，不纳入 M8 发布边界，
也不阻塞按既定优先级进入 M9。

## 2. 固定环境

| 项目 | 值 |
|---|---|
| SoC | A3，16 NPU |
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| venv | `/mnt/workspace/code/.venvs/afd-v023-vllm-cann` |
| vLLM | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，`releases/v0.23.0` |
| vLLM-Ascend | `3da28f9414583d2d0b672a8f06d1fae142404bda`，`rfc/vllm_cann` |
| M8 基线前 afd-plugin | `a4699d5b34563314992728568308476859d311a9` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| connector | `P2pHcclAFDConnector`，同步 HCCL `send/recv` |

部署使用 8 个 Attention 物理 rank 和 8 个 FFN 物理 rank。TP 从 1 改为 2 后，
每侧逻辑 DP 从 8 改为 4；EP world size 仍为 8。物理 A/F 数量仍相等，Attention
rank `i` 与 FFN rank `i` 交换 hidden、output 和 IDs。

## 3. 关键改动及原因

1. control payload 增加 `tensor_parallel_size`，旧 payload 缺省为 1。
   原因是 FFN 不能只从本地配置推断 Attention 的 TP，必须在执行前校验两侧一致。
2. MTP 的逻辑 DP token count 按 TP rank 复制到物理 rank。
   原因是 HCCL peer 和 buffer 按物理 rank 建立，而调度器 metadata 按逻辑 DP 生成。
3. FFN 收到完整 control payload 后直接更新 connector state。
   原因是只用 `dp_metadata_list` 重建 payload 会丢失 TP、Graph、warmup 和 MTP phase。
4. 部署、validator、native baseline 和 performance manifest 显式记录物理 rank、DP 和 TP。
   原因是 A8F8 在 TP2 下代表 DP4/TP2，不能再把物理 rank 数误写成 DP8。
5. TP2 明确使用 `flashinfer_all2allv` 并保持 `enable_sp=False`。
   原因是固定 vLLM-Ascend 栈用该后端关闭 vLLM 的隐式 sequence-parallel MoE；SP
   已按路线图后移，不属于 M8。
6. feature validation 只接受 TP1/TP2；TP2 只接受标准 HCCL P2P、等量 A/F 和
   `physical ranks = DP x TP`。TP3、CAMP2P TP2、非等量 TP2 继续 fail-fast。
7. 已复现失败的 TP2 full-draft Graph U2 组合增加精确 fail-fast。
   原因是不能把组件协议通过扩写成实模 kernel 组合已经可发布。

这些改动没有引入新 connector，也没有调用 CAMP2P 自定义 A2E/E2A op。数据面仍是
`torch.distributed.send/recv` 对应的标准 HCCL P2P。

## 4. 验证结果

### 4.1 HCCL 组件

| 场景 | 拓扑 | 结果 |
|---|---|---|
| eager，2 stage x 2 step | A2F2，DP1/TP2 | 通过 |
| eager + MTP | A2F2，DP1/TP2 | 通过 |
| target/full-draft Graph + MTP capture/replay | A2F2，DP1/TP2 | 通过 |

三个场景都完成 hidden、IDs、output、MTP header、Graph 静态输入更新和 connector
close；所有 worker 返回码为 0。

### 4.2 原生 TP2 golden

原生单实例使用 NPU0-7、DP4/TP2、seed 1024、temperature 0、max tokens 16。
10 条固定 prompt 连续运行 3 轮，结果文件 `passed=true`，形成 30 个稳定的 token
序列，供 AFD F0 逐 token 对照。

### 4.3 AFD 实模 F0

AFD 使用 NPU0-7 Attention、NPU8-15 FFN、两侧 DP4/TP2、eager/U1。结果：

- 10 条 prompt x 3 轮，30/30 token IDs 与原生 TP2 golden 精确一致；
- batch 1/8/32 均返回正确 choice 数和有效结构；
- 启动耗时 312.169 秒；Attention/FFN fatal log gate 通过；
- Attention 先停、FFN 随后退出，两侧返回码为 0；
- 清理后 `npu-smi info` 无运行进程。

batch 并发 token exact 只作诊断，分别为 1/1、7/8、20/32；正式确定性门禁是串行
30 请求的 30/30 token exact，与既有 validator 口径一致。

## 5. 补充最大组合与限制

额外执行了 A8F8、DP4/TP2、`FULL_DECODE_ONLY`、U2、full-draft MTP、batch 32
冒烟。两侧正确解析为 `all2all_backend=flashinfer_all2allv`、`enable_sp=False`；
Attention 完成 Graph capture/replay，U2 门禁观测到两个 stage。FFN 在 Graph warmup
后的 device synchronize 报错：

```text
AFD NPU FFN worker loop failed
error code is 507015
The aicore execution is abnormal
The MPU address access is invalid
```

因此不能声称 TP2 已覆盖所有 Graph/U2/MTP 组合。M8 发布边界保持 eager/U1；
`TP2 + full-draft MTP Graph U2` 现在启动即 fail-fast。该问题后续应拆分验证 TP2
Graph U1、target Graph + eager draft、eager U2 + MTP，再决定是否解除限制。

## 6. 验证产物

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_tp2_component_a2f2_20260821_m8/summary.json
/mnt/workspace/validation/dsv4_afd_v023_hccl_tp2_eager_mtp_component_a2f2_20260821_m8/summary.json
/mnt/workspace/validation/dsv4_afd_v023_hccl_tp2_full_graph_mtp_component_a2f2_20260821_m8/summary.json
/mnt/workspace/validation/dsv4_v023_vllm_cann_native_tp2_m8_20260821/golden_results.json
/mnt/workspace/validation/dsv4_afd_v023_hccl_tp2_eager_u1_a8f8_20260821_m8_f0_retry1/validation_summary.json
/mnt/workspace/validation/dsv4_afd_v023_hccl_tp2_graph_u2_full_draft_mtp_a8f8_20260821_m8_smoke_retry1/validation_summary.json
```

最后一个目录是明确保留的失败证据，不是通过产物。

## 7. 后续优先级

M8 冻结后直接进入 M9 Mooncake PD：先审计固定目标栈中的 Mooncake/KV connector，
冻结 P/D/AF 角色和 ownership，再完成最小组件生命周期，随后先做 TP1 eager/U1 的
PD + A8F8，最后复用本报告冻结的 DP4/TP2 control/rank 契约。SP、CP、DCP 和 PP
统一后移到 M9 之后，不作为 Mooncake PD 的前置条件。
