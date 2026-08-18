# DeepSeek-V4 AFD A3-P7M0 原生 MTP 基线与协议冻结报告

## 1. 结论

A3-P7M0 已在目标栈完成。非 AFD 原生 DeepSeek-V4 MTP 能够以
`num_speculative_tokens=1`、eager、DP8/EP8 启动并完成推理；10 条 prompt
连续 3 轮共 30 个请求的最终 token IDs 与同栈 MTP-off 基线 30/30 完全一致。

本阶段只冻结 AFD MTP 的角色、权重和消息契约。AFD 的 MTP fail-fast 仍然
保留，尚不能把本报告解释为 HCCL P2P AFD 已支持 MTP。下一阶段是
A3-P7M1：A8F8、HCCL P2P、eager/U1 + MTP。

## 2. 固定环境与启动范围

| 项目 | 值 |
|---|---|
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| venv | `/mnt/workspace/code/.venvs/afd-v023-vllm-cann` |
| vLLM | `releases/v0.23.0` / `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann` / `3da28f9414583d2d0b672a8f06d1fae142404bda` |
| torch-npu | `2.10.0.post2` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| 主模型 | `DeepseekV4ForCausalLM`，43 层 |
| draft 模型 | `DeepSeekV4MTPModel`，1 个 MTP layer |
| 并行 | NPU 0-7，DP8/EP8，TP1/PP1 |
| 执行 | eager，`method=mtp`，`num_speculative_tokens=1` |

原生脚本增加默认关闭的显式开关：

```bash
ENABLE_MTP=1 MTP_NUM_SPECULATIVE_TOKENS=1 \
  tools/dsv4/run_v023_native_baseline.sh
```

脚本的 `VLLM_PLUGINS` 不包含 `afd`，所以该结果没有经过 AFD model wrapper
或 connector。

## 3. 验证结果

| 门禁 | 结果 |
|---|---:|
| 原生 MTP 启动 | 通过 |
| draft model load | 8/8 worker 通过 |
| 三轮内部稳定 | 30/30 |
| 对同栈 MTP-off token IDs | 30/30 |
| drafted tokens | 264 |
| accepted tokens | 198 |
| 聚合 acceptance rate | 75.0% |
| MTP-on 每 rank 权重 | 47.4348 GiB |
| MTP-off 每 rank 权重 | 44.4493 GiB |
| MTP-on 可用 KV cache | 约 4.74 GiB |
| MTP-off 可用 KV cache | 约 7.76-7.77 GiB |
| 运行期间最大观测 HBM | 59,713 MiB / 65,536 MiB |
| 停止后 NPU 残留进程 | 0 |

这里的 acceptance rate 来自服务端 16 个统计窗口的计数求和，不是吞吐
收益结论。本阶段没有运行 P1/P2 性能矩阵。

停止服务时 vLLM 0.23 的进程管理器以 abort 模式结束 8 个 worker，TBE helper
记录了父进程退出告警；API shutdown、EngineCore teardown 均完成，随后
`npu-smi info` 无残留。该现象保留在日志中，M1 生命周期门禁仍需重新检查。

验证产物：

```text
/mnt/workspace/validation/dsv4_v023_vllm_cann_native_mtp_m0_20260818/
  server.log
  golden_results.json
  native_mtp_vs_native_no_mtp.json
  mtp_weight_contract.json
  validation_summary.json
  npu_during_service.txt
  npu_after_cleanup.txt
```

## 4. 权重所有权

真实 checkpoint index 共发现 2,347 个 `mtp.*` key：

| 所有者 | 原始 key 规则 | key 数 |
|---|---|---:|
| FFN | `mtp.<layer>.ffn.*` | 2,315 |
| Attention | 其余全部 `mtp.<layer>.*` | 32 |

Attention 的 32 个 key 包含 embedding、`e_proj`、`h_proj`、norm、Attention、
HC 和 head。`mtp.0.ffn_norm.*` 与 `mtp.0.hc_ffn_*` 仍属于 Attention，因为
它们在远端 MoE 调用的前后执行；只有 `mtp.0.ffn.*` 的 gate、routed/shared
experts 属于 FFN。

W8A8 的 weight、scale、offset 以原始 tensor family 一起分类，禁止跨角色。
审计工具 `tools/dsv4/audit_mtp_contract.py` 直接读取 safetensors index，当前
结果为 Attention 32、FFN 2,315、混合角色 tensor family 0。

M1 继续保持 checkpoint iterator one-shot：先按以上原始 key 过滤，再分别
交给基于原生 DSV4/MTP loader 的角色 loader；不能加载后再删除另一侧参数。

## 5. Target Hidden States 契约

原生 target 在第 42 层完成后、`hc_head` 之前，把 HC residual flatten 后写入
稳定地址 `_mtp_hidden_buffer`。当前配置为：

```text
capacity shape = [max_num_batched_tokens, hc_mult * hidden_size]
               = [1024, 4 * 4096]
               = [1024, 16384]
dtype          = bfloat16
valid slice    = [:current_target_token_count]
```

生产者是 Attention role 的 target model；消费者是 Attention role 的 MTP
proposer。每次 target forward 都必须以 `copy_` 刷新有效前缀。有效期只到
下一次 target forward，消费者必须同时使用当前 token count，不能读取 buffer
尾部，也不能跨 step 或 request 缓存旧 view。异常、取消和 shutdown 时清除
Python 引用及当前有效 token count；稳定预分配 buffer 可以复用，不要求逐字节
清零。

## 6. Draft、Verify 与 Sampler 所有权

调用顺序冻结为：

```text
target forward/verify
  -> target logits
  -> rejection sampler 或普通 sampler
  -> 从当前 target hidden buffer 构造 MTP draft 输入
  -> MTP draft forward
  -> draft logits/sample
  -> draft token IDs 交回 scheduler
  -> 下一 target verify
```

embedding、positions、MTP Attention/HC、LM head、logits processor、普通
sampler 和 rejection sampler 均在 Attention role。FFN role 只执行 MTP block
中的 gate 和 routed/shared experts，不拥有 logits 或 acceptance 决策。

原生 MTP 构造 `DeepseekV4MoE(..., is_draft_layer=True)`，因此当前 MTP MoE
使用学习式 gate，不使用主模型前 3 层的 hash router，也不需要把 MTP input
IDs 发送到 FFN。input IDs 和 positions 只供 Attention 侧 embedding/Attention
使用。若未来上游 MTP 改为 hash router，必须新增独立 IDs 消息并重新验收，
不能沿用当前结论。

## 7. HCCL Phase 与消息契约

M1 不把 MTP 伪装成第 43 个普通 decoder layer。传输标识至少包含：

```text
(transaction_id, stage_idx, phase, phase_layer_idx, speculative_step, token_count)
phase = decoder | mtp
```

首版 MTP 固定 `phase_layer_idx=0`、`speculative_step=0`、U1。严格消息顺序为：

```text
Attention -> FFN: MTP hidden [T, 4, 4096], bfloat16
FFN       -> Attention: MTP output [T, 4, 4096], bfloat16
```

> M1 勘误（2026-08-18）：以上三维 shape 是 M0 根据 target HC residual 得出的
> 预冻结假设。M1 真实 AF 执行确认远端 MoE 边界位于 HC collapse 之后，原生 MoE
> 只接受二维输入；线上协议已经修正为双向 `[T,4096]` BF16，并显式拒绝三维
> pre-HC tensor。M0 的权重所有权、target buffer 和“不发送 IDs”结论保持不变。

MTP phase 使用独立预分配 buffer 和独立 cache key，不能复用尚未完成的 decoder
layer transfer state。等量 A8F8 下 Attention rank `i` 只和 FFN rank `i`
交换。FFN receive state 必须在 matching output send 后于 `finally` 清理；取消、
异常和 connector close 时清除 MTP phase state，再销毁 process group。

控制面的 token count 必须先于 data recv 可见。现有 decoder layer 0 的 IDs
side channel 不扩展到 MTP phase；因此当前 MTP 每个 step 只新增一对 hidden/output
消息，不新增 IDs 消息。

## 8. M1 进入条件和范围

M0 已满足进入 M1 的条件。M1 实现范围严格限定为：

- `P2pHcclAFDConnector`；
- A8F8 等量拓扑；
- eager/U1；
- `method=mtp`、`num_speculative_tokens=1`；
- TP/PP/CP/DCP 为 1，sequence parallel、PD 和 Attention gate 关闭。

实现顺序为角色化 MTP wrapper 与 loader、target hidden buffer、MTP phase
metadata/cache、FFN virtual-layer dispatch、CPU/Mock 与 NPU component，最后再做
A8F8 golden、batch、生命周期和 P1 单点 guard。Graph、U2、非等量拓扑和更多
speculative token 继续 fail-fast。
