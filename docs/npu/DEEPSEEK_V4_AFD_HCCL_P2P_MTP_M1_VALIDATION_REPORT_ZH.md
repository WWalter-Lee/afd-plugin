# DeepSeek-V4 AFD HCCL P2P MTP M1 验证报告

## 1. 结论

A3-P7M1 已在目标栈完成 DeepSeek-V4 AFD eager/U1 + MTP 功能闭环。实现只使用
`P2pHcclAFDConnector` 的标准阻塞式 HCCL `send/recv`，没有调用 CAMP2P 自定义
传输 op，也没有修改固定 vLLM 或 vLLM-Ascend 源码。

支持边界为 A8F8 等量拓扑、eager、U1、1 个 MTP layer、`method=mtp` 和
`num_speculative_tokens=1`。10 条 prompt 连续 3 轮共 30/30 最终 token IDs 与
目标栈 golden 一致；batch 8/32 分别 8/8、32/32 token exact。P1 单点 128/128
请求成功，无 OOM、timeout、fatal 日志或 NPU 进程残留。

M1 是功能 tag，不是性能 tag。P1 单轮吞吐优于最近可比 MTP-off 均值，只能证明
没有灾难性回退；正式性能收益必须等 M2 后按 P2 三轮公平对照验收。

## 2. 背景和目标

M0 证明目标栈原生 MTP 可运行，并冻结了 2,347 个 `mtp.*` checkpoint key 的
角色所有权。M1 要解决的是严格 AF 分离后的缺口：Attention role 仍负责 target、
MTP embedding/Attention/HC/head、proposal 和 verify，FFN role 只拥有并执行 MTP
MoE；两侧通过 HCCL 交换 draft MoE 输入和输出。

如果直接复用原生完整 MTP model，Attention 与 FFN 都会构造对方模块，失去 AFD
的参数/HBM 分离意义。如果只移除 speculative fail-fast，又会因缺少 target hidden
buffer、MTP phase 和 FFN virtual-layer dispatch 而在运行时失败。因此本阶段同时
补齐角色化构造、权重加载、调度和通信协议。

## 3. 固定环境

| 项目 | 固定值 |
|---|---|
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| venv | `/mnt/workspace/code/.venvs/afd-v023-vllm-cann` |
| vLLM | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `3da28f9414583d2d0b672a8f06d1fae142404bda` |
| torch-npu | `2.10.0.post2` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| 拓扑 | Attention NPU0-7，FFN NPU8-15，DP8/TP1/EP8 |
| 插件 | `ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd` |
| 执行 | eager/U1，MTP 1 layer，1 speculative token |

## 4. 关键改动及原因

### 4.1 角色化 MTP model 和权重

- 新增 role-aware DSV4 MTP wrapper。Attention 构造 embedding、projection、norm、
  Attention、HC、head 和 sampler 路径；FFN 只构造 gate、routed/shared experts。
- 原始 checkpoint key 先按 `mtp.<layer>.ffn.*` 过滤给 FFN，其余 `mtp.*` 给
  Attention，再交给原生 loader。weight/scale/offset tensor family 始终同角色。
- checkpoint iterator 只消费一次，避免生成器二次遍历导致静默漏权重。
- MTP container 显式提供 `start_layer=0`、`end_layer=1`。目标 vLLM-Ascend 的
  layer-index cache 会读取这两个属性，缺失时 model load 后的首次执行会失败。

意义是保持真正的参数所有权和 HBM 分离，同时继续复用固定上游的 MTP proposer、
verify/rejection sampler 和原生量化 loader。

### 4.2 Target hidden-state 生命周期

Attention target model 恢复稳定地址 BF16 buffer，capacity 为 `[1024,16384]`。
每次 target forward 只覆盖当前 token count 的有效前缀，MTP proposer 只消费当前
step 的 slice。Python 侧当前 view 不跨 step/request 缓存，异常和 shutdown 不保留
旧引用。

这个 buffer 是 target 与 draft 的必要契约；如果缺失或复用旧前缀，proposal 可能
使用上一请求的 hidden state，最终 token 偶发错误且很难由单个 kernel 测试发现。

### 4.3 独立 MTP phase 和 HCCL 协议

普通 decoder 与 MTP virtual layer 使用不同 metadata：

```text
phase = decoder | mtp
M1: phase_layer_idx=0, speculative_step=0, stage_idx=0
```

MTP 的严格消息顺序为：

```text
Attention -> FFN: header(magic, speculative_step, DP size, per-DP token count)
Attention -> FFN: post-HC hidden [T,4096], BF16
FFN       -> Attention: MoE output [T,4096], BF16
```

header 复用预分配的 IDs HCCL group，但不会发送 input IDs。当前 draft MoE 使用学习
式 gate，input IDs 只在 Attention 侧用于 embedding/Attention；MTP proxy 也不会查询
普通 decoder 的 IDs cache。hidden/output 使用独立二维 receive buffer，并在 send 侧
拒绝 pre-HC `[T,4,4096]` tensor。

M0 根据 target HC residual 暂定过三维跨 AF 传输。M1 实机运行证明 MoE 边界位于
HC collapse 之后，原生 MoE 只接受二维 `[T,4096]`。该修正把协议对齐到真实算子
入口，并用 shape guard 防止以后无意回退到错误边界。

### 4.4 Runner、recipe 和 fail-fast

- Attention runner 继续执行上游 target/proposer/verify，只把 MTP MoE 代理到 FFN。
- FFN runner 在普通 43 层完成后接收 MTP header，构造当前 phase context，执行唯一
  MTP MoE，发送 output，并在 `finally` 路径清理当前 metadata。
- recipe 增加 `ENABLE_MTP` 和 `MTP_NUM_SPECULATIVE_TOKENS`，功能与性能 runner 都
  把真实配置写入 `runtime.json`。
- 非 HCCL connector、A/F 不等量、Graph、U2、非 MTP method、多 speculative token、
  多 MTP layer 和 draft 非 eager 均显式拒绝。

严格 fail-fast 的意义是让 M1 tag 只表达已经真实验收的范围，避免配置被接受后在
深层 HCCL recv 中挂死。

## 5. 功能验证结果

| 门禁 | 结果 |
|---|---:|
| smoke | 1/1 token exact，启动 416.104s |
| serial golden | 30/30 token exact，mismatch 0 |
| batch 8 | 8/8 token exact |
| batch 32 | 32/32 token exact |
| golden acceptance | accepted 198 / drafted 264 = 75.0% |
| 成功冷启动 | smoke、golden、batch、P1、idle-resume 共 5 次 |
| 30 分钟空闲恢复 | 空闲前后均 1/1 token exact |
| Attention/FFN 正常退出 | 两侧返回码均为 0 |
| 配置的 fatal marker | 0 |
| NPU cleanup | 每次首次清理检查通过，无残留进程 |

F0 产物：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m1_smoke_20260818_163000
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m1_20260818_164100
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m1_batch_20260818_165800
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m1_idle30_20260818_173300
```

CPU/Mock 回归覆盖 architecture 注册、role construction、真实 key 分类、W8A8
suffix、one-shot iterator、target buffer、phase/header、消息数量、二维 shape、cache、
runner dispatch 和 fail-fast。运行栈导入门禁及 Python lint 同时通过。

idle-resume 运行启动耗时 402.101s，持续空闲 1800s 后再次生成完全相同的 token
IDs；Attention/FFN 返回码均为 0，首次 cleanup 检查无进程。FFN 在计划内 SIGTERM
之后、`[shutdown] MPClient: complete` 之后打印了 vLLM 0.23
`KeyboardInterrupt: terminated` traceback 和 `ERR99999`，这是 API launcher 的
shutdown 噪声；它不属于配置的运行期 fatal marker，但作为已知日志现象保留，
不得在排障时与请求执行阶段的 traceback 混淆。

## 6. P1 单点防退化结果

P1 固定 A8F8、C32、输入 1024 token、精确输出 128 token、128 请求、1 次测量：

| 指标 | MTP-on P1 | 最近可比 MTP-off 基线 |
|---|---:|---:|
| 请求成功 | 128/128 | 每轮 128/128 |
| output throughput | 28.280 token/s | 三轮均值 17.082 token/s |
| p50 TPOT | 1026.239 ms | 三轮均值 1778.871 ms |
| acceptance rate | 85.70% | 不适用 |
| Attention 最大 HBM | 59,650 MiB | 59,963 MiB |
| FFN 最大 HBM | 44,253 MiB | 43,470 MiB |

候选吞吐相对可比均值为 `+65.560%`，高于 `-20%` 灾难性回退门限；两侧日志、
shutdown 和 NPU cleanup 均通过。由于 MTP-on 只有一轮，且 acceptance 会随 workload
变化，该数字不用于收益声明、阈值调整或性能 tag。

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m1_p1_20260818_172000
/mnt/workspace/validation/dsv4_afd_v023_vllm_cann_perf_u1_c32_1k128_r3
```

## 7. 当前边界和下一步

M1 不支持 Graph + MTP、U2 + MTP、A/F 非等量 + MTP、两个及以上 speculative
token、多个 MTP layer、PD 或 sequence parallel。普通 MTP-off eager/U1、U2 和
Graph/U1 的既有路径仍需保持回归通过。

下一阶段是 A3-P7M2：等量 A8F8、`FULL_DECODE_ONLY`、U1 + MTP。M2 需要证明
target verify 与 draft graph 的真实 capture/replay、graph key 隔离、当前 step draft
state 刷新和 HCCL 消息序列稳定；完成 F0 和单点 P1 后，才进入 P2 正式性能对照。
