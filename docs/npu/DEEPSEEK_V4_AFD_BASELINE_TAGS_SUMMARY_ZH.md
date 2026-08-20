# DeepSeek-V4 AFD 两个基线 Tag 总结报告

## 1. 报告目的

本文总结 DeepSeek-V4 在 Ascend AFD Attention/FFN 分离场景中的两个阶段性基线：

| Tag | Commit | 基线定位 |
|---|---|---|
| `dsv4-afd-eager-u1-v1` | `40981475a9270c9b79ebf5cfe46d375472ee0a06` | A8F8、eager、U1 推理正确性基线 |
| `dsv4-afd-graph-u1-v1` | `2ed98442351d4be96edbb315a6b6c8d00805bbc4` | A8F8、`FULL_DECODE_ONLY`、U1 图执行与严格生命周期基线 |

第二个 tag 直接建立在第一个 tag 之上。它不是另一套独立实现，而是在 eager/U1 已证明正确的前提下，增加 ACL Graph 支持、严格关闭协议和可复现 profiling 门禁。

这两个 tag 的共同价值，是把后续 U2、U2 + Graph 和 PD 集成拆成可比较、可回退、可定位的独立阶段，避免同时引入多个变量后无法判断问题来源。

## 2. 背景

### 2.1 AFD 拆分目标

本项目将 DeepSeek-V4 Decode 模型按职责拆成两个角色：

```text
Attention 角色：embedding、Attention/indexer/compressor、HC、norm、LM head
FFN 角色：gate、tid2eid、routed experts、shared experts
```

每层执行流程可概括为：

```text
Attention pre/Attention
        |
        | hidden states + 必要的路由信息
        v
远端 FFN/MoE
        |
        | FFN output
        v
Attention post/HC
```

这样做的目标不是简单地启动两个进程，而是确保：

1. 两个角色只分配自己拥有的参数，避免各自加载一份完整模型。
2. 拆分后的逐层数学结果与原生 DeepSeek-V4 保持一致。
3. Attention 与 FFN 的通信顺序、shape、dtype 和生命周期严格匹配。
4. 未验证的并行或执行模式必须明确失败，不能静默产生错误 token。

### 2.2 DeepSeek-V4 的特殊问题

DeepSeek-V4 的前几个 MoE 层使用 hash routing。FFN 的 expert 选择不仅依赖 hidden states，还依赖原始 `input_ids`。

通用 AFD 数据面原本只传 hidden states。若不增加 IDs side channel，FFN 即使成功完成计算，也可能选择错误 expert，最终表现为无报错但 token 错误。这类静默正确性问题比启动失败更危险。

因此本轮适配必须同时解决：

- 模型角色化构造和权重所有权；
- Attention 到 FFN 的 hidden-state 往返；
- 每 step/stage 一次的 `input_ids` 传输；
- FFN hash layer 0/1/2 对同一批 IDs 的正确复用；
- eager 与 Graph 两种执行模式下动态 IDs 都不会被固化或错用。

### 2.3 为什么先 eager/U1，再 Graph/U1

Graph 会增加静态地址、capture/replay、编译区通信算子和动态输入更新等额外约束。如果在基础数学与通信链路尚未闭环时直接调 Graph，很难区分错误来自模型拆分、数据传输还是图捕获。

开发顺序因此固定为：

```text
非 AFD golden
  -> AFD eager/U1
  -> AFD FULL_DECODE_ONLY/U1
  -> AFD eager/U2
  -> AFD FULL_DECODE_ONLY/U2
  -> PD 生产拓扑
```

## 3. 固定验证边界

两个 tag 均使用同一套固定运行栈，未修改固定 vLLM 或 vLLM-Ascend 源码：

| 项目 | 固定值 |
|---|---|
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1` |
| Python venv | `/mnt/workspace/code/.venvs/afd-v026` |
| vLLM | `568afb3a13806beb53bb2e6bd518269357b237c0` |
| vLLM-Ascend | `80d8c194f7584b17fe08065ea99a130916f6b0e7` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| 插件 | `ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd` |
| 拓扑 | Attention NPU 0-7，FFN NPU 8-15 |
| 并行配置 | A8F8、DP8、TP1、EP8、PP1、CP1、DCP1 |
| 推理配置 | seed 1024、temperature 0、U1 |
| golden | Milestone 0 保存的 10 条非 AFD token 结果 |

当前基线显式不包含：

- DBO/U2；
- MTP/speculative decoding；
- Mooncake PD；
- sequence parallel；
- Attention 侧 gate 计算；
- A/F 数量不相等；
- TP、PP、CP 或 DCP 大于 1。

这些限制不是长期产品能力定义，而是当前已经被验证的正确性边界。

## 4. Tag 演进关系

```text
upstream/main
  |
  +-- fe60b0e  固定 DSV4 AFD 运行栈与环境门禁
  +-- 17ed389  增加角色化 DSV4 模型与权重策略
  +-- e6eb98d  增加单层 loopback 数学等价测试
  +-- 63dc551  CAMP2P 增加 input IDs side channel
  +-- 4098147  eager A8F8 端到端验证
  |     `-- tag: dsv4-afd-eager-u1-v1
  |
  +-- f5b0e68  支持并验证 U1 FULL_DECODE_ONLY
  +-- ccd1de7  增加显式控制面关闭处理
  +-- b18e502  修复 Gloo 关闭时未写接收缓冲的竞态
  +-- ceaf4f1  Attention/FFN 仅 role-local DP0 profiling
  +-- 2ed9844  profile 原始产物硬门禁
        `-- tag: dsv4-afd-graph-u1-v1
```

## 5. `dsv4-afd-eager-u1-v1`

### 5.1 阶段目标

该 tag 的目标是建立第一个可运行的 DeepSeek-V4 AFD A8F8 eager/U1 正确性基线，回答以下问题：

1. Attention 与 FFN 能否只构造和加载自己拥有的参数？
2. 拆分层是否与原生单层计算等价？
3. hash routing 所需的 `input_ids` 能否与 hidden states 对齐到达 FFN？
4. 完整服务输出能否逐 token 匹配非 AFD golden？
5. 冷启动、不同 batch 和长时间空闲恢复能否工作？

### 5.2 关键改动、原因与意义

#### 固定运行环境和可重复构建

新增：

- `tools/dsv4/activate_runtime.sh`
- `tools/dsv4/check_runtime.sh`
- `tools/dsv4/install_plugin.sh`

为什么修改：

当机器同时存在 CANN 9.0.1 与 9.1.0，单纯再次 `source set_env.sh` 可能在 `PATH`、`LD_LIBRARY_PATH` 或 `PYTHONPATH` 中留下混合版本。此时模型或自定义算子故障容易被误判为 AFD 逻辑问题。

意义：

- 将 CANN、venv、editable install、Ascend ops 和固定源码 commit 变成进入验证前的硬门禁；
- 把环境/ABI 问题与模型逻辑问题分离；
- 后续 tag 可以在同一基础上做单变量比较。

#### 注册并构造角色化 DeepSeek-V4 模型

新增 `AFDDeepseekV4ForCausalLM`、角色化 Model/DecoderLayer 和远端 MoE proxy。

Attention 角色只构造：

- embedding；
- Attention、indexer、compressor；
- HC 与 layer norm；
- final norm 和 LM head。

FFN 角色只构造：

- gate 与 `tid2eid`；
- routed experts；
- shared experts。

为什么修改：

直接复用原生完整模型会让 Attention 和 FFN 各自加载全部权重，既不符合 AFD 的职责边界，也失去显存拆分的意义。

意义：

- 参数所有权在模型结构层面可检查；
- FFN 不再意外执行完整 CausalLM forward；
- Attention 通过无参数 proxy 调用远端 MoE，保持原生 layer 前后处理顺序。

#### 按原始 checkpoint key 过滤权重

权重加载策略先依据原始 key 判断归属，再交给固定 vLLM-Ascend 的原生 DSV4 loader：

```text
Attention：embedding、head、norm、Attention、HC 等
FFN：gate、tid2eid、routed/shared expert 等
两侧跳过：mtp.*
```

W8A8 的 weight、scale 和 offset 归属同一角色；checkpoint iterator 只消费一次。

为什么修改：

若在原生 loader 映射后再猜测角色，量化附属张量可能与主权重分离；若为两个角色重复遍历一次性 iterator，则第二次加载可能为空或行为不确定。

意义：

- 继续复用经过验证的原生 DSV4/W8A8 loader；
- 避免重新实现量化权重装载；
- 通过测试固定一遍迭代、完整归属和 MTP skip 规则。

#### 单层 loopback 数学等价测试

增加 layer 0、2、3、42 的拆分前后比较，覆盖 BF16/W8A8 模拟路径和不同 token 数。

为什么修改：

端到端 token 不一致时，单靠服务日志很难定位是 Attention、MoE、HC 还是通信边界的问题。先用 loopback connector 去掉真实网络变量，可以验证拆层位置本身是否正确。

意义：

- 将模型数学边界与通信实现解耦验证；
- 同时覆盖 hash routing 层、hash/普通路由切换层和最后一层；
- 要求最终三维 layer output shape、有限值和数值误差均满足门限。

#### CAMP2P input IDs side channel

`AFDA2FTransferPayload` 增加可选 `input_ids`；CAMP2P 为每个 stage 创建独立 `afd_ids` HCCL group，并预分配 NPU `int32` buffer。

U1 下的约束为：

```text
Attention rank i <-> FFN rank i
每 step/stage 发送一次 IDs
FFN layer 0 接收，layer 1/2 复用，layer 3 起清空
```

为什么修改：

DSV4 hash selector 必须读取 token ID；hidden-state 数据面无法推导该信息。每层都发送 IDs 又会增加通信量并使消息顺序复杂化。

意义：

- 保证 hash expert 路由与原生模型一致；
- IDs 与 hidden states 使用对应 stage/rank，避免跨 step 污染；
- 独立 group 和预分配 buffer 为后续 Graph 稳定地址要求打下基础。

#### A8F8 启动与 golden 验证 recipe

增加 Attention/FFN 启动脚本、自动验证器和结果归档：

- FFN 先进入等待，随后启动 Attention；
- 自动检查端口和 API ready；
- 两轮冷启动；
- 每轮 3 x 10 条 golden，逐 token 比较；
- batch 1/8/32 响应结构检查；
- 第一轮空闲 1800 秒后恢复；
- Attention 先停、FFN 后停；
- 保存运行栈、日志、请求结果和清理后的 `npu-smi info`。

为什么修改：

手工启动和肉眼查看文本无法证明 token 完全一致，也无法可靠复现冷启动、批处理和长空闲问题。

意义：

- 将“服务能启动”提升为“结果可逐 token 验收”；
- 将运行环境和结果绑定到独立产物目录；
- 为 Graph 阶段复用同一套 golden 门禁。

### 5.3 验证结论

正式 eager 产物：

```text
/mnt/workspace/validation/dsv4_afd_m4_20260811_113734
```

结果：

- 两个冷启动周期均完成；
- 每周期 30/30 golden 请求逐 token 一致；
- batch 1/8/32 均返回数量和结构完整的响应；
- batch 对单请求 golden 的信息性 exact 计数为 1/1、3/8、10/32；batch
  8/32 的门禁是响应数量、prompt IDs 和 completion shape 完整，不要求调度形态变化后
  与逐条请求的 token 完全相同；
- 第一周期空闲 1800 秒后，10/10 golden 恢复请求一致；
- 清理后的 `npu-smi info` 已保存。

Tag 注释还记录了作者信息重写前的已验证 commit `ccad3a7` 和相同源码 tree `037daf6744f56c310770cf5008913367e81882f8`。最终 tag 指向作者为 `wenhow` 的 `4098147`。

### 5.4 已知历史边界

该 tag 应准确称为“eager/U1 推理正确性基线”，不应单独称为严格关闭基线。

原因是当时验证器将请求成功作为总体通过条件，但尚未把两侧退出码和隐藏 fatal 日志纳入硬门禁。正式产物记录了 FFN return code 为 1；请求结果和 NPU 清理通过，但退出状态不够严格。

这个问题不是在报告中忽略，而是在下一个 graph tag 中作为正式工程缺口修复：新增显式 shutdown payload、peer-close 分类、确定性接收缓冲、监督式退出以及 fatal log gate。

## 6. `dsv4-afd-graph-u1-v1`

### 6.1 阶段目标

该 tag 在不改变 U1、A8F8 和模型正确性边界的前提下启用 `FULL_DECODE_ONLY`，回答以下新增问题：

1. 动态 `input_ids` 能否在图 capture/replay 外正确更新？
2. FFN 图是否使用稳定地址的当前 step IDs，而不是捕获期旧值？
3. 图模式服务能否保持与 eager 相同的逐 token golden？
4. Attention-first shutdown 能否让 FFN 正常退出，而不是产生隐藏 fatal？
5. Attention/FFN profile 能否只采 role-local DP0，并由同版 CANN 成功解析？

### 6.2 关键改动、原因与意义

#### 受控开放 `FULL_DECODE_ONLY`

feature validation 只允许 DSV4 在 U1 下选择 eager 或 `FULL_DECODE_ONLY`；启动脚本增加 `EXECUTION_MODE`，Graph capture sizes 固定为 1/2/4/8。

DBO/U2、PD、MTP、非 CAMP2P、A/F 不相等和其他并行组合继续 fail-fast。

为什么修改：

“允许非 eager”范围过宽，会让 PIECEWISE、U2 或其他未经验证组合也进入运行期。Graph 支持必须精确描述到已验证模式。

意义：

- 将 Graph 能力限定为可证明的 U1 decode-only 子集；
- 未验证组合在启动阶段直接失败；
- 为后续 U2 + Graph 保留独立门禁。

#### 将 IDs 通信移出 compiled model

Attention runner 在进入 compiled model 前发送一次当前 `input_ids`，并通过 forward context 标记 `afd_input_ids_pretransferred`，使 layer 0 proxy 不重复发送。

为什么修改：

固定 torch-npu 运行栈无法安全地把该动态 `dist.send` 放在 compiled model 内；同时图内 tensor value 检查可能形成不可用 guard 或错误固化动态数据。

意义：

- 通信仍保持每 step/stage 一条 IDs 消息；
- graph capture 只覆盖适合静态化的模型计算；
- 每次 replay 前都把当前 step IDs 写入稳定 buffer，不捕获旧 token。

#### FFN 在图区域外接收 IDs

FFN runner 在进入 warmup/capture/replay 前接收每个 stage 的 IDs，并把收到的稳定 buffer 显式传给 layer 0；layer 1/2 使用当前 `_ffn_forward()` 生命周期内的缓存，layer 3 起不再传 IDs。

为什么修改：

若在图内执行 recv，图捕获会包含动态通信；若只在第一次 capture 时接收，后续 replay 会错误复用 capture 时的 token IDs。

意义：

- capture 和 replay 使用同一稳定地址，但内容按 step 更新；
- IDs 生命周期仍限制在一次 FFN forward；
- 同时兼容 warmup、capture、replay 和 eager fallback 路径。

#### 完善 Attention-first 控制面关闭协议

`AFDControlPayload` 增加 `shutdown` 标志。Attention shutdown 时先发送显式关闭帧；FFN worker 将该帧或可识别的 Gloo peer-close 视为正常终止。

验证器同时改为：

- Attention 与 FFN 退出码都必须为 0；
- FFN supervisor 先收到 TERM，再由它通知 vLLM 父进程和 worker；
- timeout 才使用进程组 KILL；
- 扫描 `AFD NPU FFN worker loop failed`、`EngineCore encountered a fatal error` 等 fatal marker。

为什么修改：

仅终止 Attention 进程组时，FFN 可能仍阻塞在控制面 recv；直接终止整个 FFN 进程组又会绕过框架自身的有序清理。仅看外层 shell return code 还可能掩盖内部 worker fatal。

意义：

- 关闭成为显式协议，而不是依赖连接异常；
- 验收同时覆盖返回码和内部日志；
- 正常停止、异常 peer close 与真正的数据错误可以区分。

#### 修复 Gloo 未写接收缓冲竞态

控制帧的 size/body 接收缓冲从 `torch.empty` 改为零初始化，并验证 `recv()` 返回的 source rank。未写缓冲或全零 body 被分类为 `AFDControlPlaneClosedError`。

为什么修改：

实际长空闲验证发现：Gloo 对端关闭时，`recv()` 可能返回但不写目标缓冲。若缓冲来自 `torch.empty`，随机内存可能被误解释为长度或 JSON，导致非确定性的 `JSONDecodeError`、FFN worker fatal 和 EngineCore fatal。

意义：

- 将概率性关闭竞态转为确定性的 peer-close 结果；
- 消除“短 smoke 通过、长空闲后偶发失败”的不稳定行为；
- 补充 size 未写、body 未写和零长度控制帧测试。

#### role-local DP0 双侧 profiling

profiler helper 增加显式 `role_rank`，只有 Attention rank 0 和 FFN rank 0 创建 torch-npu profiler。两侧使用独立目录，固定：

```text
wait=2, warmup=1, active=10, repeat=1, skip_first=0
TORCH_PROFILER_WITH_STACK=0
```

为什么修改：

若 DP8 的所有 worker 同时采集，会显著增加内存、磁盘和执行扰动，也难以配对 Attention/FFN 同一窗口。开启 Python stack 还会产生大量事件，扭曲性能数据。

意义：

- 得到一对可对照的 Attention/FFN DP0 trace；
- 降低 profiler 对正确性和性能窗口的干扰；
- 后续 U2 可以沿用相同采样口径比较。

#### profile 原始产物硬门禁

`--profile` 模式要求每个角色恰好一个 `*_ascend_pt` 目录，并检查：

- `profiler_info_0.json` 非空；
- `FRAMEWORK/torch.op_range` 非空；
- `PROF_*` 下存在非空 CANN raw files。

为什么修改：

服务请求成功不代表 profiler 成功。torch-npu daemon 可能提示需要离线解析，甚至在 trace 缺失时不让外层验证命令失败。

意义：

- profile 缺失、重复或空文件会直接令 validation `passed=false`；
- 原始采集与后续离线解析可以分别验收；
- 防止把只有 CPU trace 或空目录误报为完整性能证据。

### 6.3 正确性与稳定性验证

正式产物：

```text
/mnt/workspace/validation/dsv4_afd_u1_full_decode_only_b18e502_20260811_200730
```

结果：

- 两个冷启动周期整体 `passed=true`；
- 每周期 30/30 golden 逐 token 一致；
- 第一周期空闲 1800 秒后 10/10 golden 一致；
- batch 1/8/32 均结构有效；
- batch 对单请求 golden 的信息性 exact 计数为 1/1、3/8、10/32；
- 两周期 Attention/FFN return code 均为 0；
- 两周期 Attention/FFN fatal marker 均为空；
- 清理后无 NPU 运行进程。

最终 tag commit 还额外执行了非 profile smoke：

```text
/mnt/workspace/validation/dsv4_afd_graph_u1_tag_smoke_2ed9844_20260811_212500
```

结果为 10/10 golden 一致、batch 1 有效、两侧退出码 0、日志门禁通过、无 NPU 进程残留。

### 6.4 Profiling 结果

采集与解析摘要：

```text
/mnt/workspace/validation/dsv4_afd_graph_u1_dp0_profile_ceaf4f1_20260811_205830/profile_validation.json
```

采集 commit 为 `ceaf4f1`，最终 artifact gate commit 为 `2ed9844`。原始数据使用固定 CANN 9.0.1 环境离线解析。

| 指标 | Attention DP0 / device 0 | FFN DP0 / device 8 |
|---|---:|---:|
| 解析 step 数 | 10 | 10 |
| 平均 Computing | 233.670 ms | 80.871 ms |
| 平均非重叠通信 | 0.226 ms | 2.120 ms |
| 平均 Free | 2.852 ms | 5.051 ms |
| 平均 Stage | 236.749 ms | 85.922 ms |
| Free median | 2.719 ms | 0.005 ms |
| Free p90 | 3.654 ms | 0.211 ms |
| Free max | 3.780 ms | 50.261 ms |

主要累计 kernel：

- Attention：`E2a`、MatMul、QuantMatMul；
- FFN：`MoeDistributeDispatchV2`、`A2e`、`MoeDistributeCombineV2`。

这些数字的意义是建立 U1 Graph 的性能观测基线，不代表已经完成性能优化。FFN 的 50.261 ms 最大 Free 是本次窗口中的异常点；在逐项关联 host API、通信事件和相邻 kernel 前，不对其根因下结论。后续比较必须保持相同运行栈、请求形态和采样配置。

## 7. 两个 Tag 的核心差异

| 维度 | eager tag | graph tag |
|---|---|---|
| 执行模式 | eager | `FULL_DECODE_ONLY` |
| U-batch | U1 | U1 |
| Graph capture sizes | 不适用 | 1/2/4/8 |
| IDs 发送位置 | layer 0 eager 路径 | compiled model 外，每 step/stage 一次 |
| FFN IDs 接收 | layer 0 数据路径 | capture/replay 前写入稳定 buffer |
| token 正确性 | 30/30 x 2，通过 | 30/30 x 2，通过 |
| 30 分钟恢复 | 10/10，通过 | 10/10，通过 |
| 严格退出码门禁 | 尚未纳入总体判定 | Attention/FFN 均要求 0 |
| fatal 日志门禁 | 无 | 有 |
| 显式 shutdown payload | 无 | 有 |
| Gloo 未写缓冲防护 | 无 | 有 |
| Profiling | 未冻结 | Attention/FFN role-local DP0 |
| Profile raw gate | 无 | 有 |

## 8. 工程意义

### 8.1 建立可回退的正确性锚点

当 U2 或 PD 引入 token 偏差时，可以分别回退到：

- `dsv4-afd-eager-u1-v1`：判断基础模型拆分、权重和 IDs side channel 是否正确；
- `dsv4-afd-graph-u1-v1`：判断问题是否由 U2 而不是 Graph 本身引入。

这比只保留最新分支更利于二分定位和环境复现。

### 8.2 把静默错误转为显式失败

角色错误、权重归属错误、重复 iterator、缺失 IDs、未验证并行配置、隐藏 worker fatal 和空 profile 都被转成明确门禁。其共同意义是：宁可启动失败，也不接受服务表面正常但 token 或验证证据错误。

### 8.3 证明 Graph 没有固化动态 token 信息

DSV4 Graph 适配最关键的结论不是“图能 capture”，而是每次 replay 仍使用当前请求的 IDs，并在两轮冷启动、30 分钟恢复和多轮 golden 中保持逐 token 一致。

### 8.4 为 U2 提供清晰的增量边界

U2 下一步只应增加双 stage/ubatch 行为，重点检查：

- ubatch 0/1 使用各自切片后的 IDs；
- 两个 stage 使用独立 IDs group、buffer 和 cache；
- 异常、取消和 shutdown 不留下跨 step/stage 状态；
- eager U2 先通过，再开启 U2 + `FULL_DECODE_ONLY`；
- 使用相同 golden、稳定性和 DP0 profiling 口径重新验收。

## 9. 使用方式

查看或切换 eager 基线：

```bash
git show dsv4-afd-eager-u1-v1
git switch --detach dsv4-afd-eager-u1-v1
```

查看或切换 Graph 基线：

```bash
git show dsv4-afd-graph-u1-v1
git switch --detach dsv4-afd-graph-u1-v1
```

从 Graph 基线继续开发 eager U2，建议创建新分支：

```bash
git switch -c feat/dsv4-afd-eager-u2 dsv4-afd-graph-u1-v1
```

Tag 只负责固定源码身份。复现验证时仍必须使用本文第 3 节的固定 CANN、venv、vLLM、vLLM-Ascend、模型和启动参数。

## 10. 结论

`dsv4-afd-eager-u1-v1` 证明了 DeepSeek-V4 的角色化模型、W8A8 权重归属、单层拆分边界、CAMP2P IDs side channel 和 A8F8 eager 端到端 token 正确性。

`dsv4-afd-graph-u1-v1` 在此基础上进一步证明了 U1 `FULL_DECODE_ONLY` 的动态 IDs 更新、图 replay 正确性、严格 Attention-first 生命周期和双侧 DP0 profiling 可用性，并补齐了 eager 阶段未严格验收的关闭可靠性缺口。

因此当前应把 graph tag 作为继续开发 eager U2 的代码起点，把两个 tag 分别保留为 eager 与 Graph 的回归锚点；在 eager U2 门禁通过前，不应同时开启 U2 Graph 或 PD 集成。
