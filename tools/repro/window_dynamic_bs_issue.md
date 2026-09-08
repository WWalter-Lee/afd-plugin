# AttentionWorkerCombine 动态 BS 复用问题

## 问题与复现方法

DSV4 Window AFD 在线服务复用同一个 `ScheduleContext` 和固定容量的 Attention Window。原实现却把本次真实 token 数作为 `expert_scales` 的第 0 维传给 `AttentionWorkerCombine`，因此同一算子的逻辑 BS 会在不同执行步之间变化。

已观察到的服务现象是：服务重启后的第一个短请求正确；第二个相同短请求错误；长请求第一次就可能错误。关闭 prefix caching 后现象不变。日志进一步确认 A2F、FFN batching 和 F2A 的输入一致，而第二次请求的 combine 输出不同；错误调用结束后，仅 token 0 的 flag 被清零，其余 token 的 flag 仍为 1。

[`window_dynamic_bs_combine.py`](window_dynamic_bs_combine.py) 将模型、vLLM、HCCL 和 FFN 计算全部移除，只保留：

1. 创建容量为 256 的 Attention Window 和 `ScheduleContext`。
2. 在 Window 中写入确定性的 BF16 token 数据和就绪 flag。
3. 复用同一个 context，按 `BS=8 → 1 → 8` 调用三次 `AttentionWorkerCombine`。这里特意使用 8，使 BS 不会与 DSV4 的 routed top-k=6 混淆。
4. 每次比较算子输出与独立计算的参考公式，并检查该次 token 的 7 个 flag 是否全部清零。

运行：

```bash
python tools/repro/window_dynamic_bs_combine.py
```

正常情况下每次应满足 `status=PASS`，最终输出 `RESULT: not reproduced by the combine-only test`。若第三次出现数值误差或只有第一个 token 的 flag 被清零，脚本返回非零并输出 `RESULT: dynamic-BS issue reproduced`。

这个单卡脚本直接写 Attention Window，因此只验证 combine 算子自身对动态 BS 的复用。如果它未复现，不能否定线上问题，而是说明下一步应把 F2A 写 Window 的过程加入一个最小双卡用例。

## 为什么 vLLM 的 BS 会变，而 ref 不变

这里有两个不同的 BS 概念：

- Window 的物理容量在 `ScheduleContext` 创建时由 `micro_batch_size` 固定。本项目取 `max_num_batched_tokens`，例如 256。
- combine 的逻辑 BS 由本次 `expert_scales.shape[0]` 决定。原实现保存真实请求的 scales，所以 prefill、decode、空闲 DP dummy step 会产生不同逻辑 BS。

vLLM 在线调度按每一步实际调度的 token 组织模型输入。一个短 prompt 的 prefill 可以有多个 token，而 decode 通常每条序列为一个 token。AFD 又要求 Attention DP rank 锁步；某个 rank 处理真实请求时，其他空闲 rank 需要执行 dummy batch。具体依据是：

- `afd_plugin/compat/patches/engine_core.py` 的主循环在本 rank 没有可执行请求、但全局仍在运行时调用 `execute_dummy_batch()`。
- `afd_plugin/v1/worker/attention_model_runner.py` 的 `_dummy_run()` 注释明确说明：其他 DP rank 服务请求时，空闲 DP rank 会执行 vLLM dummy batch。
- `AttentionWorkerCombine` 的 host tiling 从 `expert_scales` 第 0 维读取本次 BS；内核又按这个 BS 定位 token data/token info，并只清理本次 BS 对应的 flags。代码位于 `ops-transformer/attention/attention_worker_combine/op_host/attention_worker_combine_tiling.cpp` 和 `op_kernel/attention_worker_combine_*.h`。

对应的关键代码分别是：

```python
# afd_plugin/compat/patches/engine_core.py
executed = self._process_engine_step()
if not executed and not self.model_executor.is_sleeping:
    self.execute_dummy_batch()
```

```cpp
// attention_worker_combine_tiling.cpp
auto expertScalesShapeTuple = GetShapeTuple(context_, EXPERT_SCALES_INDEX);
int64_t batchSize = std::get<SHAPE_IDX_BS>(expertScalesShapeTuple);
tilingData_.set_BS(batchSize);
```

```cpp
// attention_worker_combine_split_bs.h
ClearTokenInfo(tokenInfoStart, bsLoopNum * (tl_->K + 1));
```

ref 不是 vLLM 在线动态 batching。`cann-recipes-infer/models/deepseek_v4/ref/modeling_deepseek.py` 在模型初始化时从环境变量 `BATCH_SIZE` 得到 `self.batch_size`，随后用它创建固定 `micro_batch_size` 的 `ScheduleContext`，并在固定 decode 流程中反复执行。因此 ref 的 Window 物理容量和每次调用的逻辑 BS 保持一致，没有经历在线服务中的 prefill/decode/dummy BS 切换。

```python
# cann-recipes-infer/models/deepseek_v4/ref/modeling_deepseek.py
self.batch_size = int(os.getenv("BATCH_SIZE", "1"))

context_holder = torch_npu._afd.create_schedule_context_holder(
    # Other arguments omitted.
    micro_batch_size=(
        self.batch_size * self.spec_len
        // self.attn_dies // micro_batch_number
    ),
)
```

当前已验证的规避方案是保持逻辑 BS 固定为 Window 容量：A2F 输入、active mask 和 combine scales 都 pad 到 256，combine 后只截取真实 token 行。该修改已经通过短请求、长请求、重复请求和开启 prefix caching 的服务验证。它说明根因与动态逻辑 BS 复用有关，而不是 prefix cache 或请求内容。

向算子上游需要确认的问题是：

1. 同一个 `ScheduleContext` 和固定物理 Window 是否允许 `expert_scales.shape[0]` 在调用间变化？
2. 若允许，算子 tiling/cache key 和 token-info 清理是否都应使用本次真实 BS？
3. 若不允许，接口文档是否应明确要求逻辑 BS 始终等于初始化的 `micro_batch_size`？
