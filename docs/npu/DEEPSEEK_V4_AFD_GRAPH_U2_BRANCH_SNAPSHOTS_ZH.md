# DeepSeek V4 AFD Graph/U2 分支快照记录

## 1. 记录目的

本文记录 2026-09-01 冻结的两个远端验证分支。两个分支严格线性，三流验证分支完整包含
V1；当前开发工作树中的其他未提交改动不属于这两个快照。

| 快照 | 远端分支 | 固定提交 | 语义 |
|---|---|---|---|
| V1 混合 DAG | `feat/dsv4-afd-graph-u2-hybrid-v1` | `891e794885ccee2d498dcc1b80275a4d140f2cc2` | Attention Graph 使用 side compute，send/recv 留在 parent stream；FFN Graph receive 留在 parent stream，并逐层 join 两个 stage 的 send |
| 三流实现基座 | 同下方全开分支的父提交 | `a4096b591979e29280aa54cb36aac1c5a277017e` | 增加 Attention 三流、FFN 独立 recv、FFN cross-layer 三个实验开关，源码默认关闭 |
| 三项默认全开预设 | `feat/dsv4-afd-graph-u2-multistream-all-on-v1` | `477df01e73b9a752e9e552a58d17f28a5398e252` | 在三流实现基座上将三个实验开关的 connector 与性能脚本默认值统一改为开启 |

提交关系：

```text
6c63696 -> 891e794 (V1) -> a4096b5 (三流实现，默认关闭)
                             -> 477df01 (验证预设，默认全开)
```

## 2. 全开预设

未显式传参时，全开分支等价于：

```text
AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=1
AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=1
AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=1
```

`run_performance.py` 对应的默认参数为：

```text
--graph-u2-attention-three-stream on
--graph-u2-ffn-recv-stream on
--graph-u2-ffn-cross-layer on
```

需要回退到同源码 V1 物理流映射时，必须显式将三项同时设为 `0`/`off`。其中
`cross-layer=on, recv-stream=off` 是非法组合，会在启动参数检查或 connector 构造时失败。

这些开关只改变已满足 Graph/U2、双 micro-batch、compute-overlap 等既有门控后的物理流映射
与事件 join；不会增加初始化时创建的物理 stream 数量，也不改变 U1/eager 的现有调度路径。

## 3. 验证边界

该全开分支是后续性能定位分支，不是生产默认策略。既有 CANN 9.0.0 对齐 C32 三轮结果为：

| 配置 | 平均吞吐 | 相对 V1 |
|---|---:|---:|
| V1 | 148.820 token/s | 基线 |
| 三项全开 | 134.829 token/s | -9.401% |

因此历史开发报告中“三项生产默认关闭”的结论保持不变。后续需要在固定 vLLM
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、vLLM-Ascend
`3da28f9414583d2d0b672a8f06d1fae142404bda`、CANN 9.0.0 和相同 workload 下重新采集
双侧 Profile，再决定是否逐项合入生产分支。

提交前的代码级门禁：connector Graph 相关用例 `31 passed`，性能 recipe 开关相关用例
`3 passed`，修改的生产 Python 文件通过 `py_compile`。真实 NPU smoke、吞吐和双侧 Profile
不由这些单元测试替代。
