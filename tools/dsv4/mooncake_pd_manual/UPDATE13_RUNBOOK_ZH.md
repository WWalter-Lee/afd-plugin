# Mooncake PD + AFD update13 外部拓扑操作顺序

当前阶段采用本机优先策略：先完成全部计划功能代码和 F0-local，F0 暂不生成或
比较 golden。batch-invariant 与本手册中的双 A3 正确性步骤统一延后，不作为当前
`afd-plugin` 组合开发的前置条件。

update13 在 update12 的路径匹配 control 基础上保留两项后续外部验收能力：

1. Decode DP4/TP2，以及 eager/U2、Graph/U1、Graph/U2、一 token MTP 的逐项
   control/AFD 配置入口；
2. 对交付的 vLLM-Ascend batch-invariant 两文件补丁、独立 venv 和自定义 OPP
   的严格校验与角色加载。

这些是代码和部署门禁，不代表组合实模 F0 已完成。TP2 full-draft Graph U2 + MTP
仍拒绝，多 speculative token 仍未支持。

## 1. 两个包的职责

```text
afd-plugin-dsv4-batch-invariant-dual-a3-20260828.tar.gz
afd-plugin-dsv4-mooncake-pd-update13-20260828.tar.gz
```

第一个包留到全部计划功能开发完成后，在两台 A3 上安装/验证 vLLM-Ascend 补丁；
当前阶段只使用第二个包覆盖 `/data/z00569729/code/afd-plugin` 的 M9 源码、脚本和
文档。不要把 vLLM-Ascend
补丁文件直接解压到 afd-plugin。

## 2. 先覆盖 update13

A3-P 和 A3-D 都执行：

```bash
cd /data/z00569729/code
tar -xzf /data/z00569729/packages/afd-plugin-dsv4-mooncake-pd-update13-20260828.tar.gz

git -c safe.directory=/data/z00569729/code/afd-plugin \
  -C /data/z00569729/code/afd-plugin rev-parse HEAD
```

HEAD 仍应为交付基线 `49bb4a1dda5f7a59dcfbb45ea36d3ad1b2b89193`；覆盖文件
显示为已修改/未跟踪是预期状态，`pd.sh` 只允许 manifest 中的这些路径。

## 3. 延后执行：独立验证 batch invariance

全部计划功能代码和 F0-local 完成后再严格执行：

```text
docs/npu/DEEPSEEK_V4_BATCH_INVARIANT_DUAL_A3_VALIDATION_GUIDE_ZH.md
```

届时两台机器各自两次 direct Prefill `10 x 3` 都通过后，再把 PD 六份配置统一切换为：

```bash
VENV_ROOT="/data/z00569729/code/.venvs/afd-v023-vllm-cann-batch-invariant"
VLLM_ASCEND_WORKTREE_MODE="batch_invariant_patch"
ENABLE_BATCH_INVARIANT="1"
BATCH_INVARIANT_OPP_ROOT="/data/z00569729/code/.ascend/custom-opp/batch-invariant-a3-1.0.0"
```

## 4. 延后执行：双 A3 F0-topology 与 F1 顺序

F0-topology 先验证每个组合能够启动、完成请求、执行真实数据路径并正常清理，暂不
要求 golden。全部组合功能完成后进入 F1：每一项先生成独立的 `pd_control`，再用
相同配置验证 `pd_afd`。F1 每项使用不同的 `RUN_ROOT` 和
`PD_CONTROL_GOLDEN_PATH`，不得复用前一项 golden。

| 顺序 | Decode 配置 |
|---|---|
| 1 | DP8/TP1、eager、U1、MTP off |
| 2 | DP4/TP2、eager、U1、MTP off |
| 3 | DP8/TP1、eager、U2、MTP off |
| 4 | DP8/TP1、Graph、U1、MTP off |
| 5 | DP8/TP1、Graph、U2、MTP off |
| 6 | DP8/TP1、eager/U1 + 一 token MTP |
| 7 | 其余 TP2、Graph、U2、MTP 组合，每次只增加一个变量 |

配置变量映射：

```bash
DECODE_DP_SIZE="8"                 # TP2 改 4
DECODE_TP_SIZE="1"                 # TP2 改 2
DECODE_EXECUTION_MODE="eager"      # Graph 改 full-decode-only
DECODE_U_BATCHES="1"               # U2 改 2
DECODE_ENABLE_MTP="0"              # MTP 改 1
DECODE_MTP_DRAFT_EXECUTION="eager" # full draft 才改 graph
```

每项仍按 Prefill、Decode FFN/Attention、Proxy 的顺序启动。F0-topology 只运行
功能冒烟和生命周期；F1 才由 control 执行 `record-control`、AFD 执行 `validate`，
要求同路径稳定且 control 对 AFD `30/30`。当前结果和旧 direct native golden 的
差异只记录。

## 5. 回传

延后的 batch-invariant direct 阶段回传两台机器各一个不超过 2 MiB 的 support 包。
Mooncake 每个外部拓扑阶段只回传 Prefill、Decode、Proxy 的 `collect` 包和 `.sha256`。
优先回传前两项 TP1/TP2，后续组合无需等待上游旧 golden 问题关闭即可继续执行。
