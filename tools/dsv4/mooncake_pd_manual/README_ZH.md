# DeepSeek-V4 M9 Mooncake PD 双机手工管理脚本

该目录把安装指南中的双机命令收敛为一个入口：

```bash
bash pd.sh <install|check|start|status|validate|stop|collect> config.env
```

## 1. 准备三个配置

在 A3-P 创建 Prefill 配置：

```bash
bash pd.sh init /mnt/workspace/pd-prefill.env
vi /mnt/workspace/pd-prefill.env
```

设置 `NODE_ROLE=prefill`。在 A3-D 创建 Decode 配置并设置
`NODE_ROLE=decode`。Proxy 可以放在任一节点，创建第三个配置并设置
`NODE_ROLE=proxy`。

三个配置中的 `PREFILL_IP`、`DECODE_IP`、源码路径、wheel 路径和
`AFD_PD_COMMIT` 必须完全一致。`AFD_PD_COMMIT` 必须是已经提交并推送的 M9
commit，不能填写分支名。

## 2. 安装和预检

在 A3-P 和 A3-D 分别执行：

```bash
bash pd.sh install /mnt/workspace/pd-prefill.env
bash pd.sh check /mnt/workspace/pd-prefill.env
```

Decode 节点将配置文件换成 `pd-decode.env`。默认不执行 `sudo apt`；系统依赖
尚未安装时，先手工安装，或者显式设置 `INSTALL_SYSTEM_PACKAGES=1`。

`check` 默认执行本机两 NPU、2 MiB、两轮 Mooncake round-trip。只在已经有
独立证据且需要快速重跑时设置 `RUN_LOCAL_ROUNDTRIP=0`。

Proxy 配置只需执行 `check`。

## 3. 启动和验证

按以下顺序在对应节点执行：

```bash
# A3-P
bash pd.sh start /mnt/workspace/pd-prefill.env

# A3-D；脚本会紧邻启动 FFN 和 Attention，并等待 8 个 FFN loop
bash pd.sh start /mnt/workspace/pd-decode.env

# Proxy 所在节点
bash pd.sh start /mnt/workspace/pd-proxy.env
bash pd.sh validate /mnt/workspace/pd-proxy.env
```

随时检查：

```bash
bash pd.sh status /mnt/workspace/pd-prefill.env
bash pd.sh status /mnt/workspace/pd-decode.env
bash pd.sh status /mnt/workspace/pd-proxy.env
```

`validate` 通过 Proxy 执行 smoke、10 条 prompt x 3 轮、batch 1/8/32 和请求
取消恢复。它不能读取远端 Decode 日志，因此最终结论还要求 Decode 输出件中
存在 `KV cache transfer ... remote_session_id` 成功记录。

## 4. 停止顺序

```bash
bash pd.sh stop /mnt/workspace/pd-proxy.env
bash pd.sh stop /mnt/workspace/pd-decode.env
bash pd.sh stop /mnt/workspace/pd-prefill.env
```

脚本只停止自身 PID 文件记录、且命令行能够识别的 process group。默认不会
发送 KILL；TERM 超时时先检查进程，确认需要后再设置 `FORCE_KILL=1`。

## 5. 生成要回传的小输出件

三个角色分别执行：

```bash
bash pd.sh collect /mnt/workspace/pd-prefill.env
bash pd.sh collect /mnt/workspace/pd-decode.env
bash pd.sh collect /mnt/workspace/pd-proxy.env
```

每次输出两个文件：

```text
dsv4-m9-pd-<role>-<timestamp>.tar.gz
dsv4-m9-pd-<role>-<timestamp>.tar.gz.sha256
```

默认每个压缩包最大 2 MiB，三个角色合计不会超过约 6 MiB。内容只有：

- 三个源码 commit、关键 Python 包版本和 Mooncake wheel SHA256；
- `status`、PID、监听端口和 `npu-smi info`；
- runtime check、本地 round-trip 结果；
- 每个角色日志最后 256 KiB、最近 50 条 KV transfer 证据和最近 200 条
  fatal marker；
- Proxy 的 smoke、golden、batch、取消恢复结果。

不会包含完整日志、模型、wheel、profiler、core dump、完整环境变量或 API key。
将三个 `.tar.gz` 和对应 `.sha256` 发回即可；通常实际总大小会明显低于上限。
