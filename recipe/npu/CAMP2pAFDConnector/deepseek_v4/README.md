# DeepSeek-V4 A8F8 eager/U1 AFD validation

This recipe is the pinned correctness baseline for one Atlas A3 node with 16
logical NPUs:

- Attention: NPU 0-7, DP8/TP1
- FFN: NPU 8-15, DP8/TP1/EP8
- CAMP2P: A8F8, one stage, eager execution
- MTP, DBO, graph execution, PD, sequence parallelism, and Attention-side gate
  computation are disabled

The launchers source `tools/dsv4/activate_runtime.sh`, so they require the
pinned CANN 9.0.1 and `afd-v026` environment documented there. The AFD plugin
must already be installed with Ascend ops by `tools/dsv4/install_plugin.sh`.

Run one complete validation with automatic cleanup:

```bash
source tools/dsv4/activate_runtime.sh
python recipe/npu/CAMP2pAFDConnector/deepseek_v4/run_validation.py \
  --output-dir /mnt/workspace/validation/dsv4_afd_m4_$(date +%Y%m%d_%H%M%S)
```

By default the runner performs two cold starts. Each cycle compares three
rounds of the ten Milestone 0 prompts token-by-token and tests batch sizes
1/8/32 for successful, structurally complete responses. Batch token equality
against the single-request golden is recorded but is not a capacity gate. The
first cycle idles for 1800 seconds and repeats a golden request.
Shutdown always signals Attention before FFN and captures `npu-smi info` after
cleanup.

The next correctness gate keeps U1 and enables decode-only ACL graphs:

```bash
source tools/dsv4/activate_runtime.sh
python recipe/npu/CAMP2pAFDConnector/deepseek_v4/run_validation.py \
  --execution-mode full-decode-only \
  --output-dir /mnt/workspace/validation/dsv4_afd_graph_u1_$(date +%Y%m%d_%H%M%S)
```

This selects `FULL_DECODE_ONLY` with capture sizes 1/2/4/8. DBO remains
disabled. Collect Attention and FFN traces in a separate one-cycle run so the
profiler does not perturb the correctness comparison:

```bash
python recipe/npu/CAMP2pAFDConnector/deepseek_v4/run_validation.py \
  --execution-mode full-decode-only --profile \
  --cycles 1 --idle-seconds 0 --rounds 1 --batch-sizes 1 \
  --output-dir /mnt/workspace/validation/dsv4_afd_graph_u1_profile_$(date +%Y%m%d_%H%M%S)
```

Both role profilers use `TORCH_PROFILER_WITH_STACK=0`; their schedule is wait
2, warmup 1, active 10, repeat 1, with no skipped steps.

For a startup smoke test while developing the recipe:

```bash
python recipe/npu/CAMP2pAFDConnector/deepseek_v4/run_validation.py \
  --output-dir /mnt/workspace/validation/dsv4_afd_m4_smoke \
  --cycles 1 --idle-seconds 0 --rounds 1 --batch-sizes 1
```

The role launchers can also be run directly. Start `afd_ffn.sh` first, then
`afd_attention.sh`. Override `API_PORT`, `AFD_PORT`, `HCCL_IF_IP`, or visible
devices through environment variables when the defaults conflict with another
service. The scheduler defaults match Milestone 0 (`1024` batched tokens and
`8` sequences); use `MAX_NUM_BATCHED_TOKENS` and `MAX_NUM_SEQS` only for
separate capacity runs.
