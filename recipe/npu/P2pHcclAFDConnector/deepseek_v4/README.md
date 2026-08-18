# DeepSeek-V4 HCCL P2P validation

This recipe validates Attention/FFN disaggregation through standard
`torch.distributed.send` and `torch.distributed.recv` calls backed by HCCL. It
does not use `afd_camp2p_send_attn_output`, `afd_ascend.a2e`, or
`afd_ascend.e2a`.

Supported execution boundary:

- Attention: NPU 0-7, DP8/TP1;
- FFN: NPU 8-15, DP8/TP1/EP8;
- A8F8 one-to-one deployment and A2F1/A4F2 component topologies;
- integer-multiple topology contract `A >= F` and `A % F == 0`;
- eager U1 or eager U2, including the integer-multiple topologies above;
- `FULL_DECODE_ONLY` Graph U1 for equal Attention/FFN rank counts;
- Graph U2, Graph with unequal Attention/FFN ranks, MTP, PD, sequence
  parallelism, and Attention-side gate are disabled.

The current implementation deliberately remains synchronous: the data path
uses blocking `torch.distributed.send/recv` and retains its explicit NPU
synchronization boundary. It does not use `isend/irecv`, background transfer
threads, or an asynchronous custom HCCL op.

Under Graph U1, torch-npu lowers the hidden-state `send/recv` calls into the
ACL Graph. Input IDs remain on the one-shot HCCL side channel before capture or
replay, so no host metadata or dynamically sized ID message is placed in the
captured graph.

Run the Graph U1 correctness gate on the vLLM 0.23 + `rfc/vllm_cann` stack:

```bash
source tools/dsv4/activate_v023_vllm_cann_runtime.sh
python recipe/npu/P2pHcclAFDConnector/deepseek_v4/run_validation.py \
  --execution-mode full-decode-only --u-batches 1 \
  --golden /mnt/workspace/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json \
  --cycles 1 --idle-seconds 0 --rounds 3 --batch-sizes 1 8 32 \
  --output-dir /mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u1_<timestamp>
```

Run a U1 smoke validation:

```bash
source tools/dsv4/activate_runtime.sh
python recipe/npu/P2pHcclAFDConnector/deepseek_v4/run_validation.py \
  --cycles 1 --idle-seconds 0 --rounds 1 --batch-sizes 1 \
  --output-dir /mnt/workspace/validation/dsv4_afd_hccl_p2p_u1_smoke_$(date +%Y%m%d_%H%M%S)
```

Run the complete eager U2 correctness gate:

```bash
source tools/dsv4/activate_runtime.sh
python recipe/npu/P2pHcclAFDConnector/deepseek_v4/run_validation.py \
  --u-batches 2 \
  --dbo-decode-token-threshold 2 \
  --dbo-prefill-token-threshold 12 \
  --cycles 1 --idle-seconds 0 --rounds 3 --batch-sizes 1 8 32 \
  --output-dir /mnt/workspace/validation/dsv4_afd_hccl_p2p_u2_$(date +%Y%m%d_%H%M%S)
```

The HCCL connector rejects Graph U2, Graph with unequal rank counts, `A < F`,
and non-integer A/F ratios. The shared validator records the selected connector
in `runtime.json` and preserves the same golden, lifecycle, fatal-log, and NPU
cleanup gates used by the CAMP2P baseline.

The data path uses blocking HCCL point-to-point operations. Under U2 the
Attention scheduler switches stages after receiving the matching FFN output,
which keeps both stage groups aligned with the FFN layer-major receive loop.
Concurrent batch token exact counts are diagnostic; the batch gate checks
request structure, while the serial 30-request golden gate checks deterministic
token equality.

## A3-P4 performance reference

The performance runner uses the pinned Python environment directly and keeps
normal benchmark runs separate from profiler runs. Its default measurement
matrix is:

- A8F8, eager, standard HCCL send/recv;
- random input length 1024 and exact output length 128;
- concurrency 1, 8, and 32;
- three runs per concurrency;
- request rate `inf`, seed 1024, temperature 0, and EOS ignored;
- max batched tokens 1024 and max sequences 8, matching the validated service
  configuration;
- role-local HCCL base ports 51000 (Attention) and 52000 (FFN), avoiding the
  default allocator reaching or colliding at the 65536 port boundary;
- TTFT/TPOT/E2EL p50, p90, and p99;
- NPU utilization/HBM samples every two seconds.

The A3-P4 reference repeatability gate requires output-throughput CV to be no
more than 10% at every concurrency. Later performance claims must still exceed
the observed three-run range; passing this reference gate alone is not a
performance-benefit result.

Run the U1 and U2 performance references separately:

```bash
source tools/dsv4/activate_runtime.sh
python recipe/npu/P2pHcclAFDConnector/deepseek_v4/run_performance.py \
  --u-batches 1 \
  --output-dir /mnt/workspace/validation/dsv4_afd_a3_p4_hccl_u1_<timestamp>

python recipe/npu/P2pHcclAFDConnector/deepseek_v4/run_performance.py \
  --u-batches 2 \
  --dbo-decode-token-threshold 2 \
  --dbo-prefill-token-threshold 12 \
  --output-dir /mnt/workspace/validation/dsv4_afd_a3_p4_hccl_u2_<timestamp>
```

Collect profiler runs into different directories so profiler overhead never
enters the formal throughput results:

```bash
python recipe/npu/P2pHcclAFDConnector/deepseek_v4/run_performance.py \
  --u-batches 1 --profile \
  --output-dir /mnt/workspace/validation/dsv4_afd_a3_p4_hccl_u1_profile_<timestamp>

python recipe/npu/P2pHcclAFDConnector/deepseek_v4/run_performance.py \
  --u-batches 2 --profile \
  --dbo-decode-token-threshold 2 \
  --dbo-prefill-token-threshold 12 \
  --output-dir /mnt/workspace/validation/dsv4_afd_a3_p4_hccl_u2_profile_<timestamp>
```

The formal runner intentionally stays within the validated 4096-token service
limit. The 8K/32K and 128K capacity workloads require a separate max-model-len
and KV/HBM gate and are not mixed into this decode reference.

## A3-P5 blocking HCCL component gate

Run the component gate with the pinned runtime. It loads no model and validates
different peer token counts, int32 IDs, BF16 hidden states, output splitting,
two stages, two consecutive steps, and process-group close:

```bash
source tools/dsv4/activate_runtime.sh

python tools/dsv4/validate_hccl_p2p_roundtrip.py \
  --attention-devices 0,1 --ffn-devices 8 \
  --stages 2 --steps 2 --port 29841 \
  --output /mnt/workspace/validation/dsv4_afd_a3_p5_a2f1/summary.json

python tools/dsv4/validate_hccl_p2p_roundtrip.py \
  --attention-devices 0,1,2,3 --ffn-devices 8,9 \
  --stages 2 --steps 2 --port 29851 \
  --output /mnt/workspace/validation/dsv4_afd_a3_p5_a4f2/summary.json
```

Use `python -m pytest`, not the venv's `pytest` entry point: the entry point may
retain a stale shebang after the fixed environment is relocated.

Validated A3-P5 component evidence:

```text
/mnt/workspace/validation/dsv4_afd_a3_p5_hccl_a2f1_20260817_1055/summary.json
/mnt/workspace/validation/dsv4_afd_a3_p5_hccl_a4f2_20260817_1105/summary.json
```

## A3 unequal-topology model capacity result

A8F4 is a valid connector topology, but the fixed 64 GiB A3 runtime cannot
construct the FFN EP4 model: each FFN rank owns 64 experts and model loading
reaches about 60.62 GiB before a further 514 MiB allocation fails. An A10F5
capacity proxy also cannot run on the fixed stack because 256 experts are not
evenly divisible by EP5 and the upstream EPLB placement check rejects it.
These are model-capacity/upstream-placement gates, not HCCL topology failures.
The A8F4 full-model gate is therefore deferred to the higher-HBM A5 target.

```text
/mnt/workspace/validation/dsv4_afd_a3_p6_hccl_a8f4_u1_smoke_20260817_1120
/mnt/workspace/validation/dsv4_afd_a3_p6_hccl_a10f5_u1_smoke_20260817_1125
```

## A3-P7 synchronous optimization guard

The synchronous-only optimization caches each step/stage peer layout, reuses
FFN stage token metadata and the Ascend forward context across all layers, and
avoids NPU-side `min().item()`/`max().item()` host readback for
scheduler-provided input IDs. CPU/Mock validation still performs the ID
value-domain check. Per-layer execution only updates input IDs, AFD metadata,
and the MoE layer index; HCCL calls and synchronization remain blocking.

The stage-layout-only version passed A4F2 but its first non-profiler C32 guards
did not improve throughput. A CANN 9.0.1 profile confirmed that Attention
`aten::item/_local_scalar_dense` time fell from about 1211.681 ms to 5.498 ms,
then identified repeated per-layer FFN forward-context setup as the next host
cost. After caching that context per step/stage, 10/10 golden prompts matched
token-for-token.

The final A8F8 U1 C32 formal run uses 1024 input tokens, 128 exact output
tokens, 128 requests per repeat, and three repeats. Output throughput is
57.277/57.653/58.243 token/s, with a mean of 57.724 and CV of 0.689%. Compared
with the frozen P4 mean of 49.118 token/s, throughput improves by 17.521% and
p50 TPOT improves by 14.756%. This result covers C32 only; C1/C8, cold-service
repeatability, and the non-AFD fair-resource comparison remain open P7 gates.

```text
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_a4f2_20260817_1135/summary.json
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_u1_guard_c32_retry_20260817_1150/performance_summary.json
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_u1_profile_20260817_120056/performance_summary.json
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_context_cache_golden_20260817_1315/validation_summary.json
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_context_cache_c32_formal3_20260817_1345/performance_summary.json
```
