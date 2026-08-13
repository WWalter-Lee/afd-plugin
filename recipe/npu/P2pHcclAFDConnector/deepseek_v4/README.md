# DeepSeek-V4 A8F8 HCCL P2P validation

This recipe validates Attention/FFN disaggregation through standard
`torch.distributed.send` and `torch.distributed.recv` calls backed by HCCL. It
does not use `afd_camp2p_send_attn_output`, `afd_ascend.a2e`, or
`afd_ascend.e2a`.

Validated first-stage topology and execution boundary:

- Attention: NPU 0-7, DP8/TP1;
- FFN: NPU 8-15, DP8/TP1/EP8;
- A8F8 one-to-one rank mapping;
- eager U1 or eager U2;
- MTP, graph execution, PD, sequence parallelism, and Attention-side gate are
  disabled.

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

The HCCL connector currently rejects graph execution and unequal Attention/FFN
rank counts. The shared validator records the selected connector in
`runtime.json` and preserves the same golden, lifecycle, fatal-log, and NPU
cleanup gates used by the CAMP2P baseline.

The data path uses blocking HCCL point-to-point operations. Under U2 the
Attention scheduler switches stages after receiving the matching FFN output,
which keeps both stage groups aligned with the FFN layer-major receive loop.
Concurrent batch token exact counts are diagnostic; the batch gate checks
request structure, while the serial 30-request golden gate checks deterministic
token equality.
