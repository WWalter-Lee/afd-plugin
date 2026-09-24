# DeepSeek-V4 NPU recipe common helpers

This directory contains only connector-shared recipe infrastructure:

- `activate_role_runtime.sh` activates the pinned CANN, vLLM, and
  vLLM-Ascend runtime used by both connector recipes;
- `run_validation.py` owns process lifecycle, golden requests, cleanup, and
  shared topology bookkeeping, while selecting the launcher directory from
  the explicitly required connector;
- `validate_golden.py` compares deterministic token IDs through an OpenAI
  completion endpoint.

Transport implementation and feature switches must remain in their owning
connector modules and recipe directories. In particular, HCCL P2P, TP2, MTP,
Graph U2, and Mooncake PD launcher logic belongs under
`recipe/npu/P2pHcclAFDConnector/deepseek_v4/`; it must not be added to the
CAMP2p launchers.
