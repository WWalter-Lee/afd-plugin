"""Check AttentionWorkerCombine reuse with changing logical batch sizes.

Run this script on one Ascend NPU. It intentionally reuses one fixed-capacity
Attention Window while changing only expert_scales.shape[0]: 8 -> 1 -> 8.
It is a combine-only control, not a reproduction of the complete AFD path.
"""

from __future__ import annotations

import sys

import torch
import torch_npu


DEVICE = "npu:0"
CAPACITY = 256
HIDDEN_SIZE = 7168
ROUTED_TOPK = 6
SELECTED_EXPERT_NUM = ROUTED_TOPK + 1  # Routed experts plus one shared expert.
BATCH_SEQUENCE = (8, 1, 8)
ALIGNMENT = 512
TOKEN_DTYPE = 1  # BF16, as defined by AttentionWorkerCombine.


def align_up(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def main() -> int:
    torch.npu.set_device(DEVICE)

    token_info_bytes = align_up(4 * CAPACITY * SELECTED_EXPERT_NUM)
    token_data_bytes = 2 * CAPACITY * SELECTED_EXPERT_NUM * HIDDEN_SIZE
    window_size = token_info_bytes + token_data_bytes
    window = torch.empty(window_size, dtype=torch.uint8, device=DEVICE)

    holder = torch_npu._afd.create_schedule_context_holder(
        schedule_mode=1,
        session_num=1,
        micro_batch_num=1,
        micro_batch_size=CAPACITY,
        selected_expert_num=SELECTED_EXPERT_NUM,
        expert_num=SELECTED_EXPERT_NUM,
        attn_to_ffn_token_size=HIDDEN_SIZE * 2,
        ffn_to_attn_token_size=HIDDEN_SIZE * 2,
        attention_window=window.data_ptr(),
        attention_window_size=window_size,
    )
    schedule_context = holder.get_schedule_context_tensor()

    flags = window[: 4 * CAPACITY * SELECTED_EXPERT_NUM].view(
        torch.int32
    ).reshape(CAPACITY, SELECTED_EXPERT_NUM)
    token_data = window[
        token_info_bytes:token_info_bytes + token_data_bytes
    ].view(torch.bfloat16).reshape(
        CAPACITY,
        SELECTED_EXPERT_NUM,
        HIDDEN_SIZE,
    )

    max_batch = max(BATCH_SEQUENCE)
    values = torch.arange(
        max_batch * SELECTED_EXPERT_NUM * HIDDEN_SIZE,
        dtype=torch.float32,
    ).remainder_(97).sub_(48).div_(64).to(torch.bfloat16).to(DEVICE)
    values = values.reshape(max_batch, SELECTED_EXPERT_NUM, HIDDEN_SIZE)
    scales = torch.arange(
        1,
        max_batch * ROUTED_TOPK + 1,
        dtype=torch.float32,
    ).reshape(max_batch, ROUTED_TOPK)
    scales = (scales / scales.sum(dim=1, keepdim=True)).to(DEVICE)
    layer_id = torch.zeros(1, dtype=torch.int32, device=DEVICE)

    failed = False
    for call_id, batch_size in enumerate(BATCH_SEQUENCE, start=1):
        token_data[:batch_size].copy_(values[:batch_size])
        flags[:batch_size].fill_(1)

        output, _ = torch_npu.npu_attention_worker_combine(
            schedule_context,
            scales[:batch_size].contiguous(),
            layer_id,
            HIDDEN_SIZE,
            token_dtype=TOKEN_DTYPE,
            need_schedule=1,
        )
        torch.npu.synchronize()

        expected = (
            values[:batch_size, :ROUTED_TOPK].float()
            * scales[:batch_size, :, None]
        ).sum(dim=1)
        expected.add_(values[:batch_size, ROUTED_TOPK].float())
        max_error = (output.float() - expected).abs().max().item()
        remaining_flags = flags[:batch_size].sum(dim=1).cpu().tolist()
        call_failed = max_error > 0.02 or any(remaining_flags)
        failed |= call_failed
        print(
            f"call={call_id} bs={batch_size} max_abs_error={max_error:.6f} "
            f"remaining_flag_sums={remaining_flags} "
            f"status={'FAIL' if call_failed else 'PASS'}",
            flush=True,
        )

    holder.stop_schedule()
    if failed:
        print("RESULT: dynamic-BS issue reproduced", flush=True)
        return 1
    print("RESULT: not reproduced by the combine-only test", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
