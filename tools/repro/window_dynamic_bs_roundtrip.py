"""Run a minimal four-operator Window AFD round trip on two NPUs.

Rank 0 is one FFN worker and rank 1 is one Attention worker. No model or vLLM
scheduler is involved. Use --mode dynamic for the pre-fix input contract and
--mode fixed for the fixed-capacity workaround.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch_npu


CAPACITY = 256
HIDDEN_SIZE = 7168
ROUTED_TOPK = 6
SELECTED_EXPERT_NUM = ROUTED_TOPK + 1
BATCH_SEQUENCE = (8, 1, 8)
WORLD_SIZE = 2
FFN_RANK = 0
ATTENTION_RANK = 1
ALIGNMENT = 512
QUANT_MODE = 2


def align_up(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def get_window_size(rank: int) -> int:
    if rank == ATTENTION_RANK:
        info = align_up(4 * CAPACITY * SELECTED_EXPERT_NUM)
        data = 2 * CAPACITY * SELECTED_EXPERT_NUM * HIDDEN_SIZE
        return info + data
    info = align_up(4 * (CAPACITY * SELECTED_EXPERT_NUM + 2))
    data = align_up(HIDDEN_SIZE + 4) * CAPACITY * SELECTED_EXPERT_NUM
    return info + data


def create_context(rank: int, window: torch.Tensor):
    common = {
        "session_num": 1,
        "micro_batch_num": 1,
        "micro_batch_size": CAPACITY,
        "selected_expert_num": SELECTED_EXPERT_NUM,
        "expert_num": SELECTED_EXPERT_NUM,
        "attn_to_ffn_token_size": align_up(HIDDEN_SIZE + 4),
        "ffn_to_attn_token_size": HIDDEN_SIZE * 2,
    }
    if rank == ATTENTION_RANK:
        return torch_npu._afd.create_schedule_context_holder(
            schedule_mode=1,
            attention_window=window.data_ptr(),
            attention_window_size=window.numel() * window.element_size(),
            **common,
        )
    return torch_npu._afd.create_schedule_context_holder(
        schedule_mode=0,
        ffn_window=window.data_ptr(),
        ffn_window_size=window.numel() * window.element_size(),
        **common,
    )


def run_attention(
    mode: str,
    group_name: str,
    context: torch.Tensor,
    window: torch.Tensor,
) -> bool:
    device = window.device
    hidden_row = torch.arange(HIDDEN_SIZE, dtype=torch.float32)
    hidden_row = hidden_row.remainder_(31).sub_(15).div_(32).to(torch.bfloat16)
    hidden_row = hidden_row.to(device)
    expert_row = torch.arange(ROUTED_TOPK, dtype=torch.int32, device=device)
    scale_row = torch.zeros(ROUTED_TOPK, dtype=torch.float32, device=device)
    scale_row[0] = 1.0

    expert_rank_table = torch.zeros(
        (1, SELECTED_EXPERT_NUM, 3), dtype=torch.int32, device=device
    )
    expert_rank_table[0, :, 0] = 1
    expert_rank_table[0, :, 1] = FFN_RANK
    expert_rank_table[0, :, 2] = torch.arange(
        SELECTED_EXPERT_NUM, dtype=torch.int32, device=device
    )

    info_bytes = 4 * CAPACITY * SELECTED_EXPERT_NUM
    flags = window[:info_bytes].view(torch.int32).reshape(
        CAPACITY, SELECTED_EXPERT_NUM
    )
    session_id = torch.zeros(1, dtype=torch.int32, device=device)
    micro_batch_id = torch.zeros(1, dtype=torch.int32, device=device)
    layer_id = torch.zeros(1, dtype=torch.int32, device=device)
    failed = False

    for call_id, batch_size in enumerate(BATCH_SEQUENCE, start=1):
        if mode == "dynamic":
            x = torch.zeros(
                (1, CAPACITY, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
            )
            x[0, :batch_size] = hidden_row
            expert_ids = torch.zeros(
                (1, CAPACITY, ROUTED_TOPK), dtype=torch.int32, device=device
            )
            expert_ids[0, :batch_size] = expert_row
            active_mask = torch.zeros(
                (1, CAPACITY), dtype=torch.bool, device=device
            )
            active_mask[0, :batch_size] = True
            combine_scales = scale_row.repeat(batch_size, 1)
        else:
            x = hidden_row.reshape(1, 1, HIDDEN_SIZE).repeat(1, CAPACITY, 1)
            expert_ids = expert_row.reshape(1, 1, ROUTED_TOPK).repeat(
                1, CAPACITY, 1
            )
            active_mask = torch.ones(
                (1, CAPACITY), dtype=torch.bool, device=device
            )
            combine_scales = torch.zeros(
                (CAPACITY, ROUTED_TOPK), dtype=torch.float32, device=device
            )
            combine_scales[:batch_size] = scale_row

        torch_npu.npu_attention_to_ffn(
            x,
            session_id,
            micro_batch_id,
            layer_id,
            expert_ids,
            expert_rank_table,
            group_name,
            WORLD_SIZE,
            [1, 1, 2 + CAPACITY * SELECTED_EXPERT_NUM],
            [1, 1, CAPACITY, SELECTED_EXPERT_NUM, align_up(HIDDEN_SIZE + 4)],
            [1, CAPACITY, SELECTED_EXPERT_NUM],
            ROUTED_TOPK,
            quant_mode=QUANT_MODE,
            sync_flag=0,
            ffn_start_rank_id=FFN_RANK,
            active_mask=active_mask,
        )
        output, _ = torch_npu.npu_attention_worker_combine(
            context,
            combine_scales,
            layer_id,
            HIDDEN_SIZE,
            token_dtype=1,
            need_schedule=1,
        )
        torch.npu.synchronize()

        expected_value = 0.25 * call_id
        max_error = (
            output[:batch_size].float() - expected_value
        ).abs().max().item()
        remaining_flags = flags[:batch_size].sum(dim=1).cpu().tolist()
        call_failed = max_error > 0.01 or any(remaining_flags)
        failed |= call_failed
        print(
            f"attention call={call_id} mode={mode} bs={batch_size} "
            f"max_abs_error={max_error:.6f} flags={remaining_flags} "
            f"status={'FAIL' if call_failed else 'PASS'}",
            flush=True,
        )
    return failed


def run_ffn(group_name: str, context: torch.Tensor) -> None:
    device = context.device
    attn_rank_table = torch.tensor(
        [ATTENTION_RANK], dtype=torch.int32, device=device
    )
    for call_id, _ in enumerate(BATCH_SEQUENCE, start=1):
        outputs = torch_npu.npu_ffn_worker_batching(
            context,
            SELECTED_EXPERT_NUM,
            [1, CAPACITY, SELECTED_EXPERT_NUM, HIDDEN_SIZE],
            token_dtype=2,
            need_schedule=1,
            layer_num=0,
        )
        (
            hidden_states,
            _,
            session_ids,
            micro_batch_ids,
            token_ids,
            expert_offsets,
            _,
            actual_token_num,
        ) = outputs
        ffn_output = torch.full(
            (hidden_states.shape[0], HIDDEN_SIZE),
            0.125 * call_id,
            dtype=torch.bfloat16,
            device=device,
        )
        torch_npu.npu_ffn_to_attention(
            ffn_output,
            session_ids,
            micro_batch_ids,
            token_ids,
            expert_offsets,
            actual_token_num.reshape(-1),
            group_name,
            WORLD_SIZE,
            [1, CAPACITY, SELECTED_EXPERT_NUM],
            [1, CAPACITY, SELECTED_EXPERT_NUM, HIDDEN_SIZE],
            attn_rank_table=attn_rank_table,
        )
        torch.npu.synchronize()
        print(
            f"ffn call={call_id} actual_token_num={actual_token_num.item()}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dynamic", "fixed"), default="dynamic")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != WORLD_SIZE:
        raise RuntimeError("This test requires exactly two processes")
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    control_group = dist.new_group(ranks=[FFN_RANK, ATTENTION_RANK], backend="gloo")

    process_group = dist.distributed_c10d._get_default_group()
    backend = process_group._get_backend(torch.device("npu"))
    group_name = str(backend.get_hccl_comm_name(rank))
    backend._window_register_and_exchange(get_window_size(rank), [1 - rank])
    window = backend._get_window_mem()
    holder = create_context(rank, window)
    context = holder.get_schedule_context_tensor()

    failed = False
    if rank == ATTENTION_RANK:
        failed = run_attention(args.mode, group_name, context, window)
    else:
        run_ffn(group_name, context)

    dist.barrier(group=control_group)
    holder.stop_schedule()
    dist.destroy_process_group(control_group)
    dist.destroy_process_group()
    if rank == ATTENTION_RANK:
        print(
            f"RESULT: mode={args.mode} {'FAIL' if failed else 'PASS'}",
            flush=True,
        )
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
