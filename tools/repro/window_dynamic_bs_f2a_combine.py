"""Test dynamic BS using only FfnToAttention and AttentionWorkerCombine.

Run with two NPUs. Rank 0 synthesizes FFN batching metadata and sends constant
BF16 rows; rank 1 combines them through one fixed-capacity Attention Window.
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


def align_up(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def attention_window_size() -> int:
    info = align_up(4 * CAPACITY * SELECTED_EXPERT_NUM)
    data = 2 * CAPACITY * SELECTED_EXPERT_NUM * HIDDEN_SIZE
    return info + data


def run_ffn(mode: str, group_name: str, device: torch.device) -> None:
    row_count = CAPACITY * SELECTED_EXPERT_NUM
    session_ids = torch.zeros(row_count, dtype=torch.int32, device=device)
    micro_batch_ids = torch.zeros(row_count, dtype=torch.int32, device=device)
    token_ids = torch.arange(CAPACITY, dtype=torch.int32, device=device)
    token_ids = token_ids.repeat_interleave(SELECTED_EXPERT_NUM)
    expert_offsets = torch.arange(
        SELECTED_EXPERT_NUM, dtype=torch.int32, device=device
    ).repeat(CAPACITY)
    attn_rank_table = torch.tensor(
        [ATTENTION_RANK], dtype=torch.int32, device=device
    )

    for call_id, batch_size in enumerate(BATCH_SEQUENCE, start=1):
        active_batch = CAPACITY if mode == "fixed" else batch_size
        actual_token_num = torch.tensor(
            [active_batch * SELECTED_EXPERT_NUM],
            dtype=torch.int64,
            device=device,
        )
        ffn_output = torch.full(
            (row_count, HIDDEN_SIZE),
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
            actual_token_num,
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


def run_attention(
    mode: str,
    context: torch.Tensor,
    window: torch.Tensor,
) -> bool:
    device = window.device
    scale_row = torch.zeros(ROUTED_TOPK, dtype=torch.float32, device=device)
    scale_row[0] = 1.0
    layer_id = torch.zeros(1, dtype=torch.int32, device=device)
    info_bytes = 4 * CAPACITY * SELECTED_EXPERT_NUM
    flags = window[:info_bytes].view(torch.int32).reshape(
        CAPACITY, SELECTED_EXPERT_NUM
    )
    issue_reproduced = False

    for call_id, batch_size in enumerate(BATCH_SEQUENCE, start=1):
        combine_batch = CAPACITY if mode == "fixed" else batch_size
        scales = scale_row.repeat(combine_batch, 1)
        output, _ = torch_npu.npu_attention_worker_combine(
            context,
            scales,
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
        issue_reproduced |= call_failed
        print(
            f"attention call={call_id} mode={mode} bs={batch_size} "
            f"max_abs_error={max_error:.6f} flags={remaining_flags} "
            f"status={'STALE' if call_failed else 'PASS'}",
            flush=True,
        )
    return issue_reproduced


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dynamic", "fixed"), default="dynamic")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != WORLD_SIZE:
        raise RuntimeError("This test requires exactly two processes")
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")
    control_group = dist.new_group(ranks=[FFN_RANK, ATTENTION_RANK], backend="gloo")

    process_group = dist.distributed_c10d._get_default_group()
    backend = process_group._get_backend(torch.device("npu"))
    group_name = str(backend.get_hccl_comm_name(rank))
    size = attention_window_size()
    backend._window_register_and_exchange(size, [1 - rank])
    window = backend._get_window_mem()

    holder = None
    issue_reproduced = False
    if rank == ATTENTION_RANK:
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
            attention_window_size=window.numel() * window.element_size(),
        )
        context = holder.get_schedule_context_tensor()
        issue_reproduced = run_attention(args.mode, context, window)
    else:
        run_ffn(args.mode, group_name, device)

    dist.barrier(group=control_group)
    if holder is not None:
        holder.stop_schedule()
    dist.destroy_process_group(control_group)
    dist.destroy_process_group()

    if rank == ATTENTION_RANK:
        expected_failure = args.mode == "dynamic"
        if issue_reproduced and expected_failure:
            result = "ISSUE_REPRODUCED"
        elif not issue_reproduced and not expected_failure:
            result = "PASS"
        elif issue_reproduced:
            result = "UNEXPECTED_FIXED_MODE_FAILURE"
        else:
            result = "ISSUE_NOT_REPRODUCED"
        print(f"RESULT: mode={args.mode} {result}", flush=True)
        return int(issue_reproduced != expected_failure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
