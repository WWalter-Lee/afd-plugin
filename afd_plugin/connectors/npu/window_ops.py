# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Thin Python boundary for the AFD Window communication operators."""

from __future__ import annotations

import torch


class WindowAFDDataPathNotReady(RuntimeError):
    """Compatibility exception retained for callers of the old placeholder."""


def attention_to_ffn(
    x: torch.Tensor,
    session_id: torch.Tensor,
    micro_batch_id: torch.Tensor,
    layer_id: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_rank_table: torch.Tensor,
    group: str,
    world_size: int,
    ffn_token_info_table_shape: list[int],
    ffn_token_data_shape: list[int],
    attn_token_info_table_shape: list[int],
    moe_expert_num: int,
    *,
    scales: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
    quant_mode: int = 0,
    sync_flag: int = 0,
    ffn_start_rank_id: int = 0,
) -> None:
    import torch_npu

    torch_npu.npu_attention_to_ffn(
        x,
        session_id,
        micro_batch_id,
        layer_id,
        expert_ids,
        expert_rank_table,
        group,
        world_size,
        ffn_token_info_table_shape,
        ffn_token_data_shape,
        attn_token_info_table_shape,
        moe_expert_num,
        scales=scales,
        active_mask=active_mask,
        quant_mode=quant_mode,
        sync_flag=sync_flag,
        ffn_start_rank_id=ffn_start_rank_id,
    )


def ffn_worker_batching(
    schedule_context: torch.Tensor,
    expert_num: int,
    max_out_shape: list[int],
    *,
    token_dtype: int = 0,
    need_schedule: int = 0,
    layer_num: int = 0,
) -> tuple[torch.Tensor, ...]:
    import torch_npu

    return torch_npu.npu_ffn_worker_batching(
        schedule_context,
        expert_num,
        max_out_shape,
        token_dtype=token_dtype,
        need_schedule=need_schedule,
        layer_num=layer_num,
    )


def ffn_worker_scheduler(
    schedule_context: torch.Tensor,
    *,
    sync_group_size: int,
) -> None:
    import torch_npu

    torch_npu.ffn_worker_scheduler_(
        schedule_context,
        sync_group_size=sync_group_size,
    )


def ffn_to_attention(
    x: torch.Tensor,
    session_ids: torch.Tensor,
    micro_batch_ids: torch.Tensor,
    token_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    actual_token_num: torch.Tensor,
    group: str,
    world_size: int,
    token_info_table_shape: list[int],
    token_data_shape: list[int],
    *,
    attn_rank_table: torch.Tensor | None = None,
) -> None:
    import torch_npu

    torch_npu.npu_ffn_to_attention(
        x,
        session_ids,
        micro_batch_ids,
        token_ids,
        expert_offsets,
        actual_token_num,
        group,
        world_size,
        token_info_table_shape,
        token_data_shape,
        attn_rank_table=attn_rank_table,
    )


def attention_worker_combine(
    schedule_context: torch.Tensor,
    expert_scales: torch.Tensor,
    layer_id: torch.Tensor,
    hidden_size: int,
    *,
    token_dtype: int = 0,
    need_schedule: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    return torch_npu.npu_attention_worker_combine(
        schedule_context,
        expert_scales,
        layer_id,
        hidden_size,
        token_dtype=token_dtype,
        need_schedule=need_schedule,
    )


__all__ = [
    "WindowAFDDataPathNotReady",
    "attention_to_ffn",
    "attention_worker_combine",
    "ffn_to_attention",
    "ffn_worker_batching",
    "ffn_worker_scheduler",
]
