#!/usr/bin/env python3
"""Validate blocking HCCL AFD transfers on physical NPUs."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


def _port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _token_counts(
    attention_size: int,
    *,
    step_idx: int,
    stage_idx: int,
) -> list[int]:
    return [
        2 + ((step_idx + stage_idx + attention_rank) % 4)
        for attention_rank in range(attention_size)
    ]


def _input_id_values(
    attention_rank: int,
    num_tokens: int,
    *,
    step_idx: int,
    stage_idx: int,
) -> list[int]:
    start = 1 + step_idx * 40 + stage_idx * 16 + attention_rank * 6
    values = list(range(start, start + num_tokens))
    if step_idx == 0 and stage_idx == 0 and attention_rank == 0:
        values[0] = -1
    return values


def _hidden_value(attention_rank: int, *, step_idx: int, stage_idx: int) -> int:
    return 100 * step_idx + 10 * stage_idx + attention_rank


def _mtp_token_counts(attention_size: int, *, step_idx: int) -> list[int]:
    return [
        1 + ((2 * step_idx + attention_rank) % 3)
        for attention_rank in range(attention_size)
    ]


def _mtp_hidden_value(attention_rank: int, *, step_idx: int) -> int:
    return 40 + 20 * step_idx + attention_rank


def _ffn_token_counts(counts: list[int], *, ffn_size: int) -> list[int]:
    ratio = len(counts) // ffn_size
    return [sum(counts[rank * ratio : (rank + 1) * ratio]) for rank in range(ffn_size)]


def _validate_graph_transport(
    *,
    connector,
    role: str,
    role_rank: int,
    attention_size: int,
    stages: int,
    step_idx: int,
) -> list[dict[str, object]]:
    import torch
    import torch.distributed as dist

    from afd_plugin.connectors.metadata import (
        AFDControlPayload,
        AFDDPMetadata,
        AFDTransferContext,
        AFDTransferMetadata,
    )

    counts_by_stage = {
        stage_idx: _token_counts(
            attention_size,
            step_idx=step_idx,
            stage_idx=stage_idx,
        )
        for stage_idx in range(stages)
    }
    control_payload = AFDControlPayload(
        dp_metadata_list={
            stage_idx: AFDDPMetadata(torch.tensor(counts, dtype=torch.int32))
            for stage_idx, counts in counts_by_stage.items()
        },
        is_graph_capturing=True,
        is_warmup=False,
    )
    if role == "attention":
        connector.control_plane.update_state_from_dp_metadata(control_payload)
        connector.control_plane.send_dp_metadata_list(control_payload)
    else:
        received_control = connector.control_plane.recv_dp_metadata_list()
        connector.control_plane.update_state_from_dp_metadata(received_control)

    static_hidden: dict[int, torch.Tensor] = {}
    returned_buffers: dict[int, torch.Tensor] = {}
    received_ids: dict[int, list[int]] = {}
    for stage_idx, counts in counts_by_stage.items():
        if role == "attention":
            num_tokens = counts[role_rank]
            input_ids = torch.tensor(
                _input_id_values(
                    role_rank,
                    num_tokens,
                    step_idx=step_idx,
                    stage_idx=stage_idx,
                ),
                dtype=torch.int32,
                device="npu:0",
            )
            connector.send_input_ids(input_ids, ubatch_idx=stage_idx)
            static_hidden[stage_idx] = torch.full(
                (num_tokens, 16),
                _hidden_value(role_rank, step_idx=step_idx, stage_idx=stage_idx),
                dtype=torch.bfloat16,
                device="npu:0",
            )
            returned_buffers[stage_idx] = torch.empty_like(static_hidden[stage_idx])
            continue

        first_attention_rank = role_rank * connector.ratio
        peer_counts = counts[
            first_attention_rank : first_attention_rank + connector.ratio
        ]
        ids = connector.recv_input_ids(sum(peer_counts), ubatch_idx=stage_idx)
        received_ids[stage_idx] = ids.cpu().tolist()

    dist.barrier(group=connector.p2p_pg)
    graph = torch.npu.NPUGraph()
    connector.is_graph_capturing = True
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        if role == "attention":
            for stage_idx, hidden in static_hidden.items():
                context = AFDTransferContext(
                    metadata=AFDTransferMetadata.create_attention_metadata(
                        layer_idx=1,
                        stage_idx=stage_idx,
                        seq_len=int(hidden.shape[0]),
                    ),
                )
                connector.send_attn_output(hidden, context)
                connector.recv_ffn_output(
                    returned_buffers[stage_idx],
                    ubatch_idx=stage_idx,
                )
        else:
            for stage_idx in counts_by_stage:
                payload = connector.recv_attn_output(
                    ubatch_idx=stage_idx,
                    layer_idx=1,
                )
                connector.send_ffn_output(
                    payload.hidden_states + 1,
                    payload.context,
                    ubatch_idx=stage_idx,
                )
    connector.is_graph_capturing = False
    torch.npu.synchronize()
    dist.barrier(group=connector.p2p_pg)

    checks: list[dict[str, object]] = []
    if role == "attention":
        for stage_idx, hidden in static_hidden.items():
            checks.append(
                {
                    "phase": "graph_capture",
                    "stage": stage_idx,
                    "tokens": int(hidden.shape[0]),
                    "captured": True,
                }
            )
    else:
        first_attention_rank = role_rank * connector.ratio
        for stage_idx, counts in counts_by_stage.items():
            expected_ids: list[int] = []
            for attention_rank in range(
                first_attention_rank,
                first_attention_rank + connector.ratio,
            ):
                expected_ids.extend(
                    _input_id_values(
                        attention_rank,
                        counts[attention_rank],
                        step_idx=step_idx,
                        stage_idx=stage_idx,
                    )
                )
            if received_ids[stage_idx] != expected_ids:
                raise AssertionError(
                    f"graph-external input IDs mismatch for ffn={role_rank} "
                    f"stage={stage_idx}"
                )
            checks.append(
                {
                    "phase": "graph_capture",
                    "stage": stage_idx,
                    "peer_tokens": counts[
                        first_attention_rank : first_attention_rank + connector.ratio
                    ],
                    "input_ids_external": True,
                    "fan_in_out": True,
                }
            )

    replay_step_idx = step_idx + 7
    if role == "attention":
        for stage_idx, hidden in static_hidden.items():
            hidden.fill_(
                _hidden_value(
                    role_rank,
                    step_idx=replay_step_idx,
                    stage_idx=stage_idx,
                )
            )
    dist.barrier(group=connector.p2p_pg)
    graph.replay()
    torch.npu.synchronize()
    dist.barrier(group=connector.p2p_pg)

    if role == "attention":
        for stage_idx, hidden in static_hidden.items():
            if not torch.equal(returned_buffers[stage_idx].cpu(), (hidden + 1).cpu()):
                raise AssertionError(
                    "graph replay round-trip mismatch for "
                    f"attention={role_rank} stage={stage_idx}"
                )
            checks.append(
                {
                    "phase": "graph_replay",
                    "stage": stage_idx,
                    "tokens": int(hidden.shape[0]),
                    "updated_input": True,
                    "roundtrip": True,
                }
            )
    else:
        checks.append(
            {
                "phase": "graph_replay",
                "stages": stages,
                "fan_in_out": True,
            }
        )
    return checks


def _validate_mtp_graph_transport(
    *,
    connector,
    role: str,
    role_rank: int,
    attention_size: int,
    ffn_size: int,
    stages: int,
    step_idx: int,
) -> list[dict[str, object]]:
    """Capture and replay the merged MTP phase on physical HCCL groups."""

    import torch
    import torch.distributed as dist

    from afd_plugin.connectors.metadata import (
        AFDControlPayload,
        AFDDPMetadata,
        AFDTransferContext,
        AFDTransferMetadata,
    )

    attention_peer_counts = [0] * attention_size
    for stage_idx in range(stages):
        stage_counts = _token_counts(
            attention_size,
            step_idx=step_idx,
            stage_idx=stage_idx,
        )
        for attention_rank, count in enumerate(stage_counts):
            attention_peer_counts[attention_rank] += count

    control_payload = AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(torch.tensor(attention_peer_counts, dtype=torch.int32))
        },
        is_graph_capturing=True,
        is_warmup=False,
    )
    if role == "attention":
        connector.control_plane.update_state_from_dp_metadata(control_payload)
        connector.control_plane.send_dp_metadata_list(control_payload)
    else:
        received_control = connector.control_plane.recv_dp_metadata_list()
        connector.control_plane.update_state_from_dp_metadata(received_control)

    static_hidden = None
    returned_buffer = None
    header = None
    if role == "attention":
        num_tokens = attention_peer_counts[role_rank]
        static_hidden = torch.full(
            (num_tokens, 16),
            _mtp_hidden_value(role_rank, step_idx=step_idx),
            dtype=torch.bfloat16,
            device="npu:0",
        )
        returned_buffer = torch.empty_like(static_hidden)

    dist.barrier(group=connector.p2p_pg)
    graph = torch.npu.NPUGraph()
    connector.is_graph_capturing = True
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        if role == "attention":
            assert static_hidden is not None
            assert returned_buffer is not None
            context = AFDTransferContext(
                metadata=AFDTransferMetadata.create_attention_metadata(
                    layer_idx=0,
                    stage_idx=0,
                    seq_len=int(static_hidden.shape[0]),
                    phase="mtp",
                    speculative_step=0,
                ),
            )
            connector.send_attn_output(
                static_hidden,
                context,
                num_tokens_across_dp=torch.tensor(
                    attention_peer_counts,
                    dtype=torch.int32,
                ),
            )
            connector.recv_ffn_output(
                returned_buffer,
                ubatch_idx=0,
                phase="mtp",
            )
        else:
            header = connector.recv_mtp_header_for_graph(
                stage_idx=0,
                attention_peer_counts=tuple(attention_peer_counts),
            )
            payload = connector.recv_attn_output(
                ubatch_idx=0,
                layer_idx=0,
                phase="mtp",
                speculative_step=header.speculative_step,
                num_tokens=header.num_tokens,
            )
            connector.send_ffn_output(
                payload.hidden_states + 2,
                payload.context,
                ubatch_idx=0,
            )
    connector.is_graph_capturing = False
    torch.npu.synchronize()
    dist.barrier(group=connector.p2p_pg)

    checks: list[dict[str, object]] = []
    if role == "attention":
        checks.append(
            {
                "phase": "mtp_graph_capture",
                "tokens": attention_peer_counts[role_rank],
                "peer_tokens": attention_peer_counts,
                "captured": True,
            }
        )
        assert static_hidden is not None
        static_hidden.fill_(_mtp_hidden_value(role_rank, step_idx=step_idx + 7))
    else:
        assert header is not None
        expected_ffn_counts = _ffn_token_counts(
            attention_peer_counts,
            ffn_size=ffn_size,
        )
        if header.num_tokens_across_dp.tolist() != expected_ffn_counts:
            raise AssertionError(
                "MTP graph FFN count projection mismatch: "
                f"{header.num_tokens_across_dp.tolist()} != {expected_ffn_counts}"
            )
        checks.append(
            {
                "phase": "mtp_graph_capture",
                "peer_tokens": attention_peer_counts[
                    role_rank * connector.ratio :
                    (role_rank + 1) * connector.ratio
                ],
                "header_fan_in": True,
                "captured": True,
            }
        )

    dist.barrier(group=connector.p2p_pg)
    graph.replay()
    torch.npu.synchronize()
    dist.barrier(group=connector.p2p_pg)

    if role == "attention":
        assert static_hidden is not None
        assert returned_buffer is not None
        if not torch.equal(returned_buffer.cpu(), (static_hidden + 2).cpu()):
            raise AssertionError(
                f"MTP graph replay mismatch for attention={role_rank}"
            )
        checks.append(
            {
                "phase": "mtp_graph_replay",
                "tokens": int(static_hidden.shape[0]),
                "updated_input": True,
                "roundtrip": True,
            }
        )
    else:
        checks.append(
            {
                "phase": "mtp_graph_replay",
                "fan_in_out": True,
            }
        )
    return checks


def _worker(
    role: str,
    role_rank: int,
    physical_device: int,
    attention_size: int,
    ffn_size: int,
    port: int,
    stages: int,
    steps: int,
    enable_mtp: bool,
    graph_transport: bool,
    mtp_graph_transport: bool,
    result_path: Path,
) -> None:
    connector = None
    result: dict[str, object] = {
        "role": role,
        "role_rank": role_rank,
        "physical_device": physical_device,
        "passed": False,
    }
    try:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(physical_device)
        os.environ["HCCL_EXEC_TIMEOUT"] = "0"

        import torch
        import torch.distributed as dist
        import torch_npu  # noqa: F401

        from afd_plugin.config import AFDConfig
        from afd_plugin.connectors.metadata import (
            AFDControlPayload,
            AFDDPMetadata,
            AFDTransferContext,
            AFDTransferMetadata,
        )
        from afd_plugin.connectors.npu.p2p_hccl import P2pHcclAFDConnector

        torch.npu.set_device(0)
        if graph_transport:
            world_rank = role_rank if role == "ffn" else ffn_size + role_rank
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://127.0.0.1:{port}",
                world_size=attention_size + ffn_size,
                rank=world_rank,
            )
        vllm_config = SimpleNamespace(
            additional_config={"afd": {"connector_extra_config": {}}},
            parallel_config=SimpleNamespace(
                data_parallel_size=(
                    attention_size if role == "attention" else ffn_size
                ),
                data_parallel_rank=role_rank,
                prefill_context_parallel_size=1,
                tensor_parallel_size=1,
                num_ubatches=stages,
            ),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
            model_config=SimpleNamespace(
                dtype=torch.bfloat16,
                enforce_eager=not graph_transport,
                hf_config=SimpleNamespace(
                    architectures=["DeepseekV4ForCausalLM"],
                    hidden_size=16,
                    num_hidden_layers=2,
                    vocab_size=128,
                ),
            ),
            speculative_config=(
                SimpleNamespace(
                    method="mtp",
                    enforce_eager=not mtp_graph_transport,
                )
                if enable_mtp
                else None
            ),
        )
        afd_config = AFDConfig(
            connector="P2pHcclAFDConnector",
            role=role,
            host="127.0.0.1",
            port=port,
            num_attention_ranks=attention_size,
            num_ffn_ranks=ffn_size,
        )
        connector = P2pHcclAFDConnector(
            rank=role_rank,
            local_rank=0,
            vllm_config=vllm_config,
            afd_config=afd_config,
            role_rank=role_rank,
        )
        connector.init_afd_connector()

        checks: list[dict[str, object]] = []
        for step_idx in range(steps):
            counts_by_stage = {
                stage_idx: _token_counts(
                    attention_size,
                    step_idx=step_idx,
                    stage_idx=stage_idx,
                )
                for stage_idx in range(stages)
            }
            control_payload = AFDControlPayload(
                dp_metadata_list={
                    stage_idx: AFDDPMetadata(
                        torch.tensor(counts, dtype=torch.int32),
                    )
                    for stage_idx, counts in counts_by_stage.items()
                },
                is_graph_capturing=False,
                is_warmup=False,
            )
            if role == "attention":
                connector.control_plane.update_state_from_dp_metadata(
                    control_payload,
                )
                connector.control_plane.send_dp_metadata_list(control_payload)
            else:
                received_control = connector.control_plane.recv_dp_metadata_list()
                connector.control_plane.update_state_from_dp_metadata(
                    received_control,
                )

            for stage_idx, counts in counts_by_stage.items():
                if role == "attention":
                    num_tokens = counts[role_rank]
                    id_values = _input_id_values(
                        role_rank,
                        num_tokens,
                        step_idx=step_idx,
                        stage_idx=stage_idx,
                    )
                    input_ids = torch.tensor(
                        id_values,
                        dtype=torch.int32,
                        device="npu:0",
                    )
                    hidden = torch.full(
                        (num_tokens, 16),
                        _hidden_value(
                            role_rank,
                            step_idx=step_idx,
                            stage_idx=stage_idx,
                        ),
                        dtype=torch.bfloat16,
                        device="npu:0",
                    )
                    connector.send_input_ids(input_ids, ubatch_idx=stage_idx)
                    context = AFDTransferContext(
                        metadata=AFDTransferMetadata.create_attention_metadata(
                            layer_idx=1,
                            stage_idx=stage_idx,
                            seq_len=num_tokens,
                        ),
                    )
                    connector.send_attn_output(hidden, context)
                    returned = connector.recv_ffn_output(
                        ref_tensor=torch.empty_like(hidden),
                        ubatch_idx=stage_idx,
                    )
                    if not torch.equal(returned.cpu(), (hidden + 1).cpu()):
                        raise AssertionError(
                            "round-trip mismatch for "
                            f"attention={role_rank} step={step_idx} "
                            f"stage={stage_idx}",
                        )
                    checks.append(
                        {
                            "step": step_idx,
                            "stage": stage_idx,
                            "tokens": num_tokens,
                            "roundtrip": True,
                        },
                    )
                    continue

                first_attention_rank = role_rank * connector.ratio
                peer_attention_ranks = list(
                    range(
                        first_attention_rank,
                        first_attention_rank + connector.ratio,
                    ),
                )
                peer_counts = [counts[index] for index in peer_attention_ranks]
                aggregate_tokens = sum(peer_counts)
                received_ids = connector.recv_input_ids(
                    aggregate_tokens,
                    ubatch_idx=stage_idx,
                )
                received = connector.recv_attn_output(
                    ubatch_idx=stage_idx,
                    layer_idx=0,
                    input_ids=received_ids,
                )
                expected_ids: list[int] = []
                expected_hidden_values: list[int] = []
                for attention_rank, num_tokens in zip(
                    peer_attention_ranks,
                    peer_counts,
                    strict=True,
                ):
                    expected_ids.extend(
                        _input_id_values(
                            attention_rank,
                            num_tokens,
                            step_idx=step_idx,
                            stage_idx=stage_idx,
                        ),
                    )
                    expected_hidden_values.extend(
                        [
                            _hidden_value(
                                attention_rank,
                                step_idx=step_idx,
                                stage_idx=stage_idx,
                            )
                        ]
                        * num_tokens,
                    )
                if received.input_ids.cpu().tolist() != expected_ids:
                    raise AssertionError(
                        f"input IDs mismatch for ffn={role_rank} "
                        f"step={step_idx} stage={stage_idx}",
                    )
                actual_hidden_values = received.hidden_states[:, 0].cpu().tolist()
                if actual_hidden_values != expected_hidden_values:
                    raise AssertionError(
                        f"hidden aggregation mismatch for ffn={role_rank} "
                        f"step={step_idx} stage={stage_idx}",
                    )
                if received.context.metadata.seq_lens != peer_counts:
                    raise AssertionError(
                        f"peer lengths mismatch: "
                        f"{received.context.metadata.seq_lens} != {peer_counts}",
                    )
                connector.send_ffn_output(
                    received.hidden_states + 1,
                    received.context,
                    ubatch_idx=stage_idx,
                )
                checks.append(
                    {
                        "step": step_idx,
                        "stage": stage_idx,
                        "peer_tokens": peer_counts,
                        "aggregate_tokens": aggregate_tokens,
                        "ids": True,
                        "hidden": True,
                        "output_split": True,
                    },
                )

            if not enable_mtp or mtp_graph_transport:
                continue

            mtp_counts = _mtp_token_counts(attention_size, step_idx=step_idx)
            expected_ffn_counts = _ffn_token_counts(
                mtp_counts,
                ffn_size=ffn_size,
            )
            if role == "attention":
                num_tokens = mtp_counts[role_rank]
                hidden = torch.full(
                    (num_tokens, 16),
                    _mtp_hidden_value(role_rank, step_idx=step_idx),
                    dtype=torch.bfloat16,
                    device="npu:0",
                )
                context = AFDTransferContext(
                    metadata=AFDTransferMetadata.create_attention_metadata(
                        layer_idx=0,
                        stage_idx=0,
                        seq_len=num_tokens,
                        phase="mtp",
                        speculative_step=0,
                    ),
                )
                connector.send_attn_output(
                    hidden,
                    context,
                    num_tokens_across_dp=torch.tensor(
                        mtp_counts,
                        dtype=torch.int32,
                    ),
                )
                returned = connector.recv_ffn_output(
                    ref_tensor=torch.empty_like(hidden),
                    ubatch_idx=0,
                    phase="mtp",
                )
                if not torch.equal(returned.cpu(), (hidden + 2).cpu()):
                    raise AssertionError(
                        "MTP round-trip mismatch for "
                        f"attention={role_rank} step={step_idx}"
                    )
                checks.append(
                    {
                        "phase": "mtp",
                        "step": step_idx,
                        "tokens": num_tokens,
                        "ffn_tokens": expected_ffn_counts,
                        "roundtrip": True,
                    }
                )
                continue

            first_attention_rank = role_rank * connector.ratio
            peer_attention_ranks = list(
                range(
                    first_attention_rank,
                    first_attention_rank + connector.ratio,
                )
            )
            peer_counts = [mtp_counts[index] for index in peer_attention_ranks]
            header = connector.recv_mtp_header(stage_idx=0)
            if header.num_tokens != sum(peer_counts):
                raise AssertionError(
                    f"MTP aggregate token mismatch for ffn={role_rank}: "
                    f"{header.num_tokens} != {sum(peer_counts)}"
                )
            if header.num_tokens_across_dp.tolist() != expected_ffn_counts:
                raise AssertionError(
                    f"MTP FFN counts mismatch: "
                    f"{header.num_tokens_across_dp.tolist()} != "
                    f"{expected_ffn_counts}"
                )
            received = connector.recv_attn_output(
                ubatch_idx=0,
                layer_idx=0,
                phase="mtp",
                speculative_step=header.speculative_step,
                num_tokens=header.num_tokens,
            )
            expected_hidden_values: list[int] = []
            for attention_rank, num_tokens in zip(
                peer_attention_ranks,
                peer_counts,
                strict=True,
            ):
                expected_hidden_values.extend(
                    [_mtp_hidden_value(attention_rank, step_idx=step_idx)] * num_tokens
                )
            actual_hidden_values = received.hidden_states[:, 0].cpu().tolist()
            if actual_hidden_values != expected_hidden_values:
                raise AssertionError(
                    f"MTP hidden aggregation mismatch for ffn={role_rank} "
                    f"step={step_idx}"
                )
            if received.context.metadata.seq_lens != peer_counts:
                raise AssertionError(
                    f"MTP peer lengths mismatch: "
                    f"{received.context.metadata.seq_lens} != {peer_counts}"
                )
            connector.send_ffn_output(
                received.hidden_states + 2,
                received.context,
                ubatch_idx=0,
            )
            checks.append(
                {
                    "phase": "mtp",
                    "step": step_idx,
                    "peer_tokens": peer_counts,
                    "aggregate_tokens": header.num_tokens,
                    "ffn_tokens": expected_ffn_counts,
                    "header_fan_in": True,
                    "hidden_fan_in": True,
                    "output_split": True,
                }
            )

        if graph_transport:
            checks.extend(
                _validate_graph_transport(
                    connector=connector,
                    role=role,
                    role_rank=role_rank,
                    attention_size=attention_size,
                    stages=stages,
                    step_idx=2,
                )
            )
        if mtp_graph_transport:
            checks.extend(
                _validate_mtp_graph_transport(
                    connector=connector,
                    role=role,
                    role_rank=role_rank,
                    attention_size=attention_size,
                    ffn_size=ffn_size,
                    stages=stages,
                    step_idx=2,
                )
            )

        torch.npu.synchronize()
        result.update(passed=True, checks=checks)
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        if connector is not None and connector.is_initialized:
            try:
                connector.close()
            except BaseException:
                result["passed"] = False
                result["close_error"] = traceback.format_exc()
        default_group_closed = True
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
            default_group_closed = not dist.is_initialized()
        except BaseException:
            default_group_closed = False
            result["passed"] = False
            result["default_group_close_error"] = traceback.format_exc()
        result["closed"] = bool(
            connector is not None
            and not connector.is_initialized
            and default_group_closed,
        )
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
        )
    if not result["passed"]:
        raise SystemExit(1)


def _parse_devices(raw: str) -> list[int]:
    try:
        devices = [int(value) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "devices must be a comma-separated integer list",
        ) from exc
    if not devices or any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError("devices must be non-negative")
    return devices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=29831)
    parser.add_argument("--attention-devices", type=_parse_devices, default=[0])
    parser.add_argument("--ffn-devices", type=_parse_devices, default=[8])
    parser.add_argument("--stages", type=int, choices=(1, 2), default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--enable-mtp", action="store_true")
    parser.add_argument("--graph-transport", action="store_true")
    parser.add_argument("--mtp-graph-transport", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    attention_size = len(args.attention_devices)
    ffn_size = len(args.ffn_devices)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.mtp_graph_transport and not args.enable_mtp:
        parser.error("--mtp-graph-transport requires --enable-mtp")
    if args.mtp_graph_transport and not args.graph_transport:
        parser.error("--mtp-graph-transport requires --graph-transport")
    if attention_size < ffn_size or attention_size % ffn_size != 0:
        parser.error("Attention count must be an integer multiple of FFN count")
    all_devices = [*args.attention_devices, *args.ffn_devices]
    if len(set(all_devices)) != len(all_devices):
        parser.error("Attention and FFN device lists must not overlap")
    if not _port_is_free(args.port):
        raise RuntimeError(f"port {args.port} is already in use")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    result_paths: dict[tuple[str, int], Path] = {}
    process_specs: list[tuple[str, int, int]] = []
    for role, devices in (
        ("ffn", args.ffn_devices),
        ("attention", args.attention_devices),
    ):
        for role_rank, physical_device in enumerate(devices):
            result_paths[(role, role_rank)] = (
                args.output.parent / f"{role}_{role_rank}_result.json"
            )
            process_specs.append((role, role_rank, physical_device))

    context = mp.get_context("spawn")
    workers = [
        context.Process(
            target=_worker,
            args=(
                role,
                role_rank,
                physical_device,
                attention_size,
                ffn_size,
                args.port,
                args.stages,
                args.steps,
                args.enable_mtp,
                args.graph_transport,
                args.mtp_graph_transport,
                result_paths[(role, role_rank)],
            ),
            name=f"hccl-p2p-{role}-{role_rank}",
        )
        for role, role_rank, physical_device in process_specs
    ]
    for worker in workers:
        worker.start()

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if all(worker.exitcode is not None for worker in workers):
            break
        if any(
            worker.exitcode is not None and worker.exitcode != 0 for worker in workers
        ):
            break
        time.sleep(0.2)
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        if worker.is_alive():
            worker.join(timeout=30)

    exit_codes = {worker.name: worker.exitcode for worker in workers}
    results = []
    for role, role_rank, physical_device in process_specs:
        result_path = result_paths[(role, role_rank)]
        if result_path.is_file():
            results.append(json.loads(result_path.read_text()))
        else:
            results.append(
                {
                    "role": role,
                    "role_rank": role_rank,
                    "physical_device": physical_device,
                    "passed": False,
                    "error": "role result file was not written",
                },
            )
    passed = (
        len(results) == len(workers)
        and all(bool(result.get("passed")) for result in results)
        and all(exit_code == 0 for exit_code in exit_codes.values())
    )
    summary = {
        "passed": passed,
        "topology": {
            "attention_size": attention_size,
            "ffn_size": ffn_size,
            "ratio": attention_size // ffn_size,
            "attention_devices": args.attention_devices,
            "ffn_devices": args.ffn_devices,
        },
        "port": args.port,
        "stages": args.stages,
        "steps": args.steps,
        "enable_mtp": args.enable_mtp,
        "graph_transport": args.graph_transport,
        "mtp_graph_transport": args.mtp_graph_transport,
        "results": results,
        "exit_codes": exit_codes,
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
