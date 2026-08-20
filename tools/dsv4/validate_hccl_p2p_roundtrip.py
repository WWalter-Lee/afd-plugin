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


def _worker(
    role: str,
    role_rank: int,
    physical_device: int,
    attention_size: int,
    ffn_size: int,
    port: int,
    stages: int,
    steps: int,
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
                hf_config=SimpleNamespace(
                    architectures=["DeepseekV4ForCausalLM"],
                    hidden_size=16,
                    num_hidden_layers=2,
                    vocab_size=128,
                ),
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
        result["closed"] = bool(
            connector is not None and not connector.is_initialized,
        )
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
        )


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
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    attention_size = len(args.attention_devices)
    ffn_size = len(args.ffn_devices)
    if args.steps <= 0:
        parser.error("--steps must be positive")
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
                result_paths[(role, role_rank)],
            ),
            name=f"hccl-p2p-{role}-{role_rank}",
        )
        for role, role_rank, physical_device in process_specs
    ]
    for worker in workers:
        worker.start()

    deadline = time.monotonic() + args.timeout
    for worker in workers:
        worker.join(timeout=max(0.0, deadline - time.monotonic()))
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
        "results": results,
        "exit_codes": exit_codes,
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
