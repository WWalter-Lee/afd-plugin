#!/usr/bin/env python3
"""Validate P2pHcclAFDConnector with two physical NPUs."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
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


def _worker(
    role: str,
    physical_device: int,
    port: int,
    stages: int,
    steps: int,
    result_path: Path,
) -> None:
    connector = None
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
                data_parallel_size=1,
                data_parallel_rank=0,
                prefill_context_parallel_size=1,
                tensor_parallel_size=1,
                num_ubatches=stages,
            ),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
            model_config=SimpleNamespace(
                dtype=torch.bfloat16,
                hf_config=SimpleNamespace(
                    architectures=["DeepseekV4ForCausalLM"],
                    hidden_size=16,
                    vocab_size=128,
                ),
            ),
        )
        afd_config = AFDConfig(
            connector="P2pHcclAFDConnector",
            role=role,
            host="127.0.0.1",
            port=port,
            num_attention_ranks=1,
            num_ffn_ranks=1,
        )
        connector = P2pHcclAFDConnector(
            rank=0,
            local_rank=0,
            vllm_config=vllm_config,
            afd_config=afd_config,
            role_rank=0,
        )
        connector.init_afd_connector()

        checks: list[dict[str, object]] = []
        for step_idx in range(steps):
            stage_counts = {
                stage_idx: 2 + step_idx + stage_idx for stage_idx in range(stages)
            }
            payload = AFDControlPayload(
                dp_metadata_list={
                    stage_idx: AFDDPMetadata(
                        torch.tensor([num_tokens], dtype=torch.int32),
                    )
                    for stage_idx, num_tokens in stage_counts.items()
                },
                is_graph_capturing=False,
                is_warmup=False,
            )
            if role == "attention":
                connector.control_plane.update_state_from_dp_metadata(payload)
                connector.control_plane.send_dp_metadata_list(payload)
            else:
                received_control = connector.control_plane.recv_dp_metadata_list()
                connector.control_plane.update_state_from_dp_metadata(received_control)

            for stage_idx, num_tokens in stage_counts.items():
                ids = torch.arange(
                    step_idx * 16 + stage_idx * 4,
                    step_idx * 16 + stage_idx * 4 + num_tokens,
                    dtype=torch.int32,
                    device="npu:0",
                )
                hidden = torch.full(
                    (num_tokens, 16),
                    float(step_idx * 10 + stage_idx),
                    dtype=torch.bfloat16,
                    device="npu:0",
                )
                if role == "attention":
                    connector.send_input_ids(ids, ubatch_idx=stage_idx)
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
                    expected = hidden + 1
                    if not torch.equal(returned.cpu(), expected.cpu()):
                        raise AssertionError(
                            f"round-trip mismatch at step={step_idx} stage={stage_idx}",
                        )
                    checks.append(
                        {
                            "step": step_idx,
                            "stage": stage_idx,
                            "tokens": num_tokens,
                            "roundtrip": True,
                        },
                    )
                else:
                    received_ids = connector.recv_input_ids(
                        num_tokens,
                        ubatch_idx=stage_idx,
                    )
                    received = connector.recv_attn_output(
                        ubatch_idx=stage_idx,
                        layer_idx=0,
                        input_ids=received_ids,
                    )
                    if not torch.equal(received.input_ids.cpu(), ids.cpu()):
                        raise AssertionError(
                            f"input IDs mismatch at step={step_idx} stage={stage_idx}",
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
                            "tokens": num_tokens,
                            "ids": True,
                        },
                    )

        result_path.write_text(
            json.dumps(
                {
                "role": role,
                "physical_device": physical_device,
                "passed": True,
                "checks": checks,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    except BaseException:
        result_path.write_text(
            json.dumps(
                {
                    "role": role,
                    "physical_device": physical_device,
                    "passed": False,
                    "error": traceback.format_exc(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    finally:
        if connector is not None and connector.is_initialized:
            try:
                connector.close()
            except BaseException:
                result_path.write_text(
                    json.dumps(
                        {
                            "role": role,
                            "physical_device": physical_device,
                            "passed": False,
                            "error": (
                                "connector close failed\n" + traceback.format_exc()
                            ),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=29831)
    parser.add_argument("--attention-device", type=int, default=0)
    parser.add_argument("--ffn-device", type=int, default=8)
    parser.add_argument("--stages", type=int, choices=(1, 2), default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not _port_is_free(args.port):
        raise RuntimeError(f"port {args.port} is already in use")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    context = mp.get_context("spawn")
    role_result_paths = {
        role: args.output.parent / f"{role}_result.json"
        for role in ("attention", "ffn")
    }
    workers = [
        context.Process(
            target=_worker,
            args=(
                "ffn",
                args.ffn_device,
                args.port,
                args.stages,
                args.steps,
                role_result_paths["ffn"],
            ),
            name="hccl-p2p-ffn",
        ),
        context.Process(
            target=_worker,
            args=(
                "attention",
                args.attention_device,
                args.port,
                args.stages,
                args.steps,
                role_result_paths["attention"],
            ),
            name="hccl-p2p-attention",
        ),
    ]
    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join(timeout=args.timeout)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=30)

    exit_codes = {worker.name: worker.exitcode for worker in workers}
    results = []
    for role, result_path in role_result_paths.items():
        if result_path.is_file():
            results.append(json.loads(result_path.read_text()))
        else:
            results.append(
                {
                    "role": role,
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
