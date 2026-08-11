#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.npu.camp2p import CAMP2pAFDConnector


def _vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={"afd": {"connector_extra_config": {}}},
        parallel_config=SimpleNamespace(num_ubatches=2),
        scheduler_config=SimpleNamespace(
            max_num_seqs=8,
            max_num_batched_tokens=32,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"],
                hidden_size=4096,
                num_experts_per_tok=6,
                n_routed_experts=256,
                vocab_size=129280,
            )
        ),
    )


def _worker(process_idx: int, port: int, output_dir: str) -> None:
    role = "ffn" if process_idx == 0 else "attention"
    torch.npu.set_device(process_idx)
    connector = CAMP2pAFDConnector(
        process_idx,
        process_idx,
        _vllm_config(),
        AFDConfig(
            connector="CAMP2pAFDConnector",
            role=role,
            host="127.0.0.1",
            port=port,
            num_attention_ranks=1,
            num_ffn_ranks=1,
        ),
        0,
    )
    cases = [
        (0, [-1, 0, 129279]),
        (1, [7, 8, 9, 10, 11]),
        (0, [17]),
        (1, [-1, 23, 29, 31]),
    ]
    received = []
    invalid_rejected = False
    try:
        connector.init_afd_connector()
        for stage_idx, values in cases:
            if role == "attention":
                connector._send_input_ids(
                    torch.tensor(values, dtype=torch.int64, device="npu"),
                    ubatch_idx=stage_idx,
                )
            else:
                tensor = connector._recv_input_ids(
                    len(values),
                    ubatch_idx=stage_idx,
                )
                actual = tensor.cpu().tolist()
                if actual != values:
                    raise AssertionError(
                        f"stage {stage_idx}: expected {values}, got {actual}"
                    )
                received.append(actual)
        if role == "attention":
            try:
                connector._send_input_ids(
                    torch.tensor([129280], dtype=torch.int64, device="npu"),
                    ubatch_idx=0,
                )
            except ValueError:
                invalid_rejected = True
            else:
                raise AssertionError("out-of-vocabulary ID was not rejected")
        torch.npu.synchronize()
    finally:
        connector.close()

    result = {
        "role": role,
        "received": received,
        "invalid_rejected": invalid_rejected,
        "closed": not connector.is_initialized,
        "ids_groups_after_close": len(connector.ids_pg_list),
        "buffers_after_close": len(connector.input_ids_buffers),
    }
    result_path = Path(output_dir) / f"ids_roundtrip_{role}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=29741)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mp.spawn(
        _worker,
        args=(args.port, str(args.output_dir)),
        nprocs=2,
        join=True,
    )
    results = [
        json.loads((args.output_dir / f"ids_roundtrip_{role}.json").read_text())
        for role in ("attention", "ffn")
    ]
    summary = {"status": "passed", "workers": results}
    (args.output_dir / "ids_roundtrip_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
