#!/usr/bin/env python3
"""Compare per-layer DSV4 captures for one-shot and chunked prefill."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional


def _load(root: Path, position: int) -> dict[str, dict[str, Any]]:
    captures = {}
    for metadata_path in sorted(root.rglob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("selected_position") != position:
            continue
        tensor_path = metadata_path.with_suffix(".pt")
        tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
        captures[metadata["layer_name"]] = {"metadata": metadata, "tensors": tensors}
    return captures


def _metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    difference = (left - right).abs()
    return {
        "exact": bool(torch.equal(left, right)),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "cosine": float(
            functional.cosine_similarity(
                left.reshape(1, -1), right.reshape(1, -1)
            ).item()
        ),
        "finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
    }


def _layer_index(name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    if match is None:
        raise ValueError(f"Cannot extract layer index from {name!r}")
    return int(match.group(1))


def compare(one_shot_root: Path, chunked_root: Path, position: int) -> dict[str, Any]:
    one_shot = _load(one_shot_root, position)
    chunked = _load(chunked_root, position)
    common_layers = sorted(
        one_shot.keys() & chunked.keys(),
        key=_layer_index,
    )
    layers = []
    for layer_name in common_layers:
        left = one_shot[layer_name]
        right = chunked[layer_name]
        layers.append(
            {
                "layer_name": layer_name,
                "compress_ratio": left["metadata"]["compress_ratio"],
                "one_shot_mode": left["metadata"].get("attention_mode", "prefill"),
                "chunked_mode": right["metadata"].get("attention_mode", "prefill"),
                "one_shot_call": left["metadata"].get(
                    "forward_call_index", left["metadata"].get("prefill_call_index")
                ),
                "chunked_call": right["metadata"].get(
                    "forward_call_index", right["metadata"].get("prefill_call_index")
                ),
                "hidden": _metrics(
                    left["tensors"]["hidden"], right["tensors"]["hidden"]
                ),
                "attention_output": _metrics(
                    left["tensors"]["attention_output"],
                    right["tensors"]["attention_output"],
                ),
            }
        )
    first_hidden_difference = next(
        (layer["layer_name"] for layer in layers if not layer["hidden"]["exact"]),
        None,
    )
    first_attention_difference = next(
        (
            layer["layer_name"]
            for layer in layers
            if not layer["attention_output"]["exact"]
        ),
        None,
    )
    return {
        "schema_version": 1,
        "position": position,
        "one_shot_capture_count": len(one_shot),
        "chunked_capture_count": len(chunked),
        "compared_layer_count": len(layers),
        "first_hidden_difference": first_hidden_difference,
        "first_attention_difference": first_attention_difference,
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("one_shot", type=Path)
    parser.add_argument("chunked", type=Path)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = compare(args.one_shot, args.chunked, args.position)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["compared_layer_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
