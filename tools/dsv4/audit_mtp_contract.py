#!/usr/bin/env python3
"""Audit DeepSeek-V4 MTP checkpoint keys for the AFD role contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

MTPRole = Literal["attention", "ffn"]


def classify_mtp_key(name: str) -> MTPRole:
    """Return the strict AFD owner of one raw ``mtp.*`` checkpoint key."""
    normalized = name.removeprefix("model.")
    parts = normalized.split(".")
    if len(parts) < 3 or parts[0] != "mtp" or not parts[1].isdigit():
        raise ValueError(f"not a DeepSeek-V4 MTP checkpoint key: {name}")
    return "ffn" if parts[2] == "ffn" else "attention"


def _quant_suffix(name: str) -> str:
    if name.endswith((".weight_scale", ".scale")):
        return "scale"
    if name.endswith((".weight_offset", ".offset")):
        return "offset"
    if name.endswith(".weight"):
        return "weight"
    return "auxiliary"


def _tensor_family(name: str) -> str:
    for suffix in (".weight_scale", ".weight_offset", ".scale", ".offset"):
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".weight"
    return name


def build_report(index_path: Path, *, expected_layers: int = 1) -> dict[str, Any]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"checkpoint index has no weight_map object: {index_path}")

    mtp_keys = sorted(
        name
        for name in weight_map
        if name.removeprefix("model.").startswith("mtp.")
    )
    if not mtp_keys:
        raise ValueError(f"checkpoint index has no MTP keys: {index_path}")

    role_counts: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    role_keys: dict[str, list[str]] = defaultdict(list)
    family_roles: dict[str, set[str]] = defaultdict(set)
    layer_indices: set[int] = set()
    for name in mtp_keys:
        normalized = name.removeprefix("model.")
        layer_indices.add(int(normalized.split(".", 2)[1]))
        role = classify_mtp_key(name)
        role_counts[role] += 1
        suffix_counts[_quant_suffix(name)] += 1
        role_keys[role].append(name)
        family_roles[_tensor_family(name)].add(role)

    mixed_role_families = sorted(
        family for family, roles in family_roles.items() if len(roles) != 1
    )
    expected_layer_indices = list(range(expected_layers))
    actual_layer_indices = sorted(layer_indices)
    passed = (
        actual_layer_indices == expected_layer_indices
        and not mixed_role_families
        and role_counts["attention"] > 0
        and role_counts["ffn"] > 0
    )
    return {
        "index": str(index_path.resolve()),
        "total_checkpoint_keys": len(weight_map),
        "mtp": {
            "total_keys": len(mtp_keys),
            "layer_indices": actual_layer_indices,
            "expected_layer_indices": expected_layer_indices,
            "role_counts": {
                "attention": role_counts["attention"],
                "ffn": role_counts["ffn"],
            },
            "quant_suffix_counts": dict(sorted(suffix_counts.items())),
            "mixed_role_tensor_families": mixed_role_families,
            "attention_keys": role_keys["attention"],
        },
        "contract": {
            "ffn_owner_rule": "mtp.<layer>.ffn.*",
            "attention_owner_rule": "all remaining mtp.<layer>.* keys",
            "weight_scale_offset_same_role": not mixed_role_families,
        },
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-layers", type=int, default=1)
    args = parser.parse_args()
    if args.expected_layers <= 0:
        parser.error("--expected-layers must be positive")

    report = build_report(args.index, expected_layers=args.expected_layers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "MTP contract audit: "
        f"total={report['mtp']['total_keys']} "
        f"attention={report['mtp']['role_counts']['attention']} "
        f"ffn={report['mtp']['role_counts']['ffn']} "
        f"passed={report['passed']}",
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
