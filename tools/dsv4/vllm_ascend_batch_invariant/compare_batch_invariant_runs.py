#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("records"), list):
        raise ValueError(f"missing records: {path}")
    return payload


def _key(record: dict[str, Any]) -> tuple[int, int]:
    return int(record["round"]), int(record["prompt_index"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    first = _load(args.first)
    second = _load(args.second)
    first_records = {_key(record): record for record in first["records"]}
    second_records = {_key(record): record for record in second["records"]}
    all_keys = sorted(set(first_records) | set(second_records))
    mismatches: list[dict[str, int]] = []
    for key in all_keys:
        left = first_records.get(key)
        right = second_records.get(key)
        if (
            left is None
            or right is None
            or left.get("prompt_token_ids") != right.get("prompt_token_ids")
            or left.get("token_ids") != right.get("token_ids")
        ):
            mismatches.append({"round": key[0], "prompt_index": key[1]})

    report = {
        "passed": bool(first.get("passed"))
        and bool(second.get("passed"))
        and not mismatches,
        "first_passed_across_rounds": bool(first.get("passed")),
        "second_passed_across_rounds": bool(second.get("passed")),
        "request_count": len(all_keys),
        "cross_start_exact_match_count": len(all_keys) - len(mismatches),
        "mismatches": mismatches,
        "first_reference_comparison": first.get("reference_comparison"),
        "second_reference_comparison": second.get("reference_comparison"),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
