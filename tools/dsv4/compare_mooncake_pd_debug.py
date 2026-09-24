#!/usr/bin/env python3
"""Compare producer and consumer KV digests from Mooncake PD diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_events(root: Path) -> list[dict[str, Any]]:
    event_root = root / "events" if (root / "events").is_dir() else root
    events = []
    for path in sorted(event_root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events


def _digest_index(
    events: list[dict[str, Any]], event_name: str
) -> dict[tuple[str, int, int, int, int], str]:
    index = {}
    for event in events:
        if event.get("event") != event_name:
            continue
        request_id = event.get("remote_request_id", event["request_id"])
        tp_rank = event["tp_rank"]
        for group in event["groups"]:
            for part in group.get("parts", []):
                for digest in part["digests"]:
                    if "sha256" not in digest:
                        continue
                    key = (
                        request_id,
                        tp_rank,
                        group["group_index"],
                        part["part_index"],
                        digest["ordinal"],
                    )
                    index[key] = digest["sha256"]
    return index


def compare(producer_root: Path, consumer_root: Path) -> dict[str, Any]:
    producer_events = _load_events(producer_root)
    consumer_events = _load_events(consumer_root)
    producer = _digest_index(producer_events, "producer_kv_digest")
    consumer = _digest_index(consumer_events, "consumer_kv_digest")
    common_keys = sorted(producer.keys() & consumer.keys())
    mismatches = [
        {
            "request_id": key[0],
            "tp_rank": key[1],
            "group_index": key[2],
            "part_index": key[3],
            "block_ordinal": key[4],
            "producer_sha256": producer[key],
            "consumer_sha256": consumer[key],
        }
        for key in common_keys
        if producer[key] != consumer[key]
    ]
    return {
        "schema_version": 1,
        "producer_digest_count": len(producer),
        "consumer_digest_count": len(consumer),
        "compared_digest_count": len(common_keys),
        "matched_digest_count": len(common_keys) - len(mismatches),
        "mismatch_count": len(mismatches),
        "producer_only_count": len(producer.keys() - consumer.keys()),
        "consumer_only_count": len(consumer.keys() - producer.keys()),
        "mismatches": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("producer", type=Path)
    parser.add_argument("consumer", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = compare(args.producer, args.consumer)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 1 if result["mismatch_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
