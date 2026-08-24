# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run a two-process Mooncake Ascend buffer round-trip without loading a model."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import traceback
from typing import Any

BUFFER_BYTES = 2 * 1024 * 1024


def _set_device_environment(device: int) -> None:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    os.environ.setdefault("HCCL_IF_IP", "127.0.0.1")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("HCCL_SOCKET_IFNAME", "lo")


def _producer(
    device: int,
    ready_queue: Any,
    done_event: Any,
) -> None:
    try:
        _set_device_environment(device)
        import torch
        import torch_npu  # noqa: F401
        from mooncake.engine import TransferEngine

        torch.npu.set_device(0)
        source = torch.full(
            (BUFFER_BYTES,),
            0x5A,
            dtype=torch.uint8,
            device="npu:0",
        )
        engine = TransferEngine()
        initialize_ret = engine.initialize(
            "127.0.0.1",
            "P2PHANDSHAKE",
            "ascend",
            "",
        )
        if initialize_ret != 0:
            raise RuntimeError(
                f"producer TransferEngine initialization failed: {initialize_ret}"
            )
        register_ret = engine.register_memory(source.data_ptr(), source.numel())
        if register_ret != 0:
            raise RuntimeError(
                f"producer Mooncake memory registration failed: {register_ret}"
            )
        ready_queue.put(
            {
                "ok": True,
                "rpc_port": int(engine.get_rpc_port()),
                "address": int(source.data_ptr()),
                "size": int(source.numel()),
            }
        )
        if not done_event.wait(timeout=180):
            raise TimeoutError("producer timed out waiting for consumer")
        torch.npu.synchronize()
    except BaseException:
        ready_queue.put({"ok": False, "error": traceback.format_exc()})
        raise


def _consumer(
    device: int,
    remote: dict[str, int],
    result_queue: Any,
) -> None:
    try:
        _set_device_environment(device)
        import torch
        import torch_npu  # noqa: F401
        from mooncake.engine import TransferEngine

        torch.npu.set_device(0)
        destination = torch.zeros(
            (remote["size"],),
            dtype=torch.uint8,
            device="npu:0",
        )
        engine = TransferEngine()
        initialize_ret = engine.initialize(
            "127.0.0.1",
            "P2PHANDSHAKE",
            "ascend",
            "",
        )
        if initialize_ret != 0:
            raise RuntimeError(
                f"consumer TransferEngine initialization failed: {initialize_ret}"
            )
        register_ret = engine.register_memory(
            destination.data_ptr(),
            destination.numel(),
        )
        if register_ret != 0:
            raise RuntimeError(
                f"consumer Mooncake memory registration failed: {register_ret}"
            )

        session_id = f"127.0.0.1:{remote['rpc_port']}"
        transfer_results = []
        for _ in range(2):
            destination.zero_()
            transfer_ret = engine.batch_transfer_sync_read(
                session_id,
                [destination.data_ptr()],
                [remote["address"]],
                [remote["size"]],
            )
            torch.npu.synchronize()
            transfer_results.append(int(transfer_ret))
            if transfer_ret < 0:
                raise RuntimeError(f"Mooncake sync read failed: {transfer_ret}")
            if not bool(torch.all(destination == 0x5A).item()):
                raise RuntimeError("Mooncake round-trip data mismatch")

        result_queue.put(
            {
                "ok": True,
                "session_id": session_id,
                "bytes": remote["size"],
                "iterations": 2,
                "transfer_results": transfer_results,
            }
        )
    except BaseException:
        result_queue.put({"ok": False, "error": traceback.format_exc()})
        raise


def _get_result(result_queue: Any, label: str, timeout: int) -> dict[str, Any]:
    try:
        result = result_queue.get(timeout=timeout)
    except queue.Empty as error:
        raise TimeoutError(f"timed out waiting for Mooncake {label}") from error
    if not result.get("ok"):
        raise RuntimeError(f"Mooncake {label} failed:\n{result.get('error')}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer-device", type=int, default=0)
    parser.add_argument("--consumer-device", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.producer_device == args.consumer_device:
        parser.error("producer and consumer devices must differ")

    context = mp.get_context("spawn")
    ready_queue = context.Queue()
    result_queue = context.Queue()
    done_event = context.Event()
    producer = context.Process(
        target=_producer,
        args=(args.producer_device, ready_queue, done_event),
        name="mooncake-producer",
    )
    consumer = None
    try:
        producer.start()
        remote = _get_result(ready_queue, "producer startup", args.timeout)
        consumer = context.Process(
            target=_consumer,
            args=(args.consumer_device, remote, result_queue),
            name="mooncake-consumer",
        )
        consumer.start()
        result = _get_result(result_queue, "consumer transfer", args.timeout)
        done_event.set()
        consumer.join(timeout=args.timeout)
        producer.join(timeout=args.timeout)
        if consumer.exitcode != 0 or producer.exitcode != 0:
            raise RuntimeError(
                "Mooncake child exit failure: "
                f"producer={producer.exitcode}, consumer={consumer.exitcode}"
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        done_event.set()
        for process in (consumer, producer):
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
