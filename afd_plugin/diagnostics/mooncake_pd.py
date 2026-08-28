# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded Mooncake PD KV transfer diagnostics.

This module is a separate vLLM general plugin so a native Mooncake PD control
can load the diagnostics without registering AFD models or AFD connectors. It
is inert unless ``AFD_MOONCAKE_PD_DEBUG_DIR`` is set.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import socket
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from vllm.v1.core.kv_cache_utils import BlockIds
    from vllm.v1.request import Request

DEBUG_DIR_ENV = "AFD_MOONCAKE_PD_DEBUG_DIR"
MAX_REQUESTS_ENV = "AFD_MOONCAKE_PD_DEBUG_MAX_REQUESTS"
MAX_BLOCKS_ENV = "AFD_MOONCAKE_PD_DEBUG_MAX_BLOCKS"
MAX_BYTES_ENV = "AFD_MOONCAKE_PD_DEBUG_MAX_BYTES"

DEFAULT_MAX_REQUESTS = 1
DEFAULT_MAX_BLOCKS = 2
DEFAULT_MAX_BYTES = 8192
MAX_IDENTIFIER_LENGTH = 160

_logger = logging.getLogger(__name__)


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        _logger.warning("Ignoring invalid %s=%r", name, raw_value)
        return default
    if value < 1:
        _logger.warning("Ignoring non-positive %s=%r", name, raw_value)
        return default
    return value


def _debug_dir() -> Path | None:
    value = os.environ.get(DEBUG_DIR_ENV)
    return Path(value).resolve() if value else None


def _sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return sanitized[:MAX_IDENTIFIER_LENGTH] or "unknown"


def _write_record(event: str, fields: dict[str, Any]) -> None:
    output_dir = _debug_dir()
    if output_dir is None:
        return
    events_dir = output_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "event": event,
        "timestamp_ns": time.time_ns(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        **fields,
    }
    output_path = events_dir / (
        f"{_sanitize_identifier(socket.gethostname())}-pid{os.getpid()}.jsonl"
    )
    payload = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(payload)
        output_file.write("\n")


def _plan_path(engine_id: str, request_id: str) -> Path | None:
    output_dir = _debug_dir()
    if output_dir is None:
        return None
    plan_dir = output_dir / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    return plan_dir / (
        f"{_sanitize_identifier(engine_id)}--{_sanitize_identifier(request_id)}.json"
    )


def _write_plan(engine_id: str, request_id: str, plan: dict[str, Any]) -> None:
    output_path = _plan_path(engine_id, request_id)
    if output_path is None:
        return
    temporary_path = output_path.with_suffix(f".tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(plan, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def _read_plan(engine_id: str, request_id: str) -> dict[str, Any] | None:
    input_path = _plan_path(engine_id, request_id)
    if input_path is None or not input_path.is_file():
        return None
    return json.loads(input_path.read_text(encoding="utf-8"))


def _claim_request(instance: Any, request_id: str) -> bool:
    claimed_requests = instance.__dict__.setdefault(
        "_afd_pd_debug_claimed_requests", set()
    )
    if request_id in claimed_requests:
        return False
    max_requests = _positive_int_env(MAX_REQUESTS_ENV, DEFAULT_MAX_REQUESTS)
    if len(claimed_requests) >= max_requests:
        return False
    claimed_requests.add(request_id)
    return True


def _select_block_ids(block_ids: list[int]) -> list[tuple[int, int]]:
    max_blocks = _positive_int_env(MAX_BLOCKS_ENV, DEFAULT_MAX_BLOCKS)
    if len(block_ids) <= max_blocks:
        return list(enumerate(block_ids))
    if max_blocks == 1:
        return [(0, block_ids[0])]
    selected_positions = [0, len(block_ids) - 1]
    if max_blocks > 2:
        step = max(1, (len(block_ids) - 1) // (max_blocks - 1))
        selected_positions = list(range(0, len(block_ids), step))[:max_blocks]
        selected_positions[-1] = len(block_ids) - 1
    return [(position, block_ids[position]) for position in selected_positions]


def _tensor_layout(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data_ptr": tensor.data_ptr(),
        "storage_offset": tensor.storage_offset(),
        "element_size": tensor.element_size(),
    }


def _cache_parts(cache: Any, *, split_tensor: bool) -> list[torch.Tensor]:
    if isinstance(cache, (tuple, list)):
        return list(cache)
    if split_tensor:
        return list(cache)
    return [cache]


def _block_digest(
    tensor: torch.Tensor,
    block_id: int,
    *,
    max_bytes: int,
) -> dict[str, Any]:
    if block_id < 0 or block_id >= tensor.shape[0]:
        return {
            "error": "block_id_out_of_range",
            "block_id": block_id,
            "tensor_blocks": tensor.shape[0],
        }
    block = tensor[block_id].contiguous().view(dtype=__import__("torch").uint8)
    byte_count = min(block.numel(), max_bytes)
    sample = block.reshape(-1)[:byte_count].cpu()
    payload = bytes(sample.tolist())
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "sample_bytes": byte_count,
        "block_bytes": block.numel(),
        "nonzero_sample_bytes": sum(value != 0 for value in payload),
    }


def _summarize_group_blocks(
    kv_caches: dict[str, Any],
    kv_cache_config: Any,
    block_ids_by_group: list[list[int]] | tuple[list[int], ...],
    *,
    split_tensor: bool,
) -> list[dict[str, Any]]:
    max_bytes = _positive_int_env(MAX_BYTES_ENV, DEFAULT_MAX_BYTES)
    summaries = []
    for group_index, group in enumerate(kv_cache_config.kv_cache_groups):
        group_block_ids = list(block_ids_by_group[group_index])
        layer_name = next(
            (name for name in group.layer_names if name in kv_caches), None
        )
        group_summary: dict[str, Any] = {
            "group_index": group_index,
            "block_ids": group_block_ids,
            "layer_name": layer_name,
            "spec_type": type(group.kv_cache_spec).__name__,
        }
        if layer_name is None:
            group_summary["error"] = "no_group_layer_in_kv_caches"
            summaries.append(group_summary)
            continue
        selected_blocks = _select_block_ids(group_block_ids)
        part_summaries = []
        for part_index, tensor in enumerate(
            _cache_parts(kv_caches[layer_name], split_tensor=split_tensor)
        ):
            digests = []
            for ordinal, block_id in selected_blocks:
                digest = _block_digest(tensor, block_id, max_bytes=max_bytes)
                digests.append({"ordinal": ordinal, "block_id": block_id, **digest})
            part_summaries.append(
                {
                    "part_index": part_index,
                    "layout": _tensor_layout(tensor),
                    "digests": digests,
                }
            )
        group_summary["parts"] = part_summaries
        summaries.append(group_summary)
    return summaries


def _registration_record(worker: Any, kv_caches: dict[str, Any]) -> dict[str, Any]:
    groups = []
    for group_index, group in enumerate(worker.kv_cache_config.kv_cache_groups):
        groups.append(
            {
                "group_index": group_index,
                "spec_type": type(group.kv_cache_spec).__name__,
                "layer_names": list(group.layer_names),
            }
        )
    layouts = []
    for layer_name in sorted(kv_caches):
        parts = _cache_parts(kv_caches[layer_name], split_tensor=worker.use_compress)
        layouts.append(
            {
                "layer_name": layer_name,
                "parts": [_tensor_layout(tensor) for tensor in parts],
            }
        )
    return {
        "role": worker.kv_role,
        "engine_id": worker.engine_id,
        "tp_rank": worker.tp_rank,
        "num_blocks": worker.num_blocks,
        "use_hybrid": worker.use_hybrid,
        "use_compress": worker.use_compress,
        "groups": groups,
        "kv_cache_layouts": layouts,
        "kv_caches_base_addr": worker.kv_caches_base_addr,
        "block_len_per_addr": worker.block_len_per_addr,
        "block_stride_per_addr": worker.block_stride_per_addr,
        "addr_group_idx": worker.addr_group_idx,
    }


def _transfer_record(
    receiver: Any,
    req_meta: dict[str, Any],
    mamba_spec: type,
) -> dict[str, Any]:
    remote_block_ids = [list(ids) for ids in req_meta["remote_block_ids"]]
    local_block_ids = [list(ids) for ids in req_meta["local_block_ids"]]
    effective_remote_block_ids = []
    transfer_groups = []
    for group_index in range(receiver.hma_group_size):
        remote_ids = remote_block_ids[group_index]
        local_ids = local_block_ids[group_index]
        if not isinstance(receiver.kv_cache_specs[group_index], mamba_spec) and len(
            local_ids
        ) < len(remote_ids):
            remote_ids = remote_ids[-len(local_ids) :]
        effective_remote_block_ids.append(remote_ids)
        transfer_groups.append(
            {
                "group_index": group_index,
                "remote_block_ids": remote_ids,
                "local_block_ids": local_ids,
                "paired_block_ids": list(zip(remote_ids, local_ids, strict=False)),
            }
        )
    return {
        "request_id": req_meta["request_id"],
        "remote_request_id": req_meta["remote_request_id"],
        "remote_engine_id": req_meta["remote_engine_id"],
        "local_engine_id": receiver.local_engine_id,
        "tp_rank": receiver.tp_rank,
        "remote_host": req_meta["remote_host"],
        "remote_handshake_port": req_meta["remote_handshake_port"],
        "remote_block_ids": remote_block_ids,
        "effective_remote_block_ids": effective_remote_block_ids,
        "local_block_ids": local_block_ids,
        "transfer_groups": transfer_groups,
    }


def _install_mooncake_patches(connector_module: Any) -> None:
    scheduler_cls = connector_module.MooncakeConnectorScheduler
    worker_cls = connector_module.MooncakeConnectorWorker
    receiver_cls = connector_module.KVCacheRecvingThread
    if scheduler_cls.__dict__.get("_afd_pd_debug_installed", False):
        return

    original_request_finished = scheduler_cls.request_finished_all_groups
    original_register_kv_caches = worker_cls.register_kv_caches
    original_transfer = receiver_cls._transfer_kv_cache_all_groups

    # Patch reason: the native Mooncake scheduler does not expose its final
    # compressed/hybrid block selection outside the process.
    # Patch functionality: persist the selected producer block ids for bounded
    # diagnostics while preserving the upstream return value. This wrapper is
    # used because copying the scheduler function would couple diagnostics to
    # unrelated request-lifecycle code. Signature matches upstream.
    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        result = original_request_finished(self, request, block_ids)
        # ### PATCH START: Mooncake PD scheduler diagnostics
        try:
            transfer_params = result[1]
            if transfer_params is not None and transfer_params.get("remote_block_ids"):
                plan = {
                    "schema_version": 1,
                    "engine_id": self.engine_id,
                    "request_id": request.request_id,
                    "num_prompt_tokens": request.num_prompt_tokens,
                    "block_size": self.block_size,
                    "group_block_size": self.group_block_size,
                    "group_compress_ratio": self.group_compress_ratio,
                    "remote_block_ids": [
                        list(ids) for ids in transfer_params["remote_block_ids"]
                    ],
                }
                _write_plan(self.engine_id, request.request_id, plan)
                _write_record("scheduler_plan", plan)
        except Exception:
            _logger.exception("Failed to record Mooncake scheduler diagnostics")
        # ### PATCH END: Mooncake PD scheduler diagnostics
        return result

    # Patch reason: the native Mooncake worker registers raw KV buffers without
    # emitting enough layout information to verify compressed group ownership.
    # Patch functionality: record layout metadata and attach a producer-side
    # digest hook before delayed KV blocks are released. The original method is
    # delegated because it is large and owns Mooncake thread construction.
    # Signature matches upstream.
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        result = original_register_kv_caches(self, kv_caches)
        # ### PATCH START: Mooncake PD worker diagnostics
        try:
            _write_record("kv_registration", _registration_record(self, kv_caches))
            if self.kv_role == "kv_producer" and self.kv_send_thread is not None:
                original_update_done = (
                    self.kv_send_thread.task_tracker.update_done_task_count
                )

                def update_done_task_count(request_id: str):
                    try:
                        if _claim_request(self, request_id):
                            plan = _read_plan(self.engine_id, request_id)
                            if plan is None:
                                _write_record(
                                    "producer_plan_missing",
                                    {
                                        "role": "producer",
                                        "engine_id": self.engine_id,
                                        "request_id": request_id,
                                        "tp_rank": self.tp_rank,
                                    },
                                )
                            else:
                                summaries = _summarize_group_blocks(
                                    self.kv_caches,
                                    self.kv_cache_config,
                                    plan["remote_block_ids"],
                                    split_tensor=self.use_compress,
                                )
                                _write_record(
                                    "producer_kv_digest",
                                    {
                                        "role": "producer",
                                        "engine_id": self.engine_id,
                                        "request_id": request_id,
                                        "tp_rank": self.tp_rank,
                                        "groups": summaries,
                                    },
                                )
                    except Exception:
                        _logger.exception(
                            "Failed to record Mooncake producer KV diagnostics"
                        )
                    return original_update_done(request_id)

                self.kv_send_thread.task_tracker.update_done_task_count = (
                    update_done_task_count
                )
        except Exception:
            _logger.exception("Failed to initialize Mooncake worker diagnostics")
        # ### PATCH END: Mooncake PD worker diagnostics
        return result

    # Patch reason: the native Mooncake receiver logs transfer duration but not
    # its effective producer-to-consumer block mapping or received KV contents.
    # Patch functionality: record that mapping and bounded consumer block
    # digests after a successful transfer. The original transfer remains the
    # only implementation of data movement. Signature matches upstream.
    def _transfer_kv_cache_all_groups(self, req_meta: dict[str, Any]):
        # ### PATCH START: Mooncake PD receiver diagnostics
        should_record = _claim_request(self, req_meta["request_id"])
        transfer_record = None
        if should_record:
            try:
                transfer_record = _transfer_record(
                    self, req_meta, connector_module.MambaSpec
                )
                _write_record("consumer_transfer_plan", transfer_record)
            except Exception:
                _logger.exception("Failed to record Mooncake transfer plan")
        # ### PATCH END: Mooncake PD receiver diagnostics
        result = original_transfer(self, req_meta)
        # ### PATCH START: Mooncake PD receiver diagnostics
        if should_record:
            try:
                summaries = _summarize_group_blocks(
                    self.kv_caches,
                    self.kv_cache_config,
                    req_meta["local_block_ids"],
                    split_tensor=self.use_compress,
                )
                _write_record(
                    "consumer_kv_digest",
                    {
                        "role": "consumer",
                        "local_engine_id": self.local_engine_id,
                        "remote_engine_id": req_meta["remote_engine_id"],
                        "request_id": req_meta["request_id"],
                        "remote_request_id": req_meta["remote_request_id"],
                        "tp_rank": self.tp_rank,
                        "groups": summaries,
                    },
                )
            except Exception:
                _logger.exception("Failed to record Mooncake consumer KV diagnostics")
        # ### PATCH END: Mooncake PD receiver diagnostics
        return result

    scheduler_cls.request_finished_all_groups = request_finished_all_groups
    worker_cls.register_kv_caches = register_kv_caches
    receiver_cls._transfer_kv_cache_all_groups = _transfer_kv_cache_all_groups
    scheduler_cls._afd_pd_debug_installed = True
    worker_cls._afd_pd_debug_installed = True
    receiver_cls._afd_pd_debug_installed = True


def register_mooncake_pd_debug() -> None:
    """Install opt-in diagnostics for the pinned Mooncake hybrid connector."""
    if _debug_dir() is None:
        _logger.debug("Mooncake PD diagnostics disabled: %s is unset", DEBUG_DIR_ENV)
        return
    try:
        from vllm_ascend.distributed.kv_transfer.kv_p2p import (
            mooncake_hybrid_connector,
        )

        _install_mooncake_patches(mooncake_hybrid_connector)
        _write_record(
            "plugin_registered",
            {
                "debug_dir": str(_debug_dir()),
                "max_requests": _positive_int_env(
                    MAX_REQUESTS_ENV, DEFAULT_MAX_REQUESTS
                ),
                "max_blocks": _positive_int_env(MAX_BLOCKS_ENV, DEFAULT_MAX_BLOCKS),
                "max_bytes": _positive_int_env(MAX_BYTES_ENV, DEFAULT_MAX_BYTES),
            },
        )
        _logger.info("Mooncake PD diagnostics enabled in %s", _debug_dir())
    except Exception:
        _logger.exception("Failed to install Mooncake PD diagnostics")


__all__ = [
    "DEBUG_DIR_ENV",
    "register_mooncake_pd_debug",
]
