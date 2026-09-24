# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in per-layer diagnostics for DeepSeek-V4 chunked prefill."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import json
import logging
import os
import re
import socket
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

DEBUG_DIR_ENV = "AFD_DSV4_CHUNK_DEBUG_DIR"
DEBUG_DP_RANK_ENV = "AFD_DSV4_CHUNK_DEBUG_DP_RANK"
FORCE_SHORT_EXTEND_PREFILL_ENV = (
    "AFD_DSV4_CHUNK_FORCE_SHORT_EXTEND_PREFILL"
)
DEFAULT_DEBUG_DP_RANK = 0

_logger = logging.getLogger(__name__)
_DSA_MODULE = "vllm_ascend.attention.dsa_v1"


def _debug_dir() -> Path | None:
    value = os.environ.get(DEBUG_DIR_ENV)
    return Path(value).resolve() if value else None


def _debug_dp_rank() -> int:
    value = os.environ.get(DEBUG_DP_RANK_ENV, str(DEFAULT_DEBUG_DP_RANK))
    try:
        return int(value)
    except ValueError:
        _logger.warning("Ignoring invalid %s=%r", DEBUG_DP_RANK_ENV, value)
        return DEFAULT_DEBUG_DP_RANK


def _force_short_extend_prefill_enabled() -> bool:
    return os.environ.get(FORCE_SHORT_EXTEND_PREFILL_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:160] or "unknown"


def _prefill_metadata(attn_metadata: Any) -> Any | None:
    metadata_items = (
        attn_metadata if isinstance(attn_metadata, list) else [attn_metadata]
    )
    return next(
        (
            metadata
            for metadata in metadata_items
            if metadata is not None
            and getattr(metadata, "num_prefills", 0) > 0
            and getattr(metadata, "prefill", None) is not None
        ),
        None,
    )


def _active_metadata(attn_metadata: Any) -> Any | None:
    metadata_items = (
        attn_metadata if isinstance(attn_metadata, list) else [attn_metadata]
    )
    return next(
        (
            metadata
            for metadata in metadata_items
            if metadata is not None
            and int(getattr(metadata, "num_actual_tokens", 0)) > 0
            and (
                (
                    getattr(metadata, "num_prefills", 0) > 0
                    and getattr(metadata, "prefill", None) is not None
                )
                or (
                    getattr(metadata, "num_decodes", 0) > 0
                    and getattr(metadata, "decode", None) is not None
                )
            )
        ),
        None,
    )


def _selected_token(metadata: Any) -> tuple[str, Any, int, list[int]]:
    if metadata.num_prefills > 0 and metadata.prefill is not None:
        mode = "prefill"
        mode_metadata = metadata.prefill
        num_prefill_tokens = int(mode_metadata.query_start_loc[-1].item())
        token_index = int(metadata.num_decode_tokens) + num_prefill_tokens - 1
    else:
        mode = "decode"
        mode_metadata = metadata.decode
        token_index = int(metadata.num_decode_tokens) - 1

    positions = mode_metadata.input_positions.detach().cpu().reshape(-1).tolist()
    return mode, mode_metadata, token_index, positions


def _to_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _capture_record(
    instance: Any,
    layer_name: str,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    attn_metadata: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]] | None:
    metadata = _active_metadata(attn_metadata)
    if metadata is None:
        return None

    mode, mode_metadata, token_index, positions = _selected_token(metadata)
    if token_index < 0 or not positions:
        return None
    position = int(positions[-1])
    call_index = instance.__dict__.get("_afd_chunk_debug_forward_calls", 0)
    instance.__dict__["_afd_chunk_debug_forward_calls"] = call_index + 1

    record = {
        "schema_version": 2,
        "timestamp_ns": time.time_ns(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "dp_rank": int(instance.vllm_config.parallel_config.data_parallel_rank),
        "layer_name": layer_name,
        "compress_ratio": int(instance.compress_ratio),
        "attention_mode": mode,
        "forward_call_index": call_index,
        "num_decodes": int(metadata.num_decodes),
        "num_prefills": int(metadata.num_prefills),
        "num_actual_tokens": int(metadata.num_actual_tokens),
        "num_decode_tokens": int(metadata.num_decode_tokens),
        "query_start_loc": mode_metadata.query_start_loc.detach()
        .cpu()
        .reshape(-1)
        .tolist(),
        "seq_lens": mode_metadata.seq_lens.detach().cpu().reshape(-1).tolist(),
        "start_pos": mode_metadata.start_pos.detach().cpu().reshape(-1).tolist(),
        "positions": positions,
        "selected_position": position,
        "selected_token_index": token_index,
    }
    tensors = {
        "hidden": _to_cpu_float(hidden_states[token_index]),
        "attention_output": _to_cpu_float(output[token_index]),
    }
    return record, tensors


def _write_capture(record: dict[str, Any], tensors: dict[str, torch.Tensor]) -> None:
    output_dir = _debug_dir()
    if output_dir is None:
        return
    rank_dir = output_dir / f"dp{record['dp_rank']}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    stem = "--".join(
        (
            _safe_name(record["layer_name"]),
            f"call{record['forward_call_index']}",
            record["attention_mode"],
            f"pos{record['selected_position']}",
            f"pid{record['pid']}",
        )
    )
    tensor_path = rank_dir / f"{stem}.pt"
    metadata_path = rank_dir / f"{stem}.json"
    torch.save(tensors, tensor_path)
    metadata_path.write_text(
        json.dumps(record, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _install_chunk_debug_patch(dsa_module: Any) -> None:
    if _force_short_extend_prefill_enabled():
        _install_short_extend_prefill_probe(dsa_module)

    impl_cls = dsa_module.AscendDSAImpl
    if impl_cls.__dict__.get("_afd_chunk_debug_installed", False):
        return
    original_forward = impl_cls.forward

    # Upstream commit: vLLM-Ascend 3da28f9414583d2d0b672a8f06d1fae142404bda.
    # Patch reason: the native implementation exposes only final request output,
    # which cannot identify the first DSA layer affected by chunk continuation.
    # Patch functionality: capture the final real prefill token before and after
    # each DSA attention layer. The original method remains the sole executor.
    def forward(
        self,
        layer_name,
        hidden_states,
        kv_cache,
        attn_metadata,
        need_gather_q_kv=False,
        output=None,
    ):
        # ### PATCH START: bounded DeepSeek-V4 chunk diagnostics
        should_capture = (
            _debug_dir() is not None
            and int(self.vllm_config.parallel_config.data_parallel_rank)
            == _debug_dp_rank()
            and _active_metadata(attn_metadata) is not None
        )
        hidden_snapshot = None
        if should_capture:
            try:
                metadata = _active_metadata(attn_metadata)
                _, _, token_index, _ = _selected_token(metadata)
                hidden_snapshot = _to_cpu_float(hidden_states[token_index])
            except Exception:
                _logger.exception("Failed to snapshot DSV4 layer input")
        # ### PATCH END: bounded DeepSeek-V4 chunk diagnostics
        result = original_forward(
            self,
            layer_name,
            hidden_states,
            kv_cache,
            attn_metadata,
            need_gather_q_kv,
            output,
        )
        # ### PATCH START: bounded DeepSeek-V4 chunk diagnostics
        if should_capture:
            try:
                captured = _capture_record(
                    self,
                    layer_name,
                    hidden_states,
                    result,
                    attn_metadata,
                )
                if captured is not None:
                    record, tensors = captured
                    if hidden_snapshot is not None:
                        tensors["hidden"] = hidden_snapshot
                    _write_capture(record, tensors)
            except Exception:
                _logger.exception("Failed to record DSV4 layer diagnostics")
        # ### PATCH END: bounded DeepSeek-V4 chunk diagnostics
        return result

    impl_cls.forward = forward
    impl_cls._afd_chunk_debug_installed = True


def _install_short_extend_prefill_probe(dsa_module: Any) -> None:
    """Force the native short-extend control onto the prefill path.

    This is an opt-in counterfactual probe for a native, non-PD service. It is
    not a supported inference mode: PD recompute intentionally classifies the
    final prompt token as decode and must not use this probe in production.
    """
    original_split = dsa_module.split_decodes_and_prefills
    if getattr(original_split, "_afd_short_extend_prefill_probe", False) is True:
        return

    @functools.wraps(original_split)
    def split_decodes_and_prefills(
        common_attn_metadata,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True,
    ):
        del treat_short_extends_as_decodes
        return original_split(
            common_attn_metadata,
            decode_threshold=decode_threshold,
            require_uniform=require_uniform,
            treat_short_extends_as_decodes=False,
        )

    split_decodes_and_prefills._afd_short_extend_prefill_probe = True
    dsa_module.split_decodes_and_prefills = split_decodes_and_prefills
    _logger.warning(
        "%s is enabled; use only for native short-extend diagnosis",
        FORCE_SHORT_EXTEND_PREFILL_ENV,
    )


class _DeferredPatchLoader(importlib.abc.Loader):
    def __init__(self, original_loader: importlib.abc.Loader):
        self.original_loader = original_loader

    def create_module(self, spec):
        create_module = getattr(self.original_loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module) -> None:
        self.original_loader.exec_module(module)
        _install_chunk_debug_patch(module)


class _DeferredPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _DSA_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _DeferredPatchLoader(spec.loader)
        with suppress(ValueError):
            sys.meta_path.remove(self)
        return spec


def register_dsv4_chunk_debug() -> None:
    """Install bounded diagnostics only when explicitly requested."""
    if _debug_dir() is None:
        return
    try:
        loaded_module = sys.modules.get(_DSA_MODULE)
        if loaded_module is not None and hasattr(loaded_module, "AscendDSAImpl"):
            _install_chunk_debug_patch(loaded_module)
        elif not any(
            isinstance(finder, _DeferredPatchFinder) for finder in sys.meta_path
        ):
            sys.meta_path.insert(0, _DeferredPatchFinder())
        _logger.info(
            "DeepSeek-V4 chunk diagnostics scheduled in %s for DP rank %d",
            _debug_dir(),
            _debug_dp_rank(),
        )
    except Exception:
        _logger.exception("Failed to install DeepSeek-V4 chunk diagnostics")


__all__ = [
    "DEBUG_DIR_ENV",
    "FORCE_SHORT_EXTEND_PREFILL_ENV",
    "register_dsv4_chunk_debug",
]
