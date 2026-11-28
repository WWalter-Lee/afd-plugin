# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Plugin-owned NPU profiler helpers for AFD Ascend runners."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Final, Literal, Protocol

if TYPE_CHECKING:
    from afd_plugin.connectors.base import AFDConnectorBase

logger = logging.getLogger(__name__)

AFDNPUProfilerRole = Literal["attention", "ffn"]

_ENV_PREFIX: Final[dict[AFDNPUProfilerRole, str]] = {
    "attention": "AFD_NPU_ATTENTION_PROFILER",
    "ffn": "AFD_NPU_FFN_PROFILER",
}
_DEFAULT_DIR: Final[dict[AFDNPUProfilerRole, str]] = {
    "attention": "/tmp/profile/attn",
    "ffn": "/tmp/profile/ffn",
}
_DEFAULT_ACTIVE_STEPS: Final[dict[AFDNPUProfilerRole, int]] = {
    "attention": 10,
    "ffn": 20,
}
_DEFAULT_WAIT_STEPS: Final[int] = 2
_DEFAULT_WARMUP_STEPS: Final[int] = 1
_DEFAULT_REPEAT: Final[int] = 1
_DEFAULT_SKIP_FIRST_STEPS: Final[int] = 1500
_VLLM_TORCH_PROFILER_DIR_ENV: Final[str] = "VLLM_TORCH_PROFILER_DIR"


@dataclass(frozen=True)
class AFDNPUProfilerConfig:
    enabled: bool
    wait: int
    warmup: int
    active: int
    repeat: int
    skip_first: int
    trace_dir: str
    with_stack: bool
    ranks: frozenset[int] | None = frozenset({0})


class AFDNPUProfiler(Protocol):
    def step(self) -> None: ...

    def stop(self) -> None: ...


def afd_npu_profiler_config(role: AFDNPUProfilerRole) -> AFDNPUProfilerConfig:
    """Read plugin-owned profiler settings for an AFD NPU runner role."""

    prefix = _ENV_PREFIX[role]
    return AFDNPUProfilerConfig(
        enabled=_env_bool(f"{prefix}_ENABLE", default=False),
        wait=_env_int(f"{prefix}_WAIT", default=_DEFAULT_WAIT_STEPS),
        warmup=_env_int(f"{prefix}_WARMUP", default=_DEFAULT_WARMUP_STEPS),
        active=_env_int(f"{prefix}_ACTIVE", default=_DEFAULT_ACTIVE_STEPS[role]),
        repeat=_env_int(f"{prefix}_REPEAT", default=_DEFAULT_REPEAT),
        skip_first=_env_int(
            f"{prefix}_SKIP_FIRST",
            default=_DEFAULT_SKIP_FIRST_STEPS,
        ),
        trace_dir=_env_dir(f"{prefix}_DIR", default=_DEFAULT_DIR[role]),
        ranks=_env_ranks(f"{prefix}_RANKS"),
        with_stack=_env_bool(f"{prefix}_WITH_STACK", default=False),
    )


def create_afd_npu_profiler(
    role: AFDNPUProfilerRole, *, role_rank: int = 0
) -> AFDNPUProfiler | None:
    """Create a torch-npu profiler when the plugin-owned env enables it."""

    config = afd_npu_profiler_config(role)
    if not config.enabled or (
        config.ranks is not None and role_rank not in config.ranks
    ):
        return None
    return _start_trace(role, role_rank, config)


def _start_trace(
    role: AFDNPUProfilerRole, role_rank: int, config: AFDNPUProfilerConfig
) -> AFDNPUProfiler:
    import torch_npu

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    )
    logger.info(
        "AFD NPU %s rank %d profiler enabled: dir=%s, with_stack=%s",
        role,
        role_rank,
        config.trace_dir,
        config.with_stack,
    )
    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=config.wait,
            warmup=config.warmup,
            active=config.active,
            repeat=config.repeat,
            skip_first=config.skip_first,
        ),
        with_stack=config.with_stack,
        with_modules=config.with_stack,
        record_shapes=True,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            config.trace_dir,
        ),
    )
    profiler.start()
    return profiler


class AFDNPUProfilerSession:
    """One runner's profiler controlled by explicit profile RPCs."""

    def __init__(
        self, role: AFDNPUProfilerRole, role_rank: int, config: AFDNPUProfilerConfig
    ) -> None:
        self.role = role
        self.role_rank = role_rank
        self.config = config
        self._trace: AFDNPUProfiler | None = None
        self._selected = config.ranks is None or role_rank in config.ranks
        self._generation = 0
        self._requested: int | None = None
        self._active: int | None = None
        self._lock = Lock()

    def request(self, is_start: bool) -> None:
        """Arm/disarm FFN from RPC; the connector thread owns torch-npu TLS."""
        if not self._selected:
            return
        with self._lock:
            if is_start and self._requested is None:
                self._generation += 1
                self._requested = self._generation
            elif not is_start:
                self._requested = None
        if self.role == "attention":
            self._apply_request()

    def _apply_request(self) -> None:
        with self._lock:
            requested = self._requested
        if requested == self._active:
            return
        self.stop()
        if requested is not None:
            self._trace = _start_trace(self.role, self.role_rank, self.config)
            self._active = requested

    def step(self) -> None:
        if self.role == "ffn" and self._selected:
            self._apply_request()
        step_afd_npu_profiler(self._trace)

    def stop(self) -> None:
        """Close the live trace on its owner thread, including loop exit."""
        trace = self._trace
        self._trace = None
        self._active = None
        stop_afd_npu_profiler(trace)


def create_afd_npu_profiler_session(
    role: AFDNPUProfilerRole, connector: AFDConnectorBase
) -> AFDNPUProfilerSession | None:
    config = afd_npu_profiler_config(role)
    if not config.enabled:
        return None
    return AFDNPUProfilerSession(role, connector.role_rank, config)


def step_afd_npu_profiler(profiler: AFDNPUProfiler | None) -> None:
    if profiler is not None:
        profiler.step()


def stop_afd_npu_profiler(profiler: AFDNPUProfiler | None) -> None:
    if profiler is not None:
        profiler.stop()


def _env_ranks(name: str) -> frozenset[int] | None:
    value = os.getenv(name, "0").strip()
    if value.lower() == "all":
        return None
    try:
        ranks = frozenset(int(rank.strip()) for rank in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{name} expects 'all' or comma-separated ranks") from exc
    if any(rank < 0 for rank in ranks):
        raise ValueError(f"{name} expects non-negative role ranks")
    return ranks


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_dir(name: str, *, default: str) -> str:
    return os.getenv(name) or os.getenv(_VLLM_TORCH_PROFILER_DIR_ENV) or default


__all__ = [
    "AFDNPUProfilerConfig",
    "AFDNPUProfiler",
    "afd_npu_profiler_config",
    "create_afd_npu_profiler",
    "create_afd_npu_profiler_session",
    "step_afd_npu_profiler",
    "stop_afd_npu_profiler",
]
