# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Window-based AFD connector initialization for Ascend NPU.

Stage one owns only the communication resources.  The A2F/F2A data path is
deliberately left disabled until the DSV4 model integration is added.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup
from vllm.logger import init_logger

from afd_plugin.config import AFDConfig
from afd_plugin.config_utils import (
    coerce_extra_bool,
    coerce_extra_int,
    coerce_extra_positive_int,
)
from afd_plugin.connectors.base import AFDConnectorBase, ConnectorExtraInfo
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDTransferContext,
)
from afd_plugin.distributed import (
    build_window_rank_mapping,
    init_afd_process_group,
)

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class WindowAFDExtraInfo(ConnectorExtraInfo):
    """Window protocol options.

    ``micro_batch_num`` and ``async_dispatch`` are retained for the later data
    path.  Stage one requires the former to be one and does not call operators.
    """

    micro_batch_num: int = 1
    async_dispatch: bool = False
    quant_mode: int = 2

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> WindowAFDExtraInfo:
        raw = {} if raw is None else raw
        if not isinstance(raw, Mapping):
            raise TypeError(
                "WindowAFDConnector connector_extra_config must be a mapping",
            )
        allowed = {"micro_batch_num", "async_dispatch", "quant_mode"}
        unknown = sorted(str(key) for key in raw if key not in allowed)
        if unknown:
            raise ValueError(
                "unknown WindowAFDConnector connector_extra_config field(s): "
                + ", ".join(unknown),
            )
        quant_mode = coerce_extra_int(
            raw.get("quant_mode", 2),
            field_name="quant_mode",
        )
        if quant_mode not in (0, 2):
            raise ValueError(
                "WindowAFDConnector quant_mode must be 0 or 2, "
                f"got {quant_mode}",
            )
        return cls(
            micro_batch_num=coerce_extra_positive_int(
                raw.get("micro_batch_num", 1),
                field_name="micro_batch_num",
            ),
            async_dispatch=coerce_extra_bool(
                raw.get("async_dispatch", False),
                field_name="async_dispatch",
            ),
            quant_mode=quant_mode,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "micro_batch_num": self.micro_batch_num,
            "async_dispatch": self.async_dispatch,
            "quant_mode": self.quant_mode,
        }


def _align_up(value: int, alignment: int = 512) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _model_int(hf_config: Any, *names: str, default: int) -> int:
    for name in names:
        value = getattr(hf_config, name, None)
        if value is not None:
            return int(value)
    return default


def _window_sizes(
    *,
    attention_size: int,
    micro_batch_num: int,
    micro_batch_size: int,
    selected_expert_num: int,
    hidden_size: int,
    quant_mode: int,
) -> tuple[int, int, int, int]:
    """Return ``(attn_size, ffn_size, a2f_token_size, f2a_token_size)``.

    The byte formulas mirror ref/local_window_utils.py.  Stage one computes
    them once from the configured maximum batch capacity.
    """

    attn_info = _align_up(
        4 * selected_expert_num * micro_batch_size * micro_batch_num,
    )
    if quant_mode == 2:
        attn_token_size = _align_up(hidden_size + 4, 512)
    elif quant_mode == 0:
        # Non-quantized A2F payloads use the model dtype (two bytes/value).
        attn_token_size = hidden_size * 2
    else:
        raise ValueError(f"unsupported Window quant_mode={quant_mode}")
    attn_data = (
        2 * hidden_size * selected_expert_num * micro_batch_size * micro_batch_num
    )

    ffn_info = _align_up(
        4
        * (selected_expert_num * micro_batch_size + 2)
        * micro_batch_num
        * attention_size,
    )
    if quant_mode == 2:
        ffn_token_size = _align_up(hidden_size + 4, 512)
    elif quant_mode == 0:
        ffn_token_size = hidden_size * 2
    else:
        ffn_token_size = hidden_size
    ffn_data = (
        ffn_token_size
        * selected_expert_num
        * micro_batch_size
        * micro_batch_num
        * attention_size
    )
    return (
        attn_info + attn_data,
        ffn_info + ffn_data,
        attn_token_size,
        hidden_size * 2,
    )


class WindowAFDConnector(AFDConnectorBase):
    """Create the shared M2N HCCL Window and schedule context.

    The connector intentionally does not expose a connector-driven FFN loop in
    stage one.  This lets both model runners initialize resources without
    accidentally invoking unimplemented communication operators.
    """

    yield_after_attn_send = True
    data_path_ready = False
    supports_connector_driven_loop = False

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> WindowAFDExtraInfo:
        return WindowAFDExtraInfo.from_mapping(raw)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: Any,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self.mapping = build_window_rank_mapping(afd_config, role_rank)
        self.world_rank = self.mapping.world_rank
        self.world_size = self.mapping.world_size
        self.attn_size = self.mapping.attention_size
        self.ffn_size = self.mapping.ffn_size
        self.peer_ranks = self.mapping.peer_ranks
        self.process_group: ProcessGroup | None = None
        self.hccl_comm_name: str | None = None
        self.window_tensor: torch.Tensor | None = None
        self.window_size = 0
        self.window_addr = 0
        self.context_holder: Any | None = None
        self.schedule_context: torch.Tensor | None = None
        self._initialized = False

        hf_config = vllm_config.model_config.hf_config
        self.hidden_size = _model_int(hf_config, "hidden_size", default=1)
        routed_topk = _model_int(
            hf_config,
            "num_experts_per_tok",
            "num_experts_per_token",
            default=1,
        )
        shared_expert_num = _model_int(hf_config, "n_shared_experts", default=0)
        self.selected_expert_num = routed_topk + shared_expert_num
        self.expert_num = _model_int(
            hf_config,
            "n_routed_experts",
            "num_experts",
            default=1,
        ) + shared_expert_num
        scheduler_config = vllm_config.scheduler_config
        max_tokens = int(
            getattr(scheduler_config, "max_num_batched_tokens", 0)
            or getattr(scheduler_config, "max_num_seqs", 1)
            or 1,
        )
        self.micro_batch_size = max(1, math.ceil(max_tokens / self.attn_size))

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def init_afd_connector(self) -> None:
        if self._initialized:
            return
        if self.extra_info.micro_batch_num != 1:
            raise ValueError(
                "WindowAFDConnector stage one supports only micro_batch_num=1, "
                f"got {self.extra_info.micro_batch_num}",
            )

        import torch_npu

        timeout = timedelta(minutes=30)
        try:
            self.process_group = init_afd_process_group(
                backend="hccl",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.world_size,
                rank=self.world_rank,
                group_name="afd_window",
                timeout=timeout,
            )
            backend = self.process_group._get_backend(torch.device("npu"))
            getter = getattr(backend, "get_hccl_comm_name", None)
            if getter is None:
                getter = getattr(self.process_group, "get_hccl_comm_name", None)
            if getter is None:
                raise RuntimeError("HCCL ProcessGroup does not expose comm name API")
            self.hccl_comm_name = str(getter(self.world_rank))

            self.window_size = self._compute_window_size()
            backend._window_register_and_exchange(
                self.window_size,
                list(self.peer_ranks),
            )
            self.window_tensor = backend._get_window_mem()
            self.window_addr = int(self.window_tensor.data_ptr())

            _, _, a2f_token_size, f2a_token_size = _window_sizes(
                attention_size=self.attn_size,
                micro_batch_num=self.extra_info.micro_batch_num,
                micro_batch_size=self.micro_batch_size,
                selected_expert_num=self.selected_expert_num,
                hidden_size=self.hidden_size,
                quant_mode=self.extra_info.quant_mode,
            )
            context_factory = torch_npu._afd.create_schedule_context_holder
            kwargs = {
                "schedule_mode": 1 if self.afd_config.role == "attention" else 0,
                "session_num": self.attn_size,
                "micro_batch_num": self.extra_info.micro_batch_num,
                "micro_batch_size": self.micro_batch_size,
                "selected_expert_num": self.selected_expert_num,
                "expert_num": self.expert_num,
                "attn_to_ffn_token_size": a2f_token_size,
                "ffn_to_attn_token_size": f2a_token_size,
            }
            if self.afd_config.role == "attention":
                kwargs.update(
                    attention_window=self.window_addr,
                    attention_window_size=self.window_size,
                )
            else:
                kwargs.update(
                    ffn_window=self.window_addr,
                    ffn_window_size=self.window_size,
                )
            self.context_holder = context_factory(**kwargs)
            self.schedule_context = self.context_holder.get_schedule_context_tensor()
            self._initialized = True
            logger.info(
                "Window AFD initialized: role=%s role_rank=%d world_rank=%d "
                "world_size=%d peers=%s window_size=%d comm_name=%s",
                self.afd_config.role,
                self.role_rank,
                self.world_rank,
                self.world_size,
                self.peer_ranks,
                self.window_size,
                self.hccl_comm_name,
            )
        except BaseException:
            self.close()
            raise

    def _compute_window_size(self) -> int:
        attn_size, ffn_size, _, _ = _window_sizes(
            attention_size=self.attn_size,
            micro_batch_num=self.extra_info.micro_batch_num,
            micro_batch_size=self.micro_batch_size,
            selected_expert_num=self.selected_expert_num,
            hidden_size=self.hidden_size,
            quant_mode=self.extra_info.quant_mode,
        )
        return attn_size if self.afd_config.role == "attention" else ffn_size

    def close(self) -> None:
        holder = self.context_holder
        self.context_holder = None
        self.schedule_context = None
        if holder is not None:
            try:
                holder.stop_schedule()
            except Exception:
                pass
        group = self.process_group
        self.process_group = None
        if group is not None:
            try:
                dist.destroy_process_group(group)
            except Exception:
                pass
        self.window_tensor = None
        self.hccl_comm_name = None
        self.window_size = 0
        self.window_addr = 0
        self._initialized = False

    def _stage_two_error(self) -> RuntimeError:
        return RuntimeError(
            "WindowAFDConnector data path is not enabled in stage one",
        )

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        raise self._stage_two_error()

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise self._stage_two_error()

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        raise self._stage_two_error()

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        raise self._stage_two_error()


__all__ = ["WindowAFDConnector", "WindowAFDExtraInfo"]
