# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Window-based AFD connector initialization for Ascend NPU.

The connector owns the communication resources and the synchronous A2F/F2A
data path used by the initial A3 implementation.
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
    AFDTransferMetadata,
    AFDTransferState,
    AFDTransferContext,
)
from afd_plugin.distributed import (
    build_window_rank_mapping,
    init_afd_process_group,
)
from afd_plugin.connectors.npu import window_ops

logger = init_logger(__name__)


@dataclass(slots=True)
class WindowAFDTransferState(AFDTransferState):
    """Operator-produced routing metadata for one A2F exchange."""

    expert_scales: torch.Tensor
    group_list: torch.Tensor | None = None
    dynamic_scale: torch.Tensor | None = None
    session_ids: torch.Tensor | None = None
    micro_batch_ids: torch.Tensor | None = None
    token_ids: torch.Tensor | None = None
    expert_offsets: torch.Tensor | None = None
    actual_token_num: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class WindowAFDExtraInfo(ConnectorExtraInfo):
    """Window protocol options.

    ``micro_batch_num`` and ``async_dispatch`` are retained for the common
    connector configuration.  The initial implementation requires one
    micro-batch and uses the synchronous Window operators.
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

    The connector owns both the M2N Window resources and the synchronous
    operator data path used by the initial A3 implementation.
    """

    yield_after_attn_send = True
    data_path_ready = False
    supports_connector_driven_loop = True
    is_window_connector = True

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
        self.expert_rank_table: torch.Tensor | None = None
        self.attn_rank_table: torch.Tensor | None = None
        self.local_expert_num = 0
        self.local_routed_expert_num = 0
        self.shared_expert_local_start: int | None = None
        self._pending_transfers: dict[tuple[int, int], AFDTransferContext] = {}
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
        self.shared_expert_num = shared_expert_num
        self.mix_placement = bool(
            getattr(vllm_config, "additional_config", {}).get(
                "mix_placement",
                False,
            )
        )
        self.routed_expert_num = _model_int(
            hf_config,
            "n_routed_experts",
            "num_experts",
            default=1,
        )
        self.selected_expert_num = routed_topk + shared_expert_num
        self.expert_num = self.routed_expert_num + shared_expert_num
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
        if not self.afd_config.compute_gate_on_attention:
            raise ValueError(
                "WindowAFDConnector requires compute_gate_on_attention=true "
                "for the ref-style Attention-to-FFN route",
            )
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
            self._build_rank_tables()
            self.data_path_ready = True
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
        self._pending_transfers.clear()
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
        self.data_path_ready = False
        self._initialized = False

    def _build_rank_tables(self) -> None:
        """Build a compatibility expert map for a uniform EP=1 deployment.

        The native/ref implementation builds this table from each FFN rank's
        actual local expert map.  ``set_expert_rank_table`` can replace this
        compatibility map once that model-owned map is available.  Keeping a
        fallback is useful for the first uniform deployment, but it is
        intentionally not used to claim support for EPLB or uneven placement.
        """
        device = self.window_tensor.device if self.window_tensor is not None else "npu"
        table = torch.zeros(
            (1, self.expert_num, 3), dtype=torch.int32, device=device
        )
        routed_num = max(0, self.expert_num - self.shared_expert_num)
        base_experts, remainder = divmod(routed_num, self.ffn_size)
        for expert_id in range(routed_num):
            # Match the native linear EP placement: the first ``remainder``
            # FFN ranks own one extra routed expert.
            wide_rank_span = (base_experts + 1) * remainder
            if remainder and expert_id < wide_rank_span:
                ffn_rank = expert_id // (base_experts + 1)
                local_id = expert_id % (base_experts + 1)
            else:
                tail_offset = expert_id - wide_rank_span
                ffn_rank = remainder + tail_offset // max(base_experts, 1)
                local_id = tail_offset % max(base_experts, 1)
            table[0, expert_id, 0] = 1
            table[0, expert_id, 1] = ffn_rank
            table[0, expert_id, 2] = local_id
        # Shared experts, when present, are represented by the final IDs and
        # are placed on the first FFN rank for the initial EP=1 path.
        for expert_id in range(routed_num, self.expert_num):
            table[0, expert_id, 0] = 1
            table[0, expert_id, 1] = 0
            table[0, expert_id, 2] = (
                base_experts + (1 if remainder > 0 else 0) + expert_id - routed_num
            )
        self.expert_rank_table = table
        self.local_routed_expert_num = base_experts + (1 if self.role_rank < remainder else 0)
        self.shared_expert_local_start = self.local_routed_expert_num
        self.local_expert_num = self.local_routed_expert_num + (
            self.shared_expert_num if self.role_rank == 0 else 0
        )
        self.attn_rank_table = torch.arange(
            self.attn_size,
            dtype=torch.int32,
            device=device,
        ) + self.ffn_size

    def set_expert_rank_table(self, local_expert_ids: torch.Tensor | None) -> None:
        """Inject the ref-style table generated from the real FFN placement.

        ``local_expert_ids`` is a one-dimensional tensor whose index is the
        local expert slot and whose value is the global logical expert id.
        All A+F ranks must call this method so the fixed-size all-gather is
        collective; Attention ranks pass ``None`` and contribute ``-1``.
        """
        if self.process_group is None or self.window_tensor is None:
            raise RuntimeError("Window connector must be initialized first")
        if local_expert_ids is not None:
            local_expert_ids = local_expert_ids.reshape(-1).to(
                device=self.window_tensor.device,
                dtype=torch.int32,
            )
            # Separate shared experts are represented as additional local
            # slots on the first FFN rank, exactly as in the ref deployment.
            # They are not routed IDs and therefore are never sent in
            # ``expert_ids`` by Attention.
            if (
                self.afd_config.role == "ffn"
                and self.shared_expert_num > 0
                and not self.mix_placement
                and self.role_rank == 0
            ):
                shared_ids = torch.arange(
                    self.routed_expert_num,
                    self.routed_expert_num + self.shared_expert_num,
                    dtype=torch.int32,
                    device=self.window_tensor.device,
                )
                local_expert_ids = torch.cat(
                    (local_expert_ids, shared_ids),
                    dim=0,
                )
        local_count = max(
            int(local_expert_ids.numel()) if local_expert_ids is not None else 0,
            1,
        )
        count_tensor = torch.tensor(
            [local_count], dtype=torch.int32, device=self.window_tensor.device
        )
        counts = torch.empty(
            self.world_size, dtype=torch.int32, device=self.window_tensor.device
        )
        dist.all_gather_into_tensor(counts, count_tensor, group=self.process_group)
        max_count = int(counts.max().item())
        local_table = torch.full(
            (1, max_count), -1, dtype=torch.int32, device=self.window_tensor.device
        )
        if local_expert_ids is not None and local_expert_ids.numel():
            local_table[0, : local_expert_ids.numel()] = local_expert_ids
        gathered = torch.empty(
            (self.world_size, 1, max_count),
            dtype=torch.int32,
            device=self.window_tensor.device,
        )
        dist.all_gather_into_tensor(gathered, local_table, group=self.process_group)

        table = torch.zeros(
            (1, self.expert_num, max(3, 2 * self.ffn_size + 1)),
            dtype=torch.int32,
            device=self.window_tensor.device,
        )
        for rank_id in range(self.world_size):
            if rank_id < self.ffn_size:
                rank_value = rank_id
            else:
                # Attention ranks contribute only padding and are ignored.
                continue
            for local_id, global_id in enumerate(gathered[rank_id, 0].tolist()):
                if global_id < 0 or global_id >= self.expert_num:
                    continue
                instance_count = int(table[0, global_id, 0].item()) + 1
                if instance_count * 2 >= table.shape[-1]:
                    raise RuntimeError(
                        "expert_rank_table has more instances than its allocated width"
                    )
                table[0, global_id, 0] = instance_count
                table[0, global_id, instance_count * 2 - 1] = rank_value
                table[0, global_id, instance_count * 2] = local_id
        self.expert_rank_table = table
        self.local_routed_expert_num = max(
            0,
            local_count - (
                self.shared_expert_num
                if self.afd_config.role == "ffn"
                and self.shared_expert_num > 0
                and not self.mix_placement
                and self.role_rank == 0
                else 0
            ),
        )
        self.shared_expert_local_start = self.local_routed_expert_num
        self.local_expert_num = local_count

    def configure_expert_rank_table_from_model(self, model: Any) -> None:
        """Build the rank table from the model's loaded local expert map.

        ``AscendFusedMoE._expert_map`` is indexed by global logical expert and
        stores the local expert slot, with ``-1`` for experts absent on this
        rank.  The Window operator needs the inverse relation, so the local
        slot order is reconstructed before the collective gather.  Attention
        ranks intentionally contribute ``None``.
        """
        # The initial Window implementation uses the uniform fallback map for
        # separate shared experts.  Model-map discovery is a collective and
        # cannot be started from load_model(), because Attention and FFN load
        # at different points in their worker lifecycles.
        if self.shared_expert_num > 0 and not self.mix_placement:
            return
        if self.process_group is None or self.window_tensor is None:
            # FFN load_model can precede init_afd_connector.  The fallback map
            # built during connector initialization remains valid for the
            # uniform stage-one deployment.
            return
        local_expert_ids: torch.Tensor | None = None
        has_model_map = False
        if self.afd_config.role == "ffn":
            for experts in getattr(model, "moe_layers", ()):
                expert_map = getattr(experts, "_expert_map", None)
                if expert_map is None:
                    continue
                has_model_map = True
                expert_map = expert_map.reshape(-1)
                present = torch.where(expert_map >= 0)[0]
                if present.numel():
                    local_slots = expert_map[present].to(torch.int64)
                    order = torch.argsort(local_slots)
                    local_expert_ids = present[order].to(torch.int32)
                break
            if has_model_map and local_expert_ids is None:
                local_expert_ids = torch.empty(
                    (0,), dtype=torch.int32, device=self.window_tensor.device
                )
            if (
                self.shared_expert_num > 0
                and not self.mix_placement
                and self.role_rank == 0
            ):
                assert local_expert_ids is not None
                shared_ids = torch.arange(
                    self.routed_expert_num,
                    self.routed_expert_num + self.shared_expert_num,
                    dtype=torch.int32,
                    device=local_expert_ids.device,
                )
                local_expert_ids = torch.cat(
                    (local_expert_ids, shared_ids),
                    dim=0,
                )
        map_flag = torch.tensor(
            [int(has_model_map)], dtype=torch.int32, device=self.window_tensor.device
        )
        dist.all_reduce(map_flag, op=dist.ReduceOp.SUM, group=self.process_group)
        if int(map_flag.item()) == 0:
            # No rank exposes an EPLB/physical map.  Keep the uniform
            # compatibility table created during connector initialization.
            return
        self.set_expert_rank_table(local_expert_ids)

    def _operator_shapes(self, batch_size: int) -> tuple[list[int], list[int], list[int], list[int]]:
        quant_mode = self.extra_info.quant_mode
        token_size = _align_up(self.hidden_size + 4, 512) if quant_mode == 2 else self.hidden_size * 2
        ffn_info = [self.attn_size, 1, 2 + batch_size * self.selected_expert_num]
        ffn_data = [self.attn_size, 1, batch_size, self.selected_expert_num, token_size]
        attn_info = [1, batch_size, self.selected_expert_num]
        attn_data = [1, batch_size, self.selected_expert_num, token_size]
        return ffn_info, ffn_data, attn_info, attn_data

    def _token_dtype(self) -> int:
        if self.extra_info.quant_mode == 2:
            return 2
        return 1 if self.vllm_config.model_config.dtype == torch.bfloat16 else 0

    @staticmethod
    def _token_dtype_for_tensor(tensor: torch.Tensor) -> int:
        if tensor.dtype == torch.bfloat16:
            return 1
        if tensor.dtype == torch.float16:
            return 0
        raise RuntimeError(
            "Window combine requires float16 or bfloat16 reference tensor, "
            f"got {tensor.dtype}",
        )

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        self._require_data_path()
        expert_ids = kwargs.get("expert_ids")
        expert_scales = kwargs.get("expert_scales")
        if expert_ids is None or expert_scales is None:
            raise RuntimeError("Window A2F requires expert_ids and expert_scales")
        batch_size = int(hidden_states.shape[0])
        expert_ids = expert_ids.to(torch.int32).reshape(batch_size, -1)
        expert_scales = expert_scales.to(torch.float32).reshape(batch_size, -1)

        # A2F receives only routed top-k IDs.  The shared-expert slot is
        # represented by selected_expert_num (K + shared) in the Window
        # layout and rank table, not by appending a shared ID to expert_ids.
        # Some gate paths historically included that extra column, so strip
        # it at this connector boundary without changing the P2P path.
        routed_topk = self.selected_expert_num - self.shared_expert_num
        if expert_ids.shape[1] < routed_topk:
            raise RuntimeError(
                "Window A2F received fewer routed expert IDs than configured: "
                f"got {expert_ids.shape[1]}, expected {routed_topk}",
            )
        if expert_scales.shape[1] < routed_topk:
            raise RuntimeError(
                "Window A2F received fewer routed expert scales than configured: "
                f"got {expert_scales.shape[1]}, expected {routed_topk}",
            )
        if expert_ids.shape[1] != routed_topk:
            logger.debug(
                "Window A2F dropping %d non-routed expert ID columns",
                expert_ids.shape[1] - routed_topk,
            )
        expert_ids = expert_ids[:, :routed_topk]
        expert_scales = expert_scales[:, :routed_topk]
        combine_scales = expert_scales
        ffn_info, ffn_data, attn_info, _ = self._operator_shapes(batch_size)
        x = hidden_states.reshape(1, batch_size, self.hidden_size)
        session_id = torch.tensor([self.role_rank], dtype=torch.int32, device=x.device)
        micro_batch_id = torch.tensor([int(kwargs.get("micro_batch_id", 0))], dtype=torch.int32, device=x.device)
        # The Window A2F operator models one active MoE layer per invocation;
        # the model layer index is carried by the surrounding execution order.
        layer_id = torch.zeros((1,), dtype=torch.int32, device=x.device)
        debug_expert_ids = expert_ids.reshape(-1, expert_ids.shape[-1])
        print(
            "[A2F][before] "
            f"rank={self.world_rank} "
            f"session={session_id.tolist()} "
            f"micro_batch={micro_batch_id.tolist()} "
            f"layer={layer_id.tolist()} "
            f"x_shape={tuple(x.shape)} "
            f"expert_ids_shape={tuple(expert_ids.shape)} "
            f"expert_ids_first_row={debug_expert_ids[0].tolist()} "
            f"moe_expert_num={self.routed_expert_num} "
            f"ffn_info={ffn_info} "
            f"ffn_data={ffn_data} "
            f"attn_info={attn_info}",
            flush=True,
        )
        window_ops.attention_to_ffn(
            x,
            session_id,
            micro_batch_id,
            layer_id,
            expert_ids.reshape(1, batch_size, -1),
            self.expert_rank_table,
            self.hccl_comm_name,
            self.world_size,
            ffn_info,
            ffn_data,
            attn_info,
            self.routed_expert_num,
            quant_mode=self.extra_info.quant_mode,
            sync_flag=0,
            ffn_start_rank_id=0,
        )
        print(
            f"[A2F][returned] rank={self.world_rank}",
            flush=True,
        )
        torch.npu.synchronize()
        print(
            f"[A2F][sync_done] rank={self.world_rank}",
            flush=True,
        )
        logger.debug(
            "Window A2F sent layer=%d stage=%d batch=%d topk=%d",
            context.metadata.layer_idx,
            context.metadata.stage_idx,
            batch_size,
            expert_ids.shape[-1],
        )
        self._pending_transfers[(int(context.metadata.stage_idx), int(context.metadata.layer_idx))] = context
        state = WindowAFDTransferState(expert_scales=combine_scales)
        context.states = state

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._require_data_path()
        key = (int(ubatch_idx), int(kwargs.get("layer_idx", 0)))
        context = self._pending_transfers.pop(key, None)
        if context is None or not isinstance(context.states, WindowAFDTransferState):
            raise RuntimeError(f"Window F2A has no pending transfer for {key}")
        _, _, attn_info, _ = self._operator_shapes(int(ref_tensor.shape[0]))
        output, _ = window_ops.attention_worker_combine(
            self.schedule_context,
            context.states.expert_scales,
            torch.tensor([key[1]], dtype=torch.int32, device=ref_tensor.device),
            self.hidden_size,
            # ``token_dtype=2`` is only the INT8 payload mode of
            # ``ffn_worker_batching``.  ``attention_worker_combine`` accepts
            # only the output dtype modes: 0=FP16 and 1=BF16.  Use the same
            # dtype as the Attention continuation/residual, as P2P does.
            token_dtype=self._token_dtype_for_tensor(ref_tensor),
            need_schedule=1,
        )
        return output.reshape_as(ref_tensor)

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        self._require_data_path()
        batch_size = self.micro_batch_size
        _, _, _, _ = self._operator_shapes(batch_size)
        # The operator expects the logical dimensions [A, BS, K+1, H].
        # Its tiling validates K+1 independently (currently <= 64); the
        # product A*BS*(K+1) is computed internally for the output rows.
        max_out_shape = [
            self.attn_size,
            batch_size,
            self.selected_expert_num,
            self.hidden_size,
        ]
        print(
            "[Window][before_batching] "
            f"rank={self.world_rank} "
            f"max_out_shape={max_out_shape} "
            f"expert_num={self.local_expert_num} "
            f"token_dtype={self._token_dtype()} "
            f"need_schedule=1",
            flush=True,
        )
        outputs = window_ops.ffn_worker_batching(
            self.schedule_context,
            getattr(self, "local_expert_num", max(1, math.ceil(self.expert_num / self.ffn_size))),
            max_out_shape,
            token_dtype=self._token_dtype(),
            need_schedule=1,
            layer_num=0,
        )
        print(
            f"[Window][batching_returned] rank={self.world_rank} "
            f"hidden_states_shape={tuple(outputs[0].shape)} "
            f"actual_token_num_shape={tuple(outputs[-1].shape)}",
            flush=True,
        )
        torch.npu.synchronize()
        actual_token_num = outputs[-1].detach().cpu().reshape(-1).tolist()
        group_list = outputs[1].detach().cpu()
        nonzero_group_list = group_list[group_list[:, 1] > 0].tolist()
        actual_num = int(actual_token_num[0]) if actual_token_num else 0
        print(
            f"[Window][batching_sync_done] rank={self.world_rank} "
            f"actual_token_num={actual_token_num} "
            f"nonzero_group_list={nonzero_group_list[:32]} "
            f"session_ids_head={outputs[2][:min(actual_num, 16)].detach().cpu().tolist()} "
            f"token_ids_head={outputs[4][:min(actual_num, 16)].detach().cpu().tolist()}",
            flush=True,
        )
        logger.debug(
            "Window FFN batching completed layer=%d stage=%d",
            int(kwargs.get("layer_idx", 0)),
            ubatch_idx,
        )
        hidden_states, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale, actual_token_num = outputs
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_ffn_metadata(
                layer_idx=int(kwargs.get("layer_idx", 0)),
                stage_idx=int(ubatch_idx),
                seq_lens=[int(hidden_states.shape[0])],
            ),
            states=WindowAFDTransferState(
                expert_scales=torch.empty((0,), dtype=torch.float32, device=hidden_states.device),
                group_list=group_list,
                dynamic_scale=dynamic_scale,
                session_ids=session_ids,
                micro_batch_ids=micro_batch_ids,
                token_ids=token_ids,
                expert_offsets=expert_offsets,
                actual_token_num=actual_token_num,
            ),
        )
        return AFDA2FTransferPayload(hidden_states=hidden_states, context=context)

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        self._require_data_path()
        if not isinstance(context.states, WindowAFDTransferState):
            raise RuntimeError("Window F2A requires batching state")
        state = context.states
        if any(value is None for value in (state.session_ids, state.micro_batch_ids, state.token_ids, state.expert_offsets, state.actual_token_num)):
            raise RuntimeError("Window batching did not return complete routing metadata")

        # FFNWorkerBatching returns tensors allocated for the configured
        # maximum capacity.  FFNToAttention, however, requires x and every
        # per-token metadata tensor to have the current effective length Y,
        # which is reported by actual_token_num.  Keep the Window shapes
        # unchanged; trim only the payload sent by this invocation.
        actual_token_num = state.actual_token_num.reshape(-1)
        if actual_token_num.numel() != 1:
            raise RuntimeError(
                "Window batching returned actual_token_num with unexpected "
                f"shape {tuple(state.actual_token_num.shape)}"
            )
        actual_num = int(actual_token_num.item())
        if actual_num < 0:
            raise RuntimeError(f"Window batching returned negative actual_token_num={actual_num}")

        routed_output = getattr(ffn_output, "routed_output", ffn_output)
        if routed_output.dim() != 2 or routed_output.shape[0] < actual_num:
            raise RuntimeError(
                "Window F2A output capacity is smaller than actual token count: "
                f"output_shape={tuple(routed_output.shape)} actual_num={actual_num}"
            )

        metadata = (
            state.session_ids,
            state.micro_batch_ids,
            state.token_ids,
            state.expert_offsets,
        )
        if any(value.dim() != 1 or value.shape[0] < actual_num for value in metadata):
            raise RuntimeError(
                "Window F2A metadata capacity is smaller than actual token count: "
                f"actual_num={actual_num} metadata_shapes="
                f"{[tuple(value.shape) for value in metadata]}"
            )

        routed_output = routed_output.narrow(0, 0, actual_num)
        session_ids = state.session_ids.narrow(0, 0, actual_num)
        micro_batch_ids = state.micro_batch_ids.narrow(0, 0, actual_num)
        token_ids = state.token_ids.narrow(0, 0, actual_num)
        expert_offsets = state.expert_offsets.narrow(0, 0, actual_num)
        print(
            f"[Window][before_f2a] rank={self.world_rank} "
            f"actual_num={actual_num} "
            f"x_shape={tuple(routed_output.shape)} "
            f"metadata_shapes={[tuple(value.shape) for value in (session_ids, micro_batch_ids, token_ids, expert_offsets)]}",
            flush=True,
        )
        window_ops.ffn_to_attention(
            routed_output,
            session_ids,
            micro_batch_ids,
            token_ids,
            expert_offsets,
            actual_token_num,
            self.hccl_comm_name,
            self.world_size,
            [1, self.micro_batch_size, self.selected_expert_num],
            [1, self.micro_batch_size, self.selected_expert_num, self.hidden_size],
            attn_rank_table=self.attn_rank_table,
        )
        logger.debug(
            "Window F2A sent stage=%d actual_tokens=%s",
            context.metadata.stage_idx,
            state.actual_token_num,
        )

    def _require_data_path(self) -> None:
        if not self._initialized or not self.data_path_ready:
            raise RuntimeError("WindowAFDConnector data path is not initialized")

    def select_experts(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts

        return select_experts(**kwargs)


__all__ = ["WindowAFDConnector", "WindowAFDExtraInfo", "WindowAFDTransferState"]
