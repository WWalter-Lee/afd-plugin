# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""HCCL Send/Recv connector for Attention/FFN disaggregation on NPU.

``P2pHcclAFDConnector`` deliberately uses the public PyTorch distributed
point-to-point interface for its data path. Hidden states, FFN outputs, and
DeepSeek-V4 input IDs are transferred with ``torch.distributed.send`` and
``torch.distributed.recv`` over HCCL process groups. It does not load or call
the CAMP2P A2E/E2A custom operators.

The first implementation keeps a one-to-one Attention/FFN mapping. Each DBO
stage owns an independent HCCL group so two stage threads cannot consume each
other's messages. A Gloo control group carries stage token counts before the
FFN side posts receives and prepares its receive buffers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup
from vllm.forward_context import DPMetadata, get_forward_context

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import (
    AFDConnectorBase,
    AFDControlPlane,
    ConnectorExtraInfo,
)
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDDPMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
    recv_control_payload,
    send_control_payload,
)
from afd_plugin.distributed import build_rank_mapping, init_afd_process_group
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(slots=True)
class HCCLP2PTransferState(AFDTransferState):
    """State retained between one FFN receive and its matching send."""

    stage_idx: int
    num_tokens: int


class P2pHcclAFDConnector(AFDConnectorBase):
    """Move AFD tensors with standard HCCL point-to-point operations."""

    yield_after_attn_send = False

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> ConnectorExtraInfo:
        if raw is not None and not isinstance(raw, Mapping):
            raise TypeError("HCCL P2P connector_extra_config must be a mapping")
        if raw:
            raise ValueError(
                "P2pHcclAFDConnector does not support connector_extra_config",
            )
        return ConnectorExtraInfo()

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self.mapping = build_rank_mapping(afd_config, role_rank)
        # FFN token-count routing uses the connector topology contract shared
        # with CAMP2P. Here the generic rank mapping is the complete topology.
        self.topology = self.mapping
        if self.mapping.attention_size != self.mapping.ffn_size:
            raise ValueError(
                "P2pHcclAFDConnector currently requires equal Attention and FFN "
                "ranks",
            )

        self._initialized = False
        self.world_rank = self.mapping.world_rank
        self.p2p_rank = self.mapping.p2p_rank
        self.attn_size = self.mapping.attention_size
        self.ffn_size = self.mapping.ffn_size
        self.min_size = self.mapping.min_size
        self.ratio = self.mapping.ratio
        self.dst_list = list(self.mapping.dp_metadata_destinations)
        self.hidden_size = int(vllm_config.model_config.hf_config.hidden_size)
        self.dtype = vllm_config.model_config.dtype
        self.max_num_batched_tokens = int(
            vllm_config.scheduler_config.max_num_batched_tokens,
        )
        architectures = vllm_config.model_config.hf_config.architectures or []
        self.requires_input_ids = any(
            architecture in {"DeepseekV4ForCausalLM", "AFDDeepseekV4ForCausalLM"}
            for architecture in architectures
        )
        self.vocab_size = int(vllm_config.model_config.hf_config.vocab_size)

        self.data_pg_list: list[ProcessGroup] = []
        self.ids_pg_list: list[ProcessGroup] = []
        self.p2p_pg: ProcessGroup | None = None
        self.input_ids_buffers: list[torch.Tensor] = []
        self.hidden_recv_buffers: dict[int, torch.Tensor] = {}
        self.dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] = {}
        self.is_graph_capturing = False
        self.is_warmup = False
        self.control_plane = P2pHcclAFDControlPlane(self)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def init_afd_connector(self) -> None:
        """Create one data group per stage and the separate control/ID groups."""
        if self._initialized:
            return

        # Importing torch_npu registers the HCCL backend. This connector does
        # not import or require the afd_ascend custom-op extension.
        import torch_npu  # noqa: F401

        num_stages = max(1, int(self.vllm_config.parallel_config.num_ubatches))
        timeout = timedelta(minutes=30)
        try:
            for stage_idx in range(num_stages):
                suffix = "" if stage_idx == 0 else f"_{stage_idx}"
                data_group = init_afd_process_group(
                    backend="hccl",
                    init_method=(
                        f"tcp://{self.afd_config.host}:{self.afd_config.port}"
                    ),
                    world_size=self.ffn_size + self.attn_size,
                    rank=self.world_rank,
                    group_name=f"afd_hccl_p2p{suffix}",
                    timeout=timeout,
                )
                self.data_pg_list.append(data_group)

                if self.requires_input_ids:
                    ids_group = init_afd_process_group(
                        backend="hccl",
                        init_method=(
                            f"tcp://{self.afd_config.host}:{self.afd_config.port}"
                        ),
                        world_size=self.ffn_size + self.attn_size,
                        rank=self.world_rank,
                        group_name=f"afd_hccl_p2p_ids{suffix}",
                        timeout=timeout,
                    )
                    self.ids_pg_list.append(ids_group)
                    self.input_ids_buffers.append(
                        torch.empty(
                            self.max_num_batched_tokens,
                            dtype=torch.int32,
                            device=f"npu:{self.local_rank}",
                        ),
                    )

            self.p2p_pg = init_afd_process_group(
                backend="gloo",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.ffn_size + self.attn_size,
                rank=self.p2p_rank,
                group_name="afd_hccl_p2p_control",
                timeout=timeout,
            )
        except BaseException:
            self.close()
            raise
        self._initialized = True

    def close(self) -> None:
        """Destroy every process group and discard connector-owned buffers."""
        groups = [self.p2p_pg, *self.ids_pg_list, *self.data_pg_list]
        destroyed: set[int] = set()
        for group in groups:
            if group is None or id(group) in destroyed:
                continue
            destroyed.add(id(group))
            dist.destroy_process_group(group)

        self.p2p_pg = None
        self.ids_pg_list = []
        self.data_pg_list = []
        self.input_ids_buffers = []
        self.hidden_recv_buffers = {}
        self.dp_metadata_list = {}
        self._initialized = False

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        self._require_initialized()
        metadata = context.metadata
        if not torch.compiler.is_compiling() and not metadata.validate_tensor_shape(
            tuple(hidden_states.shape),
        ):
            raise ValueError(
                f"hidden_states shape {hidden_states.shape!r} does not match "
                f"HCCL P2P metadata token count {metadata.total_tokens}",
            )

        if self.requires_input_ids:
            input_ids: torch.Tensor | None = kwargs.get("input_ids")
            if metadata.layer_idx == 0:
                if input_ids is None:
                    raise RuntimeError("DSV4 HCCL P2P layer 0 requires input_ids")
                pretransferred = bool(
                    getattr(
                        get_forward_context(),
                        "afd_input_ids_pretransferred",
                        False,
                    ),
                )
                if not pretransferred:
                    self.send_input_ids(input_ids, ubatch_idx=metadata.stage_idx)
                    maybe_apply_dbo_yield(input_ids, role="attention")
            elif input_ids is not None:
                raise RuntimeError("DSV4 HCCL P2P sends input_ids only at layer 0")

        group = self._data_group(metadata.stage_idx)
        self._send_tensor(hidden_states, dst=self.role_rank, group=group)

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._require_initialized()
        group = self._data_group(ubatch_idx)
        dist.recv(ref_tensor, src=self.role_rank, group=group)
        # FFN processes layers in layer-major order (stage 0, then stage 1),
        # while each Attention DBO thread would otherwise continue directly
        # to the next layer. Yield after the blocking receive so the peer
        # stage can send the current layer before this stage sends layer + 1.
        maybe_apply_dbo_yield(ref_tensor, role="attention")
        return ref_tensor

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        self._require_initialized()
        layer_idx = int(kwargs.get("layer_idx", 0))
        num_tokens = self._num_tokens_for_stage(
            ubatch_idx,
            fallback=int(kwargs.get("max_num_tokens", 1)),
        )
        input_ids = None
        if self.requires_input_ids and layer_idx == 0:
            input_ids = kwargs.get("input_ids")
            if input_ids is None:
                input_ids = self.recv_input_ids(num_tokens, ubatch_idx=ubatch_idx)

        hidden_states = self._hidden_recv_buffer(ubatch_idx, num_tokens)
        source_rank = self.ffn_size + self.role_rank
        dist.recv(
            hidden_states,
            src=source_rank,
            group=self._data_group(ubatch_idx),
        )
        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=ubatch_idx,
            seq_lens=[num_tokens],
        )
        return AFDA2FTransferPayload(
            hidden_states=hidden_states,
            context=AFDTransferContext(
                metadata=metadata,
                states=HCCLP2PTransferState(
                    stage_idx=ubatch_idx,
                    num_tokens=num_tokens,
                ),
            ),
            input_ids=input_ids,
        )

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        self._require_initialized()
        metadata = context.metadata
        if not torch.compiler.is_compiling() and not metadata.validate_tensor_shape(
            tuple(ffn_output.shape),
        ):
            raise ValueError(
                f"ffn_output shape {ffn_output.shape!r} does not match HCCL P2P "
                f"metadata token count {metadata.total_tokens}",
            )
        stage_idx = int(kwargs.get("ubatch_idx", metadata.stage_idx))
        destination_rank = self.ffn_size + self.role_rank
        self._send_tensor(
            ffn_output,
            dst=destination_rank,
            group=self._data_group(stage_idx),
        )

    def send_input_ids(
        self,
        input_ids: torch.Tensor,
        *,
        ubatch_idx: int,
    ) -> None:
        buffer, group = self._ids_buffer_and_group(ubatch_idx)
        flat_ids = input_ids.reshape(-1)
        num_tokens = int(flat_ids.numel())
        self._validate_input_ids(flat_ids, num_tokens)
        buffer[:num_tokens].copy_(flat_ids, non_blocking=False)
        dist.send(buffer[:num_tokens], dst=self.role_rank, group=group)

    def recv_input_ids(
        self,
        num_tokens: int,
        *,
        ubatch_idx: int,
    ) -> torch.Tensor:
        if num_tokens <= 0 or num_tokens > self.max_num_batched_tokens:
            raise ValueError(
                "DSV4 input_ids token count must be in "
                f"[1, {self.max_num_batched_tokens}], got {num_tokens}",
            )
        buffer, group = self._ids_buffer_and_group(ubatch_idx)
        input_ids = buffer[:num_tokens]
        source_rank = self.ffn_size + self.role_rank
        dist.recv(input_ids, src=source_rank, group=group)
        return input_ids

    def prepare_stage_buffer(self, stage_idx: int, num_tokens: int) -> None:
        """Ensure the FFN receive buffer is allocated before posting recv."""
        if self.afd_config.role != "ffn":
            return
        self._hidden_recv_buffer(stage_idx, num_tokens)

    def _send_tensor(
        self,
        tensor: torch.Tensor,
        *,
        dst: int,
        group: ProcessGroup,
    ) -> None:
        send_tensor = tensor if tensor.is_contiguous() else tensor.contiguous()
        dist.send(send_tensor, dst=dst, group=group)

    def _hidden_recv_buffer(
        self,
        stage_idx: int,
        num_tokens: int,
    ) -> torch.Tensor:
        if num_tokens <= 0:
            raise ValueError(f"HCCL P2P token count must be positive, got {num_tokens}")
        buffer = self.hidden_recv_buffers.get(stage_idx)
        if (
            buffer is None
            or int(buffer.shape[0]) < num_tokens
            or int(buffer.shape[1]) != self.hidden_size
            or buffer.dtype != self.dtype
        ):
            buffer = torch.empty(
                (num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=f"npu:{self.local_rank}",
            )
            self.hidden_recv_buffers[stage_idx] = buffer
        return buffer[:num_tokens]

    def _num_tokens_for_stage(self, stage_idx: int, *, fallback: int) -> int:
        dp_metadata = self.dp_metadata_list.get(stage_idx)
        if dp_metadata is None:
            return max(1, fallback)
        return _num_tokens_for_attention_rank(
            dp_metadata,
            attention_rank=self.role_rank,
            attention_size=self.attn_size,
            fallback=fallback,
        )

    def _validate_input_ids(
        self,
        flat_ids: torch.Tensor,
        num_tokens: int,
    ) -> None:
        if num_tokens <= 0 or num_tokens > self.max_num_batched_tokens:
            raise ValueError(
                "DSV4 input_ids token count must be in "
                f"[1, {self.max_num_batched_tokens}], got {num_tokens}",
            )
        if torch.compiler.is_compiling():
            return
        min_id = int(flat_ids.min().item())
        max_id = int(flat_ids.max().item())
        if min_id < -1 or max_id >= self.vocab_size:
            raise ValueError(
                "DSV4 input_ids must contain -1 padding or IDs in "
                f"[0, {self.vocab_size}); got [{min_id}, {max_id}]",
            )

    def _data_group(self, stage_idx: int) -> ProcessGroup:
        if stage_idx < 0 or stage_idx >= len(self.data_pg_list):
            raise RuntimeError(
                f"HCCL P2P data stage {stage_idx} is not initialized",
            )
        return self.data_pg_list[stage_idx]

    def _ids_buffer_and_group(
        self,
        stage_idx: int,
    ) -> tuple[torch.Tensor, ProcessGroup]:
        if stage_idx < 0 or stage_idx >= len(self.ids_pg_list):
            raise RuntimeError(
                f"HCCL P2P input IDs stage {stage_idx} is not initialized",
            )
        return self.input_ids_buffers[stage_idx], self.ids_pg_list[stage_idx]

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("HCCL P2P connector is not initialized")


class P2pHcclAFDControlPlane(AFDControlPlane):
    """Stage metadata transport for ``P2pHcclAFDConnector``."""

    def __init__(self, connector: P2pHcclAFDConnector) -> None:
        self.connector = connector

    def update_state_from_dp_metadata(
        self,
        payload: AFDControlPayload,
    ) -> None:
        connector = self.connector
        connector.dp_metadata_list = payload.dp_metadata_list
        connector.is_graph_capturing = payload.is_graph_capturing
        connector.is_warmup = payload.is_warmup
        if connector.afd_config.role != "ffn":
            return
        for stage_idx, dp_metadata in payload.dp_metadata_list.items():
            num_tokens = _num_tokens_for_attention_rank(
                dp_metadata,
                attention_rank=connector.role_rank,
                attention_size=connector.attn_size,
            )
            connector.prepare_stage_buffer(int(stage_idx), num_tokens)

    def send_dp_metadata_list(
        self,
        payload: AFDControlPayload,
    ) -> None:
        connector = self.connector
        if connector.p2p_pg is None:
            return
        if connector.afd_config.role != "attention":
            return
        send_control_payload(
            payload,
            dst=connector.role_rank,
            group=connector.p2p_pg,
            device=torch.device("cpu"),
        )

    def recv_dp_metadata_list(self) -> AFDControlPayload:
        connector = self.connector
        if connector.p2p_pg is None:
            raise RuntimeError(
                "HCCL P2P DP metadata process group is not initialized",
            )
        source_rank = connector.ffn_size + connector.role_rank
        return recv_control_payload(
            src=source_rank,
            group=connector.p2p_pg,
            device=torch.device("cpu"),
        )


def _num_tokens_for_attention_rank(
    dp_metadata: DPMetadata | AFDDPMetadata,
    *,
    attention_rank: int,
    attention_size: int,
    fallback: int = 1,
) -> int:
    counts = dp_metadata.num_tokens_across_dp_cpu.flatten().tolist()
    if not counts:
        return max(1, fallback)
    if len(counts) < attention_size and attention_size % len(counts) == 0:
        tp_size = attention_size // len(counts)
        counts = [counts[index // tp_size] for index in range(attention_size)]
    if 0 <= attention_rank < len(counts):
        return max(1, int(counts[attention_rank]))
    return max(1, fallback)


__all__ = [
    "HCCLP2PTransferState",
    "P2pHcclAFDConnector",
    "P2pHcclAFDControlPlane",
]
