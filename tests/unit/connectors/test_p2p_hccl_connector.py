from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.connectors.npu import p2p_hccl as hccl_module  # noqa: E402
from afd_plugin.connectors.npu.p2p_hccl import (  # noqa: E402
    HCCLP2PTransferState,
    P2pHcclAFDConnector,
)


def _vllm_config(*, num_ubatches: int = 1, dsv4: bool = True):
    return SimpleNamespace(
        additional_config={"afd": {"connector_extra_config": {}}},
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            num_ubatches=num_ubatches,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"] if dsv4 else [],
                hidden_size=4,
                vocab_size=32,
            ),
        ),
    )


def _afd_config(*, role: str, attention: int = 1, ffn: int = 1):
    return AFDConfig(
        connector="P2pHcclAFDConnector",
        role=role,
        num_attention_ranks=attention,
        num_ffn_ranks=ffn,
    )


def _connector(*, role: str, num_ubatches: int = 1):
    connector = P2pHcclAFDConnector(
        0,
        0,
        _vllm_config(num_ubatches=num_ubatches),
        _afd_config(role=role),
        0,
    )
    connector._initialized = True
    connector.data_pg_list = [object() for _ in range(num_ubatches)]
    connector.ids_pg_list = [object() for _ in range(num_ubatches)]
    connector.input_ids_buffers = [
        torch.empty(16, dtype=torch.int32) for _ in range(num_ubatches)
    ]
    return connector


def _attention_context(*, layer_idx: int, stage_idx: int, num_tokens: int):
    return AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_len=num_tokens,
        ),
    )


def test_p2p_hccl_factory_registration():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
    )

    assert isinstance(connector, P2pHcclAFDConnector)
    assert connector.requires_input_ids is True
    assert connector.topology.role_rank == 0
    assert connector.is_initialized is False


def test_p2p_hccl_rejects_unequal_topology():
    with pytest.raises(ValueError, match="equal Attention and FFN"):
        P2pHcclAFDConnector(
            0,
            0,
            _vllm_config(),
            _afd_config(role="attention", attention=2, ffn=1),
            0,
        )


def test_p2p_hccl_attention_sends_ids_before_hidden(monkeypatch):
    connector = _connector(role="attention")
    events = []
    forward_context = SimpleNamespace(afd_input_ids_pretransferred=False)
    monkeypatch.setattr(hccl_module, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(
        hccl_module,
        "maybe_apply_dbo_yield",
        lambda tensor, **_kwargs: events.append(("yield", tensor.clone())),
    )

    def send(tensor, *, dst, group):
        kind = "ids" if tensor.dtype == torch.int32 else "hidden"
        events.append((kind, dst, group, tensor.clone()))

    monkeypatch.setattr(hccl_module.dist, "send", send)
    hidden = torch.ones((3, 4), dtype=torch.bfloat16)
    connector.send_attn_output(
        hidden,
        _attention_context(layer_idx=0, stage_idx=0, num_tokens=3),
        input_ids=torch.tensor([-1, 2, 31], dtype=torch.int64),
    )

    assert [event[0] for event in events] == ["ids", "yield", "hidden"]
    assert events[0][1:3] == (0, connector.ids_pg_list[0])
    assert events[2][1:3] == (0, connector.data_pg_list[0])
    assert events[0][3].dtype == torch.int32


def test_p2p_hccl_pretransferred_ids_send_only_hidden(monkeypatch):
    connector = _connector(role="attention")
    events = []
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(afd_input_ids_pretransferred=True),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda tensor, **_kwargs: events.append(tensor.clone()),
    )

    connector.send_attn_output(
        torch.ones((2, 4), dtype=torch.bfloat16),
        _attention_context(layer_idx=0, stage_idx=0, num_tokens=2),
        input_ids=torch.tensor([1, 2]),
    )

    assert len(events) == 1
    assert events[0].dtype == torch.bfloat16


def test_p2p_hccl_stage_groups_are_independent(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2)
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(afd_input_ids_pretransferred=True),
    )
    sent_groups = []
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda _tensor, *, dst, group: sent_groups.append((dst, group)),
    )

    for stage_idx in (0, 1):
        connector.send_attn_output(
            torch.ones((2, 4), dtype=torch.bfloat16),
            _attention_context(layer_idx=1, stage_idx=stage_idx, num_tokens=2),
        )

    assert sent_groups == [
        (0, connector.data_pg_list[0]),
        (0, connector.data_pg_list[1]),
    ]


def test_p2p_hccl_attention_yields_after_ffn_receive(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2)
    events = []
    monkeypatch.setattr(
        hccl_module.dist,
        "recv",
        lambda tensor, *, src, group: events.append(("recv", src, group)),
    )
    monkeypatch.setattr(
        hccl_module,
        "maybe_apply_dbo_yield",
        lambda tensor, **kwargs: events.append(("yield", kwargs["role"], tensor)),
    )
    output = torch.empty((2, 4), dtype=torch.bfloat16)

    assert connector.recv_ffn_output(output, ubatch_idx=1) is output
    assert events == [
        ("recv", 0, connector.data_pg_list[1]),
        ("yield", "attention", output),
    ]


def test_p2p_hccl_ffn_receives_ids_then_hidden_and_returns_state(monkeypatch):
    connector = _connector(role="ffn")
    connector.dp_metadata_list = {
        0: AFDDPMetadata(torch.tensor([3], dtype=torch.int32)),
    }
    events = []
    hidden_buffer = torch.empty((3, 4), dtype=torch.bfloat16)
    connector.hidden_recv_buffers[0] = hidden_buffer

    def recv(tensor, *, src, group):
        if tensor.dtype == torch.int32:
            events.append(("ids", src, group))
            tensor.copy_(torch.tensor([-1, 0, 31], dtype=torch.int32))
        else:
            events.append(("hidden", src, group))
            tensor.fill_(2)
        return src

    monkeypatch.setattr(hccl_module.dist, "recv", recv)
    payload = connector.recv_attn_output(ubatch_idx=0, layer_idx=0)

    assert events == [
        ("ids", 1, connector.ids_pg_list[0]),
        ("hidden", 1, connector.data_pg_list[0]),
    ]
    assert payload.input_ids.tolist() == [-1, 0, 31]
    assert torch.equal(payload.hidden_states, torch.full_like(hidden_buffer, 2))
    assert isinstance(payload.context.states, HCCLP2PTransferState)
    assert payload.context.states.num_tokens == 3


def test_p2p_hccl_ffn_send_uses_matching_stage_and_attention_rank(monkeypatch):
    connector = _connector(role="ffn", num_ubatches=2)
    sent = []
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda tensor, *, dst, group: sent.append((tensor.clone(), dst, group)),
    )
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_ffn_metadata(
            layer_idx=2,
            stage_idx=1,
            seq_lens=[3],
        ),
        states=HCCLP2PTransferState(stage_idx=1, num_tokens=3),
    )

    connector.send_ffn_output(
        torch.ones((3, 4), dtype=torch.bfloat16),
        context,
        ubatch_idx=1,
    )

    assert sent[0][1:] == (1, connector.data_pg_list[1])


def test_p2p_hccl_control_plane_prepares_stage_buffers(monkeypatch):
    connector = _connector(role="ffn", num_ubatches=2)
    prepared = []
    monkeypatch.setattr(
        connector,
        "prepare_stage_buffer",
        lambda stage_idx, num_tokens: prepared.append((stage_idx, num_tokens)),
    )
    payload = AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(torch.tensor([3], dtype=torch.int32)),
            1: AFDDPMetadata(torch.tensor([5], dtype=torch.int32)),
        },
        is_graph_capturing=False,
        is_warmup=False,
    )

    connector.control_plane.update_state_from_dp_metadata(payload)

    assert prepared == [(0, 3), (1, 5)]
    assert connector.dp_metadata_list == payload.dp_metadata_list


def test_p2p_hccl_close_destroys_all_groups(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2)
    connector.p2p_pg = object()
    groups = [
        connector.p2p_pg,
        *connector.ids_pg_list,
        *connector.data_pg_list,
    ]
    destroyed = []
    monkeypatch.setattr(
        hccl_module.dist,
        "destroy_process_group",
        lambda group: destroyed.append(group),
    )

    connector.close()

    assert destroyed == groups
    assert connector.data_pg_list == []
    assert connector.ids_pg_list == []
    assert connector.input_ids_buffers == []
    assert connector.hidden_recv_buffers == {}
    assert connector.is_initialized is False


def test_p2p_hccl_partial_init_failure_destroys_created_groups(monkeypatch):
    connector = P2pHcclAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
        0,
    )
    data_group = object()
    calls = iter((data_group, RuntimeError("IDs group failed")))

    def init_group(**_kwargs):
        result = next(calls)
        if isinstance(result, BaseException):
            raise result
        return result

    destroyed = []
    monkeypatch.setattr(hccl_module, "init_afd_process_group", init_group)
    monkeypatch.setattr(
        hccl_module.dist,
        "destroy_process_group",
        lambda group: destroyed.append(group),
    )

    with pytest.raises(RuntimeError, match="IDs group failed"):
        connector.init_afd_connector()

    assert destroyed == [data_group]
    assert connector.data_pg_list == []
    assert connector.ids_pg_list == []
    assert connector.is_initialized is False


@pytest.mark.parametrize("input_ids", [[-2], [32]])
def test_p2p_hccl_rejects_invalid_input_ids(monkeypatch, input_ids):
    connector = _connector(role="attention")
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(afd_input_ids_pretransferred=False),
    )

    with pytest.raises(ValueError, match="-1 padding"):
        connector.send_attn_output(
            torch.ones((1, 4), dtype=torch.bfloat16),
            _attention_context(layer_idx=0, stage_idx=0, num_tokens=1),
            input_ids=torch.tensor(input_ids),
        )


def test_p2p_hccl_module_has_no_camp2p_custom_op_reference():
    source = __import__("inspect").getsource(hccl_module)

    assert "torch.ops.vllm.afd_camp2p" not in source
    assert "torch.ops.afd_ascend" not in source
