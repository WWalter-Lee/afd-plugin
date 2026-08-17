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


def _vllm_config(
    *,
    num_ubatches: int = 1,
    dsv4: bool = True,
    max_num_batched_tokens: int = 16,
):
    return SimpleNamespace(
        additional_config={"afd": {"connector_extra_config": {}}},
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            num_ubatches=num_ubatches,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
        ),
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


def _connector(
    *,
    role: str,
    role_rank: int = 0,
    attention: int = 1,
    ffn: int = 1,
    num_ubatches: int = 1,
    max_num_batched_tokens: int = 16,
):
    connector = P2pHcclAFDConnector(
        0,
        0,
        _vllm_config(
            num_ubatches=num_ubatches,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        _afd_config(role=role, attention=attention, ffn=ffn),
        role_rank,
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


@pytest.mark.parametrize(
    ("role", "role_rank", "expected_subgroup", "expected_peers"),
    [
        ("attention", 0, 0, (0, 2, 3)),
        ("attention", 3, 1, (1, 4, 5)),
        ("ffn", 0, 0, (0, 2, 3)),
        ("ffn", 1, 1, (1, 4, 5)),
    ],
)
def test_p2p_hccl_accepts_integer_multiple_topology(
    role,
    role_rank,
    expected_subgroup,
    expected_peers,
):
    connector = P2pHcclAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role=role, attention=4, ffn=2),
        role_rank,
    )

    assert connector.ratio == 2
    assert connector.mapping.subgroup_index == expected_subgroup
    assert connector.mapping.subgroup_ranks == expected_peers


@pytest.mark.parametrize(
    ("attention", "ffn"),
    [(1, 2), (3, 2), (0, 1), (1, 0), (-1, 1), (1, -1)],
)
def test_p2p_hccl_rejects_invalid_unequal_topology(attention, ffn):
    with pytest.raises(ValueError, match="P2P AFD connectors require"):
        P2pHcclAFDConnector(
            0,
            0,
            _vllm_config(),
            _afd_config(role="attention", attention=attention, ffn=ffn),
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


def test_p2p_hccl_attention_uses_mapped_ffn_rank(monkeypatch):
    connector = _connector(
        role="attention",
        role_rank=3,
        attention=4,
        ffn=2,
    )
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(afd_input_ids_pretransferred=True),
    )
    events = []
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda _tensor, *, dst, group: events.append(("send", dst, group)),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "recv",
        lambda _tensor, *, src, group: events.append(("recv", src, group)),
    )
    monkeypatch.setattr(hccl_module, "maybe_apply_dbo_yield", lambda *_a, **_k: None)

    connector.send_attn_output(
        torch.ones((2, 4), dtype=torch.bfloat16),
        _attention_context(layer_idx=1, stage_idx=0, num_tokens=2),
    )
    connector.recv_ffn_output(torch.empty((2, 4), dtype=torch.bfloat16))

    assert events == [
        ("send", 1, connector.data_pg_list[0]),
        ("recv", 1, connector.data_pg_list[0]),
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
    assert payload.context.states.peer_ranks == (1,)
    assert payload.context.states.seq_lens == (3,)
    assert payload.context.states.peer_slices == ((1, 0, 3),)


def test_p2p_hccl_ffn_aggregates_unequal_peer_payloads_and_splits_output(
    monkeypatch,
):
    connector = _connector(
        role="ffn",
        role_rank=1,
        attention=4,
        ffn=2,
        max_num_batched_tokens=12,
    )
    connector.dp_metadata_list = {
        0: AFDDPMetadata(torch.tensor([2, 3, 4, 5], dtype=torch.int32)),
    }
    connector.hidden_recv_buffers[0] = torch.empty((9, 4), dtype=torch.bfloat16)
    events = []

    def recv(tensor, *, src, group):
        kind = "ids" if tensor.dtype == torch.int32 else "hidden"
        events.append((kind, src, tuple(tensor.shape), group))
        tensor.fill_(src)

    sent = []
    monkeypatch.setattr(hccl_module.dist, "recv", recv)
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda tensor, *, dst, group: sent.append(
            (tensor.clone(), dst, group),
        ),
    )

    payload = connector.recv_attn_output(ubatch_idx=0, layer_idx=0)

    assert events == [
        ("ids", 4, (4,), connector.ids_pg_list[0]),
        ("ids", 5, (5,), connector.ids_pg_list[0]),
        ("hidden", 4, (4, 4), connector.data_pg_list[0]),
        ("hidden", 5, (5, 4), connector.data_pg_list[0]),
    ]
    assert payload.context.metadata.seq_lens == [4, 5]
    assert payload.input_ids.tolist() == [4] * 4 + [5] * 5
    assert payload.hidden_states[:, 0].tolist() == [4] * 4 + [5] * 5
    assert payload.context.states == HCCLP2PTransferState(
        stage_idx=0,
        num_tokens=9,
        peer_ranks=(4, 5),
        seq_lens=(4, 5),
        peer_slices=((4, 0, 4), (5, 4, 9)),
    )

    ffn_output = torch.arange(36, dtype=torch.float32).reshape(9, 4)
    connector.send_ffn_output(ffn_output, payload.context)

    assert [(dst, tensor.shape[0]) for tensor, dst, _group in sent] == [
        (4, 4),
        (5, 5),
    ]
    assert torch.equal(sent[0][0], ffn_output[:4])
    assert torch.equal(sent[1][0], ffn_output[4:])


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
        states=HCCLP2PTransferState(
            stage_idx=1,
            num_tokens=3,
            peer_ranks=(1,),
            seq_lens=(3,),
        ),
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


def test_p2p_hccl_control_plane_prepares_aggregate_unequal_buffers(monkeypatch):
    connector = _connector(
        role="ffn",
        role_rank=1,
        attention=4,
        ffn=2,
        num_ubatches=2,
    )
    prepared = []
    monkeypatch.setattr(
        connector,
        "prepare_stage_buffer",
        lambda stage_idx, num_tokens: prepared.append((stage_idx, num_tokens)),
    )
    payload = AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(torch.tensor([2, 3, 4, 5], dtype=torch.int32)),
            1: AFDDPMetadata(torch.tensor([7, 6, 1, 2], dtype=torch.int32)),
        },
        is_graph_capturing=False,
        is_warmup=False,
    )

    connector.control_plane.update_state_from_dp_metadata(payload)

    assert prepared == [(0, 9), (1, 3)]
    assert connector.stage_layouts[0].peer_slices == ((4, 0, 4), (5, 4, 9))
    assert connector.stage_layouts[1].peer_slices == ((4, 0, 1), (5, 1, 3))


def test_p2p_hccl_reuses_control_plane_stage_layout(monkeypatch):
    connector = _connector(
        role="ffn",
        role_rank=1,
        attention=4,
        ffn=2,
    )
    prepared = []
    calls = []
    original = hccl_module._attention_token_counts
    monkeypatch.setattr(
        connector,
        "prepare_stage_buffer",
        lambda stage_idx, num_tokens: prepared.append((stage_idx, num_tokens)),
    )

    def record_counts(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(hccl_module, "_attention_token_counts", record_counts)
    payload = AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(torch.tensor([2, 3, 4, 5], dtype=torch.int32)),
        },
        is_graph_capturing=False,
        is_warmup=False,
    )

    connector.control_plane.update_state_from_dp_metadata(payload)
    first = connector._stage_layout(0, fallback=1)
    second = connector._stage_layout(0, fallback=99)

    assert first is second
    assert first.seq_lens == (4, 5)
    assert prepared == [(0, 9)]
    assert len(calls) == 1


def test_p2p_hccl_control_plane_uses_one_sender_per_subgroup(monkeypatch):
    payload = AFDControlPayload(
        dp_metadata_list={},
        is_graph_capturing=False,
        is_warmup=False,
    )
    sent = []
    monkeypatch.setattr(
        hccl_module,
        "send_control_payload",
        lambda value, *, dst, group, device: sent.append(
            (value, dst, group, device),
        ),
    )

    for role_rank in range(4):
        connector = _connector(
            role="attention",
            role_rank=role_rank,
            attention=4,
            ffn=2,
        )
        connector.p2p_pg = object()
        connector.control_plane.send_dp_metadata_list(payload)

    assert [(dst, device.type) for _value, dst, _group, device in sent] == [
        (0, "cpu"),
        (1, "cpu"),
    ]


def test_p2p_hccl_control_plane_receives_from_first_subgroup_attention(monkeypatch):
    connector = _connector(
        role="ffn",
        role_rank=1,
        attention=4,
        ffn=2,
    )
    connector.p2p_pg = object()
    expected = AFDControlPayload(
        dp_metadata_list={},
        is_graph_capturing=False,
        is_warmup=False,
    )
    calls = []

    def recv(*, src, group, device):
        calls.append((src, group, device))
        return expected

    monkeypatch.setattr(hccl_module, "recv_control_payload", recv)

    assert connector.control_plane.recv_dp_metadata_list() is expected
    assert calls == [(4, connector.p2p_pg, torch.device("cpu"))]


def test_p2p_hccl_rejects_aggregate_buffer_overflow(monkeypatch):
    connector = _connector(
        role="ffn",
        attention=2,
        ffn=1,
        max_num_batched_tokens=8,
    )
    monkeypatch.setattr(
        hccl_module.torch,
        "empty",
        lambda *_args, **_kwargs: pytest.fail("must fail before allocating"),
    )
    payload = AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(torch.tensor([4, 5], dtype=torch.int32)),
        },
        is_graph_capturing=False,
        is_warmup=False,
    )

    with pytest.raises(ValueError, match="aggregate token count.*increase FFN"):
        connector.control_plane.update_state_from_dp_metadata(payload)


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
    assert connector.stage_layouts == {}
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


def test_p2p_hccl_rejects_non_integer_input_ids_before_send():
    connector = _connector(role="attention")

    with pytest.raises(TypeError, match="int32 or int64"):
        connector._validate_input_ids(torch.ones(1, dtype=torch.float32), 1)


def test_p2p_hccl_device_input_ids_skip_host_value_readback():
    connector = _connector(role="attention")
    device_ids = torch.empty(1, dtype=torch.int64, device="meta")

    connector._validate_input_ids(device_ids, 1)


def test_p2p_hccl_module_has_no_camp2p_custom_op_reference():
    source = __import__("inspect").getsource(hccl_module)

    assert "torch.ops.vllm.afd_camp2p" not in source
    assert "torch.ops.afd_ascend" not in source
