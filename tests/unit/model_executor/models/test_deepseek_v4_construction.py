# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm_ascend")
nn = torch.nn

from vllm.config import CompilationMode  # noqa: E402

from afd_plugin.model_executor.models import deepseek_v4 as adapter  # noqa: E402


class _FakeStage(nn.Module):
    kind = "stage"

    def __init__(self, calls, *args, prefix="", **kwargs):
        super().__init__()
        del args, kwargs
        calls[self.kind].append(prefix)
        self.weight = nn.Parameter(torch.empty(1))


def _stage_type(kind: str):
    return type(f"Fake{kind.title()}", (_FakeStage,), {"kind": kind})


@pytest.fixture
def construction_env(monkeypatch):
    calls = {"attention": [], "moe": [], "norm": [], "embedding": []}

    attention_type = _stage_type("attention")
    moe_type = _stage_type("moe")
    norm_type = _stage_type("norm")
    embedding_type = _stage_type("embedding")

    monkeypatch.setattr(
        adapter.native,
        "DeepseekV4Attention",
        lambda *args, **kwargs: attention_type(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native,
        "DeepseekV4MoE",
        lambda *args, **kwargs: moe_type(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native,
        "RMSNorm",
        lambda *args, **kwargs: norm_type(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native,
        "VocabParallelEmbedding",
        lambda *args, **kwargs: embedding_type(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(device_type="cpu"),
    )
    monkeypatch.setattr(
        adapter.native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    return calls


@pytest.fixture
def mtp_construction_env(monkeypatch, construction_env):
    calls = construction_env
    calls.update(linear=[], shared_head=[], mtp_embedding=[])
    linear_type = _stage_type("linear")
    shared_head_type = _stage_type("shared_head")
    mtp_embedding_type = _stage_type("mtp_embedding")

    monkeypatch.setattr(
        adapter.native_mtp,
        "ReplicatedLinear",
        lambda *args, **kwargs: linear_type(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native_mtp,
        "RMSNorm",
        lambda *args, **kwargs: _stage_type("norm")(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native_mtp,
        "SharedHead",
        lambda *args, **kwargs: shared_head_type(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native_mtp,
        "VocabParallelEmbedding",
        lambda *args, **kwargs: mtp_embedding_type(calls, *args, **kwargs),
    )
    monkeypatch.setattr(
        adapter.native_mtp,
        "LogitsProcessor",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        adapter.native_mtp,
        "current_platform",
        SimpleNamespace(device_type="cpu"),
    )
    return calls


def _vllm_config(*, role: str, layer_count: int = 43, mtp: bool = False):
    config = SimpleNamespace(
        hc_eps=1e-6,
        hc_mult=4,
        hc_sinkhorn_iters=2,
        hidden_size=8,
        index_topk=4,
        n_group=1,
        num_attention_heads=8,
        num_hidden_layers=layer_count,
        num_nextn_predict_layers=1,
        rms_norm_eps=1e-6,
        rope_parameters={"original_max_position_embeddings": 64},
        routed_scaling_factor=1.5,
        vocab_size=32,
    )
    return SimpleNamespace(
        additional_config={"afd": {"role": role}},
        cache_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        model_config=SimpleNamespace(hf_config=config, dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        speculative_config=(
            SimpleNamespace(
                method="mtp",
                draft_model_config=SimpleNamespace(hf_config=config),
            )
            if mtp
            else None
        ),
    )


def _make_layer(monkeypatch, *, role: str, layer_idx: int):
    return adapter.AFDDeepseekV4DecoderLayer(
        _vllm_config(role=role),
        f"model.layers.{layer_idx}",
    )


def _parameter_names(module: nn.Module) -> set[str]:
    return {name for name, _ in module.named_parameters()}


def test_pinned_constructor_signatures_match_native_model():
    def without_annotations(function):
        signature = inspect.signature(function)
        parameters = [
            parameter.replace(annotation=inspect.Parameter.empty)
            for parameter in signature.parameters.values()
        ]
        return signature.replace(
            parameters=parameters,
            return_annotation=inspect.Signature.empty,
        )

    afd_layer_signature = without_annotations(
        adapter.AFDDeepseekV4DecoderLayer.__init__
    )
    native_layer_signature = without_annotations(
        adapter.native.DeepseekV2DecoderLayer.__init__
    )
    if "attn_cls" not in native_layer_signature.parameters:
        native_layer_signature = native_layer_signature.replace(
            parameters=[
                *native_layer_signature.parameters.values(),
                afd_layer_signature.parameters["attn_cls"],
            ]
        )
    assert afd_layer_signature == native_layer_signature
    assert without_annotations(
        adapter.AFDDeepseekV4Model.__init__
    ) == without_annotations(adapter.native.DeepseekV4Model.__init__)
    assert without_annotations(
        adapter.AFDDeepSeekMultiTokenPredictorLayer.__init__
    ) == without_annotations(
        adapter.native_mtp.DeepSeekMultiTokenPredictorLayer.__init__
    )
    assert without_annotations(
        adapter.AFDDeepSeekMultiTokenPredictor.__init__
    ) == without_annotations(adapter.native_mtp.DeepSeekMultiTokenPredictor.__init__)
    assert without_annotations(
        adapter.AFDDeepSeekV4MTP.__init__
    ) == without_annotations(adapter.native_mtp.DeepSeekV4MTP.__init__)


@pytest.mark.parametrize("layer_idx", [0, 2, 3, 42])
def test_attention_layer_owns_attention_hc_and_norm_only(
    monkeypatch,
    construction_env,
    layer_idx,
):
    layer = _make_layer(monkeypatch, role="attention", layer_idx=layer_idx)
    names = _parameter_names(layer)

    assert construction_env["attention"] == [f"model.layers.{layer_idx}.self_attn"]
    assert construction_env["moe"] == []
    assert isinstance(layer.mlp, adapter.AFDDeepseekV4RemoteMoEProxy)
    assert "self_attn.weight" in names
    assert "input_layernorm.weight" in names
    assert "post_attention_layernorm.weight" in names
    assert "hc_attn_fn" in names
    assert "hc_ffn_fn" in names
    assert not any(name.startswith("mlp.") for name in names)


@pytest.mark.parametrize("layer_idx", [0, 2, 3, 42])
def test_ffn_layer_owns_moe_only(
    monkeypatch,
    construction_env,
    layer_idx,
):
    layer = _make_layer(monkeypatch, role="ffn", layer_idx=layer_idx)
    names = _parameter_names(layer)

    assert construction_env["attention"] == []
    assert construction_env["moe"] == [f"model.layers.{layer_idx}.mlp"]
    assert isinstance(layer.self_attn, adapter.native.PPMissingLayer)
    assert names == {"mlp.weight"}


def _patch_make_layers(monkeypatch):
    def make_layers(count, factory, *, prefix):
        layers = nn.ModuleList(
            factory(prefix=f"{prefix}.{idx}") for idx in range(count)
        )
        return 0, count, layers

    monkeypatch.setattr(adapter.native, "make_layers", make_layers)


@pytest.mark.parametrize(
    ("role", "expected_embedding", "expected_topk_buffer"),
    [("attention", True, True), ("ffn", False, False)],
)
def test_model_constructor_enforces_role_ownership(
    monkeypatch,
    construction_env,
    role,
    expected_embedding,
    expected_topk_buffer,
):
    _patch_make_layers(monkeypatch)
    model = adapter.AFDDeepseekV4Model(
        vllm_config=_vllm_config(role=role, layer_count=2),
        prefix="model",
    )
    names = _parameter_names(model)

    assert ("embed_tokens.weight" in names) is expected_embedding
    assert (model.topk_indices_buffer is not None) is expected_topk_buffer
    assert model.aux_hidden_state_layers == ()
    assert ("hc_head_fn" in names) is (role == "attention")
    if role == "attention":
        assert not any(".mlp.weight" in name for name in names)
    else:
        assert set(names) == {"layers.0.mlp.weight", "layers.1.mlp.weight"}


def test_attention_target_allocates_mtp_hidden_buffer_only_when_enabled(
    monkeypatch,
    construction_env,
):
    _patch_make_layers(monkeypatch)
    model = adapter.AFDDeepseekV4Model(
        vllm_config=_vllm_config(role="attention", layer_count=1, mtp=True),
        prefix="model",
    )

    assert model._mtp_hidden_buffer.shape == (8, 32)
    assert model._mtp_hidden_buffer.dtype == torch.bfloat16


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_mtp_constructor_enforces_role_ownership(
    mtp_construction_env,
    role,
):
    model = adapter.AFDDeepSeekV4MTP(
        vllm_config=_vllm_config(role=role, layer_count=1, mtp=True),
    )
    names = _parameter_names(model)

    assert model.model.start_layer == 0
    assert model.model.end_layer == 1
    if role == "attention":
        assert mtp_construction_env["attention"] == ["mtp.0.self_attn"]
        assert mtp_construction_env["moe"] == []
        assert "model.layers.0.e_proj.weight" in names
        assert "model.layers.0.h_proj.weight" in names
        assert "model.layers.0.shared_head.weight" in names
        assert "model.embed_tokens.weight" in names
        assert not any("mtp_block.mlp" in name for name in names)
    else:
        assert mtp_construction_env["attention"] == []
        assert mtp_construction_env["moe"] == ["mtp.0.mlp"]
        assert names == {"model.layers.0.mtp_block.mlp.weight"}


def test_afd_activation_is_required_before_construction():
    config = _vllm_config(role="attention")
    config.additional_config = {}

    with pytest.raises(ValueError, match="requires additional_config"):
        adapter.AFDDeepseekV4DecoderLayer(config, "model.layers.0")
    with pytest.raises(ValueError, match="requires additional_config"):
        adapter.AFDDeepseekV4Model(vllm_config=config, prefix="model")
