from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from afd_plugin.diagnostics import deepseek_v4_chunk


def test_prefill_metadata_selects_first_prefill_entry():
    decode = SimpleNamespace(num_prefills=0, prefill=None)
    prefill = SimpleNamespace(num_prefills=1, prefill=object())

    assert deepseek_v4_chunk._prefill_metadata([decode, prefill]) is prefill


def test_capture_record_selects_final_prefill_token():
    torch = pytest.importorskip("torch")
    prefill = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2]),
        input_positions=torch.tensor([3, 4]),
        seq_lens=torch.tensor([5]),
        start_pos=torch.tensor([3]),
    )
    metadata = SimpleNamespace(
        num_decodes=1,
        num_prefills=1,
        num_actual_tokens=3,
        num_decode_tokens=1,
        prefill=prefill,
    )
    instance = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_rank=0)
        ),
        compress_ratio=4,
    )
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    output = hidden + 100

    record, tensors = deepseek_v4_chunk._capture_record(
        instance, "model.layers.0.self_attn", hidden, output, [metadata]
    )

    assert record["selected_position"] == 4
    assert record["selected_token_index"] == 2
    assert record["attention_mode"] == "prefill"
    assert tensors["hidden"].tolist() == [8, 9, 10, 11]
    assert tensors["attention_output"].tolist() == [108, 109, 110, 111]


def test_capture_record_selects_short_extend_decode_token():
    torch = pytest.importorskip("torch")
    decode = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1]),
        input_positions=torch.tensor([4]),
        seq_lens=torch.tensor([5]),
        start_pos=torch.tensor([4]),
    )
    metadata = SimpleNamespace(
        num_decodes=1,
        num_prefills=0,
        num_actual_tokens=1,
        num_decode_tokens=1,
        decode=decode,
        prefill=None,
    )
    instance = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_rank=0)
        ),
        compress_ratio=4,
    )
    hidden = torch.arange(4, dtype=torch.float32).reshape(1, 4)

    record, tensors = deepseek_v4_chunk._capture_record(
        instance, "model.layers.0.self_attn", hidden, hidden + 100, [metadata]
    )

    assert record["selected_position"] == 4
    assert record["selected_token_index"] == 0
    assert record["attention_mode"] == "decode"
    assert record["start_pos"] == [4]
    assert tensors["hidden"].tolist() == [0, 1, 2, 3]


def test_entry_point_is_registered():
    import importlib.metadata

    entry_points = importlib.metadata.entry_points(group="vllm.general_plugins")
    matches = [entry for entry in entry_points if entry.name == "afd_dsv4_chunk_debug"]

    assert matches
    assert matches[0].value == (
        "afd_plugin.diagnostics.deepseek_v4_chunk:register_dsv4_chunk_debug"
    )


def test_short_extend_prefill_probe_forces_classification_flag():
    original_split = Mock(return_value=(0, 1, 0, 1))
    dsa_module = SimpleNamespace(split_decodes_and_prefills=original_split)

    deepseek_v4_chunk._install_short_extend_prefill_probe(dsa_module)
    result = dsa_module.split_decodes_and_prefills(
        object(),
        decode_threshold=2,
        require_uniform=True,
        treat_short_extends_as_decodes=True,
    )

    assert result == (0, 1, 0, 1)
    original_split.assert_called_once_with(
        original_split.call_args.args[0],
        decode_threshold=2,
        require_uniform=True,
        treat_short_extends_as_decodes=False,
    )
