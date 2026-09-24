from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from afd_plugin.diagnostics import mooncake_pd


def test_sanitize_identifier_is_file_safe_and_bounded():
    value = mooncake_pd._sanitize_identifier("req:/unsafe value" * 40)

    assert "/" not in value
    assert " " not in value
    assert len(value) <= mooncake_pd.MAX_IDENTIFIER_LENGTH


def test_select_block_ids_keeps_first_and_last(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(mooncake_pd.MAX_BLOCKS_ENV, "2")

    assert mooncake_pd._select_block_ids([10, 11, 12, 13]) == [(0, 10), (3, 13)]


def test_block_digest_is_stable_and_checks_bounds():
    torch = pytest.importorskip("torch")
    tensor = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)

    first = mooncake_pd._block_digest(tensor, 1, max_bytes=32)
    second = mooncake_pd._block_digest(tensor.clone(), 1, max_bytes=32)

    assert first == second
    assert first["sample_bytes"] == 32
    assert mooncake_pd._block_digest(tensor, 3, max_bytes=32)["error"] == (
        "block_id_out_of_range"
    )


def test_write_and_read_plan_round_trip(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(mooncake_pd.DEBUG_DIR_ENV, str(tmp_path))
    plan = {"remote_block_ids": [[1, 2], [7]]}

    mooncake_pd._write_plan("engine/0", "request:0", plan)

    assert mooncake_pd._read_plan("engine/0", "request:0") == plan


def test_registration_record_preserves_group_and_tensor_layout():
    torch = pytest.importorskip("torch")
    group = SimpleNamespace(
        kv_cache_spec=SimpleNamespace(), layer_names=["layers.0.attn"]
    )
    worker = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[group]),
        kv_role="kv_producer",
        engine_id="engine-0",
        tp_rank=0,
        num_blocks=4,
        use_hybrid=True,
        use_compress=True,
        kv_caches_base_addr=[100],
        block_len_per_addr=[64],
        block_stride_per_addr=[64],
        addr_group_idx=[[0]],
    )
    kv_caches = {"layers.0.attn": torch.zeros((2, 4, 8))}

    record = mooncake_pd._registration_record(worker, kv_caches)

    assert record["groups"][0]["layer_names"] == ["layers.0.attn"]
    assert record["kv_cache_layouts"][0]["parts"][0]["shape"] == [4, 8]
    json.dumps(record)


def test_entry_point_is_registered():
    import importlib.metadata

    entry_points = importlib.metadata.entry_points(group="vllm.general_plugins")
    matches = [entry for entry in entry_points if entry.name == "afd_pd_debug"]

    assert matches
    assert matches[0].value == (
        "afd_plugin.diagnostics.mooncake_pd:register_mooncake_pd_debug"
    )
