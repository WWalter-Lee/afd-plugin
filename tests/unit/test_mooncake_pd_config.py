from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.dsv4.mooncake_pd_config import build_mooncake_pd_config

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_TOOL = ROOT_DIR / "tools/dsv4/mooncake_pd_config.py"
MANUAL_PD_DIR = ROOT_DIR / "tools/dsv4/mooncake_pd_manual"


def test_build_mooncake_pd_config_is_role_and_topology_explicit():
    config = build_mooncake_pd_config(
        role="kv_consumer",
        engine_id="decode-0",
        kv_port=30100,
        prefill_dp_size=2,
        prefill_tp_size=4,
        decode_dp_size=8,
        decode_tp_size=1,
    )

    assert config == {
        "kv_connector": "MooncakeHybridConnector",
        "kv_role": "kv_consumer",
        "kv_port": 30100,
        "engine_id": "decode-0",
        "kv_parallel_size": 1,
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 2, "tp_size": 4},
            "decode": {"dp_size": 8, "tp_size": 1},
        },
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"role": "kv_both"}, "role must be"),
        ({"engine_id": ""}, "engine_id"),
        ({"kv_port": 0}, "kv_port"),
        ({"decode_tp_size": 0}, "decode_tp_size"),
    ],
)
def test_build_mooncake_pd_config_rejects_invalid_values(overrides, message):
    values = {
        "role": "kv_producer",
        "engine_id": "prefill-0",
        "kv_port": 30000,
        "prefill_dp_size": 2,
        "prefill_tp_size": 4,
        "decode_dp_size": 8,
        "decode_tp_size": 1,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        build_mooncake_pd_config(**values)


def test_mooncake_pd_config_cli_outputs_compact_json():
    output = subprocess.check_output(
        [
            sys.executable,
            str(CONFIG_TOOL),
            "--role",
            "kv_consumer",
            "--engine-id",
            "decode-0",
            "--kv-port",
            "30100",
            "--prefill-dp-size",
            "2",
            "--prefill-tp-size",
            "4",
            "--decode-dp-size",
            "8",
            "--decode-tp-size",
            "1",
        ],
        text=True,
    )

    assert " " not in output.strip()
    assert json.loads(output)["kv_role"] == "kv_consumer"


def test_mooncake_pd_recipe_keeps_kv_transfer_off_ffn():
    hccl_recipe_dir = ROOT_DIR / "recipe/npu/P2pHcclAFDConnector/deepseek_v4"
    attention = (hccl_recipe_dir / "afd_attention.sh").read_text()
    ffn = (hccl_recipe_dir / "afd_ffn.sh").read_text()
    prefill = (hccl_recipe_dir / "mooncake_pd/prefill.sh").read_text()

    assert "--role kv_consumer" in attention
    assert "--kv-transfer-config" in attention
    assert 'source "${ROOT_DIR}/tools/dsv4/check_mooncake_runtime.sh"' in attention
    assert 'export VLLM_HOST_IP="${VLLM_HOST_IP:-${HCCL_IF_IP}}"' in attention
    assert "--kv-transfer-config" not in ffn
    assert "--role kv_producer" in prefill
    assert 'source "${ROOT_DIR}/tools/dsv4/check_mooncake_runtime.sh"' in prefill
    assert 'export VLLM_HOST_IP="${VLLM_HOST_IP:-${HCCL_IF_IP}}"' in prefill


def test_mooncake_pd_manual_entry_is_safe_and_size_capped():
    script_path = MANUAL_PD_DIR / "pd.sh"
    config_path = MANUAL_PD_DIR / "config.env.example"
    script = script_path.read_text()
    config = config_path.read_text()

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    help_output = subprocess.check_output(
        ["bash", str(script_path), "--help"], text=True
    )

    for action in (
        "init",
        "install",
        "check",
        "start",
        "status",
        "validate",
        "stop",
        "collect",
    ):
        assert action in help_output

    assert "pkill" not in script
    assert "stop_name attention; stop_name ffn" in script
    assert "ENABLE_PD=1" in script
    assert "RUN_LOCAL_ROUNDTRIP=0\n  check_action" in script
    assert '"${MODEL_NAME}"' in script
    assert 'tail -c "${ARTIFACT_LOG_TAIL_BYTES}"' in script
    assert 'tail -n 50 >"${temp_dir}/kv-transfer-evidence.txt"' in script
    assert 'tail -n 200 >"${temp_dir}/fatal-markers.txt"' in script
    assert "recipe/npu/deepseek_v4/common/validate_golden.py" in script
    assert 'ARTIFACT_LOG_TAIL_BYTES="262144"' in config
    assert 'ARTIFACT_MAX_BYTES="2097152"' in config
    assert "http://127.0.0.1:${FFN_PROCESS_PORT}" not in script
