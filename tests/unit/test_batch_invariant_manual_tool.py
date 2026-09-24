from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
TOOL_DIR = ROOT_DIR / "tools/dsv4/vllm_ascend_batch_invariant"


def test_batch_invariant_manual_script_keeps_delivery_contract():
    script_path = TOOL_DIR / "bi.sh"
    script = script_path.read_text()

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    help_output = subprocess.check_output(
        ["bash", str(script_path), "--help"], text=True
    )

    for action in (
        "download-opp",
        "apply-patch",
        "install",
        "check",
        "run-twice",
        "collect",
    ):
        assert action in help_output
    assert "pkill" not in script
    assert 'BASE_COMMIT="3da28f9414583d2d0b672a8f06d1fae142404bda"' in script
    assert (
        'PATCH_SHA256="cf97a0b6e509fbb128e847babbf8f01cc953f06cb3126936cc4111bbab60b897"'
        in script
    )
    assert (
        'OPP_SHA256="9fc692978e9420336e3fea03a92c2a85df1b50a65a7df50173e3bf8bedaea70e"'
        in script
    )
    assert (
        'WHEEL_SHA256="a5ae4cfbad39e47ba4233d1fa799b5d469960ff2f266eded4de3dc69ac0c0898"'
        in script
    )
    assert "Enabling batch-invariant mode" in script
    assert "backend unavailable" in script
    assert "cross-start-summary.json" in script
    assert "size <= 2097152" in script
    assert (
        "VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector"
        in script
    )
    assert "--data-parallel-size 2" in script
    assert "--tensor-parallel-size 4" in script
    assert "--rounds 3" in script
    assert "TORCH_DEVICE_BACKEND_AUTOLOAD=0" in script
    assert "--noconftest" in script


def test_batch_invariant_compare_reports_cross_start_exact(tmp_path):
    records = [
        {
            "round": round_index,
            "prompt_index": prompt_index,
            "prompt_token_ids": [prompt_index],
            "token_ids": [round_index, prompt_index],
        }
        for round_index in range(1, 4)
        for prompt_index in range(10)
    ]
    payload = {"passed": True, "records": records}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "summary.json"
    first.write_text(json.dumps(payload))
    second.write_text(json.dumps(payload))

    subprocess.run(
        [
            sys.executable,
            str(TOOL_DIR / "compare_batch_invariant_runs.py"),
            str(first),
            str(second),
            "--output",
            str(output),
        ],
        check=True,
    )

    report = json.loads(output.read_text())
    assert report["passed"] is True
    assert report["request_count"] == 30
    assert report["cross_start_exact_match_count"] == 30
