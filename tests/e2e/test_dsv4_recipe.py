from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "recipe/npu/CAMP2pAFDConnector/deepseek_v4/run_validation.py"
VALIDATOR_PATH = (
    REPO_ROOT / "recipe/npu/CAMP2pAFDConnector/deepseek_v4/validate_golden.py"
)
HCCL_RECIPE_DIR = REPO_ROOT / "recipe/npu/P2pHcclAFDConnector/deepseek_v4"


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "dsv4_validate_golden",
        VALIDATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runner():
    spec = importlib.util.spec_from_file_location("dsv4_run_validation", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_gate_checks_structure_without_requiring_single_request_tokens():
    validator = _load_validator()
    expected = {"prompt_token_ids": [1, 2], "token_ids": [3, 4]}
    result = {"prompt_token_ids": [1, 2], "token_ids": [9, 10]}

    assert validator._batch_result_valid(result, expected)


def test_batch_gate_rejects_bad_prompt_ids_or_completion_shape():
    validator = _load_validator()
    expected = {"prompt_token_ids": [1, 2], "token_ids": [3, 4]}

    assert not validator._batch_result_valid(
        {"prompt_token_ids": [1], "token_ids": [9, 10]}, expected
    )
    assert not validator._batch_result_valid(
        {"prompt_token_ids": [1, 2], "token_ids": [9]}, expected
    )


def test_dsv4_role_scripts_offer_u1_graph_and_eager_u2():
    recipe_dir = RUNNER_PATH.parent
    for role in ("attention", "ffn"):
        script = (recipe_dir / f"afd_{role}.sh").read_text(encoding="utf-8")
        assert 'EXECUTION_MODE="${EXECUTION_MODE:-eager}"' in script
        assert 'U_BATCHES="${U_BATCHES:-1}"' in script
        assert '"cudagraph_mode":"FULL_DECODE_ONLY"' in script
        assert '--cudagraph-capture-sizes "${CAPTURE_SIZE_ARGS[@]}"' in script
        assert "--enable-dbo" in script
        assert '--dbo-decode-token-threshold "$DBO_DECODE_TOKEN_THRESHOLD"' in script
        assert '--dbo-prefill-token-threshold "$DBO_PREFILL_TOKEN_THRESHOLD"' in script
        assert '"${UBATCH_ARGS[@]}"' in script

    ffn_script = (recipe_dir / "afd_ffn.sh").read_text(encoding="utf-8")
    assert "trap forward_shutdown TERM INT" in ffn_script
    assert "if ((shutdown_requested)); then" in ffn_script


def test_dsv4_hccl_recipe_selects_hccl_connector_without_copying_validator():
    for role in ("attention", "ffn"):
        script = (HCCL_RECIPE_DIR / f"afd_{role}.sh").read_text(encoding="utf-8")
        assert "export AFD_CONNECTOR=P2pHcclAFDConnector" in script
        assert "CAMP2pAFDConnector/deepseek_v4" in script

    runner = (HCCL_RECIPE_DIR / "run_validation.py").read_text(encoding="utf-8")
    assert 'sys.argv.extend(["--connector", "P2pHcclAFDConnector"])' in runner
    assert "runpy.run_path" in runner


def test_dsv4_runtime_manifest_records_graph_u1(monkeypatch):
    runner = _load_runner()

    def check_output(command, **kwargs):
        if "status" in command:
            return " M tracked.py\n"
        if "diff" in command:
            return b"diff data"
        return "head\n"

    monkeypatch.setattr(runner.subprocess, "check_output", check_output)

    manifest = runner._runtime_manifest(
        connector="CAMP2pAFDConnector",
        execution_mode="full-decode-only",
        u_batches=1,
        dbo_decode_token_threshold=2,
        dbo_prefill_token_threshold=12,
        profile=True,
    )

    assert manifest["execution_mode"] == "full-decode-only"
    assert manifest["u_batches"] == 1
    assert manifest["profile"] is True
    assert manifest["profile_role_ranks"] == [0]
    assert manifest["torch_profiler_with_stack"] is False
    assert manifest["afd_plugin_worktree"]["tracked_dirty"] is True
    assert manifest["afd_plugin_worktree"]["tracked_status"] == [" M tracked.py"]
    assert len(manifest["afd_plugin_worktree"]["tracked_diff_sha256"]) == 64


def test_dsv4_runtime_manifest_records_eager_u2(monkeypatch):
    runner = _load_runner()

    def check_output(command, **kwargs):
        if "status" in command:
            return ""
        if "diff" in command:
            return b""
        return "head\n"

    monkeypatch.setattr(runner.subprocess, "check_output", check_output)

    manifest = runner._runtime_manifest(
        connector="P2pHcclAFDConnector",
        execution_mode="eager",
        u_batches=2,
        dbo_decode_token_threshold=2,
        dbo_prefill_token_threshold=12,
        profile=False,
    )

    assert manifest["execution_mode"] == "eager"
    assert manifest["connector"] == "P2pHcclAFDConnector"
    assert manifest["u_batches"] == 2
    assert manifest["dbo_decode_token_threshold"] == 2
    assert manifest["dbo_prefill_token_threshold"] == 12
    assert manifest["afd_plugin_worktree"]["tracked_dirty"] is False


def test_dsv4_shutdown_gate_requires_both_roles_to_exit_cleanly(monkeypatch):
    runner = _load_runner()
    stop_calls = []

    class FakeProcess:
        def __init__(self, returncode):
            self.returncode = returncode
            self.pid = 1

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        runner,
        "_stop_process",
        lambda process, **kwargs: stop_calls.append((process, kwargs)),
    )

    clean = runner._shutdown_roles({"attention": FakeProcess(0), "ffn": FakeProcess(0)})
    failed = runner._shutdown_roles(
        {"attention": FakeProcess(0), "ffn": FakeProcess(1)}
    )

    assert clean["passed"] is True
    assert failed["passed"] is False
    assert [kwargs for _process, kwargs in stop_calls] == [
        {},
        {"signal_group": False},
        {},
        {"signal_group": False},
    ]


@pytest.mark.parametrize(
    "fatal_marker",
    ["AFD NPU FFN worker loop failed", "Exception in thread"],
)
def test_dsv4_log_gate_rejects_hidden_worker_fatal(tmp_path, fatal_marker):
    runner = _load_runner()
    (tmp_path / "attention.log").write_text("clean shutdown\n", encoding="utf-8")
    (tmp_path / "ffn.log").write_text(
        f"{fatal_marker}\n",
        encoding="utf-8",
    )

    result = runner._role_log_gate(tmp_path)

    assert result["passed"] is False
    assert result["roles"]["attention"]["passed"] is True
    assert result["roles"]["ffn"]["fatal_markers"] == [fatal_marker]


def test_dsv4_log_gate_reports_missing_role_log(tmp_path):
    runner = _load_runner()
    (tmp_path / "ffn.log").write_text("clean shutdown\n", encoding="utf-8")

    result = runner._role_log_gate(tmp_path)

    assert result["passed"] is False
    assert result["roles"]["attention"]["fatal_markers"] == ["<log missing>"]


@pytest.mark.parametrize(
    ("u_batches", "attention_log", "expected"),
    [
        (1, "key=((0, (8,)),)\n", True),
        (2, "key=((0, (8,)),)\n", False),
        (2, "key=((0, (4,)), (1, (4,)))\n", True),
    ],
)
def test_dsv4_ubatch_gate_requires_two_stage_runtime_evidence(
    tmp_path,
    u_batches,
    attention_log,
    expected,
):
    runner = _load_runner()
    (tmp_path / "attention.log").write_text(attention_log, encoding="utf-8")

    result = runner._ubatch_execution_gate(tmp_path, u_batches)

    assert result["required"] is (u_batches == 2)
    assert result["passed"] is expected


def test_dsv4_profile_gate_requires_one_nonempty_dp0_trace_per_role(tmp_path):
    runner = _load_runner()
    for role in ("attention", "ffn"):
        trace_dir = tmp_path / role / f"{role}_dp0_ascend_pt"
        (trace_dir / "FRAMEWORK").mkdir(parents=True)
        (trace_dir / "profiler_info_0.json").write_text("{}\n", encoding="utf-8")
        (trace_dir / "FRAMEWORK/torch.op_range").write_bytes(b"torch-ops")
        raw_dir = trace_dir / "PROF_000001/device_0/data"
        raw_dir.mkdir(parents=True)
        (raw_dir / "stars.data").write_bytes(b"cann-data")

    result = runner._profile_output_gate(tmp_path)

    assert result["passed"] is True
    assert result["roles"]["attention"]["cann_raw_file_count"] == 1

    (tmp_path / "attention/extra_ascend_pt").mkdir()

    result = runner._profile_output_gate(tmp_path)

    assert result["passed"] is False
    assert result["roles"]["attention"]["passed"] is False


def test_dsv4_npu_process_parser_ignores_device_rows():
    runner = _load_runner()
    output = """\
| NPU   Name                | Health        | Power(W) |
| 0     0                   | 0000:18:00.0  | 0        |
| NPU     Chip              | Process id    | Process name |
| 0       0                 | 12345         | VLLM::EngineCore |
| 7       1                 | 67890         | python |
"""

    assert runner._npu_process_ids(output) == [12345, 67890]


def test_dsv4_npu_cleanup_gate_rejects_residual_processes(monkeypatch, tmp_path):
    runner = _load_runner()
    output = """\
| NPU     Chip              | Process id    | Process name |
| 0       0                 | 12345         | VLLM::EngineCore |
"""
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
    )

    result = runner._wait_for_npu_cleanup(tmp_path / "npu.txt", timeout=0)

    assert result["passed"] is False
    assert result["process_ids"] == [12345]


def test_dsv4_npu_cleanup_gate_accepts_clean_npus(monkeypatch, tmp_path):
    runner = _load_runner()
    output = """\
| NPU     Chip              | Process id    | Process name |
| No running processes found in NPU 0 |
"""
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
    )

    result = runner._wait_for_npu_cleanup(tmp_path / "npu.txt", timeout=0)

    assert result["passed"] is True
    assert result["process_ids"] == []


def test_dsv4_npu_cleanup_gate_rejects_truncated_output(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            "npu-smi version only\n",
            "",
        ),
    )

    result = runner._wait_for_npu_cleanup(tmp_path / "npu.txt", timeout=0)

    assert result["passed"] is False
    assert result["process_table_present"] is False
