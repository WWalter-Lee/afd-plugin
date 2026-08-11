from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT / "recipe/npu/CAMP2pAFDConnector/deepseek_v4/run_validation.py"
)
VALIDATOR_PATH = (
    REPO_ROOT
    / "recipe/npu/CAMP2pAFDConnector/deepseek_v4/validate_golden.py"
)


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


def test_dsv4_role_scripts_keep_eager_default_and_offer_full_decode_only():
    recipe_dir = RUNNER_PATH.parent
    for role in ("attention", "ffn"):
        script = (recipe_dir / f"afd_{role}.sh").read_text(encoding="utf-8")
        assert 'EXECUTION_MODE="${EXECUTION_MODE:-eager}"' in script
        assert '"cudagraph_mode":"FULL_DECODE_ONLY"' in script
        assert '--cudagraph-capture-sizes "${CAPTURE_SIZE_ARGS[@]}"' in script

    ffn_script = (recipe_dir / "afd_ffn.sh").read_text(encoding="utf-8")
    assert "trap forward_shutdown TERM INT" in ffn_script
    assert "if ((shutdown_requested)); then" in ffn_script


def test_dsv4_runtime_manifest_records_graph_u1(monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *args, **kwargs: "head\n",
    )

    manifest = runner._runtime_manifest(
        execution_mode="full-decode-only",
        profile=True,
    )

    assert manifest["execution_mode"] == "full-decode-only"
    assert manifest["u_batches"] == 1
    assert manifest["profile"] is True
    assert manifest["torch_profiler_with_stack"] is False


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

    clean = runner._shutdown_roles(
        {"attention": FakeProcess(0), "ffn": FakeProcess(0)}
    )
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


def test_dsv4_log_gate_rejects_hidden_worker_fatal(tmp_path):
    runner = _load_runner()
    (tmp_path / "attention.log").write_text("clean shutdown\n", encoding="utf-8")
    (tmp_path / "ffn.log").write_text(
        "AFD NPU FFN worker loop failed\n",
        encoding="utf-8",
    )

    result = runner._role_log_gate(tmp_path)

    assert result["passed"] is False
    assert result["roles"]["attention"]["passed"] is True
    assert result["roles"]["ffn"]["fatal_markers"] == [
        "AFD NPU FFN worker loop failed"
    ]
