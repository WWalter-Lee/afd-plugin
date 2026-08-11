from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = (
    REPO_ROOT
    / "recipe/npu/CAMP2pAFDConnector/deepseek_v4/validate_golden.py"
)


def _load_validator():
    spec = importlib.util.spec_from_file_location("dsv4_validate_golden", VALIDATOR_PATH)
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
