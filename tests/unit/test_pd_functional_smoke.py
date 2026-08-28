from __future__ import annotations

import json
import sys

import pytest

from tools.dsv4 import run_pd_functional_smoke


def _response(batch_size: int) -> dict:
    return {
        "choices": [
            {
                "index": index,
                "prompt_token_ids": [1, index + 2],
                "token_ids": [100 + index],
                "text": "ok",
                "finish_reason": "length",
            }
            for index in range(batch_size)
        ],
        "usage": {"completion_tokens": batch_size},
    }


def test_validate_response_accepts_complete_batch():
    result = run_pd_functional_smoke._validate_response(_response(8), 8)

    assert result["batch_size"] == 8
    assert result["choice_count"] == 8
    assert result["output_token_counts"] == [1] * 8


@pytest.mark.parametrize(
    "response,message",
    [
        ({"choices": []}, "expected 1 completion choices"),
        (
            {"choices": [{"prompt_token_ids": [1], "token_ids": []}]},
            "no output token IDs",
        ),
        (
            {"choices": [{"prompt_token_ids": [], "token_ids": [2]}]},
            "no prompt token IDs",
        ),
    ],
)
def test_validate_response_rejects_incomplete_output(response, message):
    with pytest.raises(ValueError, match=message):
        run_pd_functional_smoke._validate_response(response, 1)


def test_main_records_functional_batches_without_golden(monkeypatch, tmp_path):
    output_path = tmp_path / "smoke.json"
    monkeypatch.setattr(
        run_pd_functional_smoke,
        "_request_batch",
        lambda _endpoint, _model, batch_size, _max_tokens, _timeout: _response(
            batch_size
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pd_functional_smoke.py",
            "--endpoint",
            "http://127.0.0.1/v1/completions",
            "--model",
            "dsv4-afd",
            "--batch-sizes",
            "1 8 32",
            "--output",
            str(output_path),
        ],
    )

    assert run_pd_functional_smoke.main() == 0
    report = json.loads(output_path.read_text())
    assert report["passed"] is True
    assert report["golden_checked"] is False
    assert [result["batch_size"] for result in report["results"]] == [1, 8, 32]
