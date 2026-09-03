# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Window AFD operator boundary.

The stage-one connector only initializes communication resources.  Keeping the
operator boundary in this module makes the later A3 synchronous and A5
asynchronous implementations replaceable without changing model code.
"""

from __future__ import annotations

from typing import Any


class WindowAFDDataPathNotReady(RuntimeError):
    """Raised when a stage-one deployment attempts to run the data path."""


def _not_ready(op_name: str, **kwargs: Any) -> None:
    del kwargs
    raise WindowAFDDataPathNotReady(
        f"Window AFD operator {op_name} is not enabled in stage one",
    )


def attention_to_ffn(**kwargs: Any) -> None:
    _not_ready("npu_attention_to_ffn", **kwargs)


def ffn_worker_batching(**kwargs: Any) -> None:
    _not_ready("npu_ffn_worker_batching", **kwargs)


def ffn_to_attention(**kwargs: Any) -> None:
    _not_ready("npu_ffn_to_attention", **kwargs)


def attention_worker_combine(**kwargs: Any) -> None:
    _not_ready("npu_attention_worker_combine", **kwargs)


__all__ = [
    "WindowAFDDataPathNotReady",
    "attention_to_ffn",
    "attention_worker_combine",
    "ffn_to_attention",
    "ffn_worker_batching",
]
