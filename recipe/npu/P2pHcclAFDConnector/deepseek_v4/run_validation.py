#!/usr/bin/env python3
"""Run the shared DSV4 validator with the HCCL P2P connector selected."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SHARED_RUNNER = (
    REPO_ROOT
    / "recipe/npu/CAMP2pAFDConnector/deepseek_v4/run_validation.py"
)


def main() -> None:
    if "--connector" not in sys.argv:
        sys.argv.extend(["--connector", "P2pHcclAFDConnector"])
    runpy.run_path(str(SHARED_RUNNER), run_name="__main__")


if __name__ == "__main__":
    main()
