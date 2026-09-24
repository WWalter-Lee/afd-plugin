#!/usr/bin/env python3
"""Run the shared DSV4 validator with CAMP2p selected."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SHARED_RUNNER = REPO_ROOT / "recipe/npu/deepseek_v4/common/run_validation.py"


def main() -> None:
    if any(arg == "--connector" or arg.startswith("--connector=") for arg in sys.argv):
        raise SystemExit("CAMP2p recipe fixes --connector=CAMP2pAFDConnector")
    sys.argv.extend(["--connector", "CAMP2pAFDConnector"])
    runpy.run_path(str(SHARED_RUNNER), run_name="__main__")


if __name__ == "__main__":
    main()
