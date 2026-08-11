#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any

RECIPE_DIR = Path(__file__).resolve().parent
REPO_ROOT = RECIPE_DIR.parents[3]
FATAL_LOG_MARKERS = (
    "AFD NPU FFN worker loop failed",
    "EngineCore encountered a fatal error",
    "RuntimeError: Worker failed with error",
)


def _port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        # Match the API server's bind behavior so a prior cycle's TIME_WAIT
        # sockets are not mistaken for an active listener.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _start_role(
    role: str,
    *,
    output_dir: Path,
    api_port: int,
    afd_port: int,
    execution_mode: str,
    profile_dir: Path | None,
) -> tuple[subprocess.Popen[bytes], Any]:
    log_handle = (output_dir / f"{role}.log").open("wb")
    env = os.environ.copy()
    env.update(
        {
            "API_PORT": str(api_port),
            "AFD_PORT": str(afd_port),
            "AFD_HOST": "127.0.0.1",
            "HCCL_IF_IP": env.get("HCCL_IF_IP", "192.169.91.106"),
            "PYTHONUNBUFFERED": "1",
            "EXECUTION_MODE": execution_mode,
        }
    )
    if profile_dir is not None:
        role_prefix = f"AFD_NPU_{role.upper()}_PROFILER"
        role_dir = profile_dir / role
        role_dir.mkdir(parents=True, exist_ok=True)
        env.update(
            {
                f"{role_prefix}_ENABLE": "1",
                f"{role_prefix}_WAIT": "2",
                f"{role_prefix}_WARMUP": "1",
                f"{role_prefix}_ACTIVE": "10",
                f"{role_prefix}_REPEAT": "1",
                f"{role_prefix}_SKIP_FIRST": "0",
                f"{role_prefix}_DIR": str(role_dir),
                f"{role_prefix}_WITH_STACK": "0",
                "TORCH_PROFILER_WITH_STACK": "0",
            }
        )
    script = RECIPE_DIR / f"afd_{role}.sh"
    process = subprocess.Popen(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_handle


def _log_tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return "<log missing>"
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def _wait_for_api(
    endpoint: str,
    processes: dict[str, subprocess.Popen[bytes]],
    log_dir: Path,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        for role, process in processes.items():
            if process.poll() is not None:
                tail = _log_tail(log_dir / f"{role}.log")
                raise RuntimeError(
                    f"{role} exited during startup with {process.returncode}\n{tail}"
                )
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(2)
    raise TimeoutError(f"API did not become ready: {last_error!r}")


def _run_validator(
    *,
    api_port: int,
    golden: Path,
    output: Path,
    rounds: int,
    batch_sizes: list[int],
    prompt_indices: list[int] | None,
) -> None:
    command = [
        sys.executable,
        str(RECIPE_DIR / "validate_golden.py"),
        "--endpoint",
        f"http://127.0.0.1:{api_port}/v1/completions",
        "--model",
        "dsv4-afd",
        "--golden",
        str(golden),
        "--output",
        str(output),
        "--rounds",
        str(rounds),
        "--batch-sizes",
        *(str(size) for size in batch_sizes),
    ]
    if prompt_indices is not None:
        command.extend(
            ["--prompt-indices", *(str(index) for index in prompt_indices)]
        )
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _signal_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    if process.poll() is None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, sig)


def _signal_process(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    if process.poll() is None:
        with suppress(ProcessLookupError):
            os.kill(process.pid, sig)


def _stop_process(
    process: subprocess.Popen[bytes],
    timeout: float = 30,
    *,
    signal_group: bool = True,
) -> None:
    if signal_group:
        _signal_group(process, signal.SIGTERM)
    else:
        _signal_process(process, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        process.wait(timeout=30)


def _shutdown_roles(processes: dict[str, subprocess.Popen[bytes]]) -> dict[str, Any]:
    result: dict[str, Any] = {"order": ["attention", "ffn"]}
    attention = processes.get("attention")
    ffn = processes.get("ffn")
    if attention is not None:
        _stop_process(attention)
        result["attention_returncode"] = attention.returncode
    if ffn is not None:
        ffn_exited_after_attention = ffn.poll() is not None
        if not ffn_exited_after_attention:
            time.sleep(2)
            ffn_exited_after_attention = ffn.poll() is not None
        result["ffn_exited_after_attention"] = ffn_exited_after_attention
        # FFN uses a supervising shell. Signal only that shell first so its
        # trap can ask the vLLM parent to shut down descendants in order. The
        # timeout path in _stop_process still kills the full process group.
        _stop_process(ffn, signal_group=False)
        result["ffn_returncode"] = ffn.returncode
    result["passed"] = all(
        result.get(f"{role}_returncode") == 0
        for role in ("attention", "ffn")
        if role in processes
    )
    return result


def _role_log_gate(log_dir: Path) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role in ("attention", "ffn"):
        log_path = log_dir / f"{role}.log"
        text = log_path.read_text(encoding="utf-8", errors="replace")
        markers = [marker for marker in FATAL_LOG_MARKERS if marker in text]
        roles[role] = {"passed": not markers, "fatal_markers": markers}
    return {
        "roles": roles,
        "passed": all(result["passed"] for result in roles.values()),
    }


def _capture_command(command: list[str], output: Path) -> None:
    with output.open("wb") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)


def _runtime_manifest(*, execution_mode: str, profile: bool) -> dict[str, Any]:
    def git_head(path: str) -> str:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True
        ).strip()

    return {
        "python": sys.version,
        "plugins": "ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd",
        "cann": "/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1",
        "venv": "/mnt/workspace/code/.venvs/afd-v026",
        "model": "/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp",
        "execution_mode": execution_mode,
        "u_batches": 1,
        "profile": profile,
        "torch_profiler_with_stack": False,
        "commits": {
            "afd_plugin": git_head(str(REPO_ROOT)),
            "vllm": git_head("/mnt/workspace/code/vllm-afd-v0.26.0"),
            "vllm_ascend": git_head(
                "/mnt/workspace/code/vllm-ascend-afd-80d8c194f"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path(
            "/mnt/workspace/validation/dsv4_milestone0_20260810/"
            "golden_results.json"
        ),
    )
    parser.add_argument("--attention-port", type=int, default=8910)
    parser.add_argument("--ffn-port", type=int, default=8911)
    parser.add_argument("--afd-port", type=int, default=29761)
    parser.add_argument("--startup-timeout", type=float, default=3600)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--idle-seconds", type=int, default=1800)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 8, 32])
    parser.add_argument("--prompt-indices", type=int, nargs="*")
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "full-decode-only"),
        default="eager",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect plugin-owned Attention and FFN torch-npu traces.",
    )
    args = parser.parse_args()

    for port in (args.attention_port, args.ffn_port, args.afd_port):
        if not _port_is_free(port):
            raise RuntimeError(f"port {port} is already in use")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runtime.json").write_text(
        json.dumps(
            _runtime_manifest(
                execution_mode=args.execution_mode,
                profile=args.profile,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    cycles = []
    overall_passed = False
    try:
        for cycle_idx in range(1, args.cycles + 1):
            cycle_dir = args.output_dir / f"cycle_{cycle_idx}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            processes: dict[str, subprocess.Popen[bytes]] = {}
            handles = []
            cycle_result: dict[str, Any] = {
                "cycle": cycle_idx,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            startup_started = time.monotonic()
            profile_dir = cycle_dir / "profiles" if args.profile else None
            if profile_dir is not None:
                cycle_result["profile"] = {
                    "enabled": True,
                    "root": str(profile_dir),
                    "wait": 2,
                    "warmup": 1,
                    "active": 10,
                    "with_stack": False,
                }
            try:
                ffn_process, ffn_handle = _start_role(
                    "ffn",
                    output_dir=cycle_dir,
                    api_port=args.ffn_port,
                    afd_port=args.afd_port,
                    execution_mode=args.execution_mode,
                    profile_dir=profile_dir,
                )
                processes["ffn"] = ffn_process
                handles.append(ffn_handle)
                time.sleep(2)
                attention_process, attention_handle = _start_role(
                    "attention",
                    output_dir=cycle_dir,
                    api_port=args.attention_port,
                    afd_port=args.afd_port,
                    execution_mode=args.execution_mode,
                    profile_dir=profile_dir,
                )
                processes["attention"] = attention_process
                handles.append(attention_handle)
                _wait_for_api(
                    f"http://127.0.0.1:{args.attention_port}/v1/models",
                    processes,
                    cycle_dir,
                    args.startup_timeout,
                )
                cycle_result["startup_seconds"] = round(
                    time.monotonic() - startup_started, 3
                )
                _capture_command(["npu-smi", "info"], cycle_dir / "npu_ready.txt")
                _run_validator(
                    api_port=args.attention_port,
                    golden=args.golden,
                    output=cycle_dir / "golden.json",
                    rounds=args.rounds,
                    batch_sizes=args.batch_sizes,
                    prompt_indices=args.prompt_indices,
                )
                if cycle_idx == 1 and args.idle_seconds > 0:
                    cycle_result["idle_seconds"] = args.idle_seconds
                    time.sleep(args.idle_seconds)
                    _run_validator(
                        api_port=args.attention_port,
                        golden=args.golden,
                        output=cycle_dir / "idle_resume.json",
                        rounds=1,
                        batch_sizes=[1],
                        prompt_indices=args.prompt_indices,
                    )
                cycle_result["passed"] = True
            finally:
                cycle_result["shutdown"] = _shutdown_roles(processes)
                for handle in handles:
                    handle.close()
                cycle_result["log_gate"] = _role_log_gate(cycle_dir)
                cycle_result["passed"] = bool(
                    cycle_result.get("passed", False)
                    and cycle_result["shutdown"]["passed"]
                    and cycle_result["log_gate"]["passed"]
                )
                _capture_command(
                    ["npu-smi", "info"], cycle_dir / "npu_after_cleanup.txt"
                )
                cycle_result["finished_at"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z"
                )
                cycles.append(cycle_result)
                (cycle_dir / "cycle_summary.json").write_text(
                    json.dumps(cycle_result, indent=2, sort_keys=True) + "\n"
                )
        overall_passed = all(cycle.get("passed", False) for cycle in cycles)
    finally:
        summary = {
            "passed": overall_passed,
            "cycles": cycles,
            "golden": str(args.golden),
            "execution_mode": args.execution_mode,
            "u_batches": 1,
            "profile": args.profile,
        }
        (args.output_dir / "validation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
