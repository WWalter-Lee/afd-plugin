#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
    "Exception in thread",
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
    connector: str,
    execution_mode: str,
    u_batches: int,
    dbo_decode_token_threshold: int,
    dbo_prefill_token_threshold: int,
    profile_dir: Path | None,
) -> tuple[subprocess.Popen[bytes], Any]:
    log_handle = (output_dir / f"{role}.log").open("wb")
    env = os.environ.copy()
    env.update(
        {
            "API_PORT": str(api_port),
            "AFD_PORT": str(afd_port),
            "AFD_HOST": "127.0.0.1",
            "AFD_CONNECTOR": connector,
            "HCCL_IF_IP": env.get("HCCL_IF_IP", "192.169.91.106"),
            "PYTHONUNBUFFERED": "1",
            "EXECUTION_MODE": execution_mode,
            "U_BATCHES": str(u_batches),
            "DBO_DECODE_TOKEN_THRESHOLD": str(dbo_decode_token_threshold),
            "DBO_PREFILL_TOKEN_THRESHOLD": str(dbo_prefill_token_threshold),
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
        command.extend(["--prompt-indices", *(str(index) for index in prompt_indices)])
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
        if not log_path.is_file():
            roles[role] = {
                "passed": False,
                "fatal_markers": ["<log missing>"],
            }
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        markers = [marker for marker in FATAL_LOG_MARKERS if marker in text]
        roles[role] = {"passed": not markers, "fatal_markers": markers}
    return {
        "roles": roles,
        "passed": all(result["passed"] for result in roles.values()),
    }


def _ubatch_execution_gate(log_dir: Path, u_batches: int) -> dict[str, Any]:
    """Require runtime evidence that both requested U2 stages executed."""
    required = u_batches == 2
    log_path = log_dir / "attention.log"
    text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )
    observed = any(
        "key=((0," in line and "), (1," in line for line in text.splitlines()
    )
    return {
        "required": required,
        "observed_two_stages": observed,
        "passed": not required or observed,
    }


def _profile_output_gate(profile_dir: Path) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role in ("attention", "ffn"):
        trace_dirs = sorted((profile_dir / role).glob("*_ascend_pt"))
        required_sizes: dict[str, int] = {}
        cann_raw_file_count = 0
        if len(trace_dirs) == 1:
            trace_dir = trace_dirs[0]
            for relative_path in (
                Path("profiler_info_0.json"),
                Path("FRAMEWORK/torch.op_range"),
            ):
                path = trace_dir / relative_path
                required_sizes[str(relative_path)] = (
                    path.stat().st_size if path.is_file() else 0
                )
            cann_raw_file_count = sum(
                1
                for prof_dir in trace_dir.glob("PROF_*")
                if prof_dir.is_dir()
                for path in prof_dir.rglob("*")
                if path.is_file() and path.stat().st_size > 0
            )
        role_passed = bool(
            len(trace_dirs) == 1
            and required_sizes
            and all(size > 0 for size in required_sizes.values())
            and cann_raw_file_count > 0
        )
        roles[role] = {
            "passed": role_passed,
            "trace_dirs": [str(path) for path in trace_dirs],
            "required_sizes": required_sizes,
            "cann_raw_file_count": cann_raw_file_count,
        }
    return {
        "roles": roles,
        "passed": all(result["passed"] for result in roles.values()),
    }


def _capture_command(command: list[str], output: Path) -> None:
    with output.open("wb") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)


def _npu_process_ids(output: str) -> list[int]:
    """Return process IDs from the process table in ``npu-smi info`` output."""
    in_process_table = False
    process_ids: list[int] = []
    for line in output.splitlines():
        if "| NPU" in line and "| Process id" in line:
            in_process_table = True
            continue
        if not in_process_table:
            continue
        match = re.match(r"^\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|", line)
        if match is not None:
            process_ids.append(int(match.group(1)))
    return sorted(set(process_ids))


def _has_npu_process_table(output: str) -> bool:
    return any(
        "| NPU" in line and "| Process id" in line for line in output.splitlines()
    )


def _wait_for_npu_cleanup(
    output: Path,
    *,
    timeout: float = 60,
    poll_interval: float = 2,
) -> dict[str, Any]:
    """Wait for role processes to leave the NPUs without killing other workloads."""
    started = time.monotonic()
    attempts = 0
    returncode = -1
    process_ids: list[int] = []
    process_table_present = False
    while True:
        attempts += 1
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            check=False,
        )
        returncode = result.returncode
        combined_output = result.stdout
        if result.stderr:
            combined_output += result.stderr
        output.write_text(combined_output, encoding="utf-8")
        process_table_present = _has_npu_process_table(result.stdout)
        process_ids = _npu_process_ids(result.stdout)
        if returncode == 0 and process_table_present and not process_ids:
            break
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            break
        time.sleep(min(poll_interval, timeout - elapsed))
    return {
        "passed": returncode == 0 and process_table_present and not process_ids,
        "process_ids": process_ids,
        "process_table_present": process_table_present,
        "npu_smi_returncode": returncode,
        "attempts": attempts,
        "waited_seconds": round(time.monotonic() - started, 3),
    }


def _runtime_manifest(
    *,
    connector: str,
    execution_mode: str,
    u_batches: int,
    dbo_decode_token_threshold: int,
    dbo_prefill_token_threshold: int,
    profile: bool,
) -> dict[str, Any]:
    def git_head(path: str) -> str:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True
        ).strip()

    afd_status = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "status",
            "--short",
            "--untracked-files=no",
        ],
        text=True,
    ).splitlines()
    afd_diff = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "diff", "--binary", "HEAD"],
    )
    return {
        "python": sys.version,
        "plugins": "ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd",
        "cann": "/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1",
        "venv": "/mnt/workspace/code/.venvs/afd-v026",
        "model": "/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp",
        "connector": connector,
        "execution_mode": execution_mode,
        "u_batches": u_batches,
        "dbo_decode_token_threshold": dbo_decode_token_threshold,
        "dbo_prefill_token_threshold": dbo_prefill_token_threshold,
        "profile": profile,
        "profile_role_ranks": [0] if profile else [],
        "torch_profiler_with_stack": False,
        "commits": {
            "afd_plugin": git_head(str(REPO_ROOT)),
            "vllm": git_head("/mnt/workspace/code/vllm-afd-v0.26.0"),
            "vllm_ascend": git_head("/mnt/workspace/code/vllm-ascend-afd-80d8c194f"),
        },
        "afd_plugin_worktree": {
            "tracked_dirty": bool(afd_status),
            "tracked_status": afd_status,
            "tracked_diff_sha256": hashlib.sha256(afd_diff).hexdigest(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path(
            "/mnt/workspace/validation/dsv4_milestone0_20260810/golden_results.json"
        ),
    )
    parser.add_argument("--attention-port", type=int, default=8910)
    parser.add_argument("--ffn-port", type=int, default=8911)
    parser.add_argument("--afd-port", type=int, default=29761)
    parser.add_argument(
        "--connector",
        choices=("CAMP2pAFDConnector", "P2pHcclAFDConnector"),
        default="CAMP2pAFDConnector",
    )
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
    parser.add_argument("--u-batches", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dbo-decode-token-threshold", type=int, default=2)
    parser.add_argument("--dbo-prefill-token-threshold", type=int, default=12)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect plugin-owned Attention and FFN torch-npu traces.",
    )
    args = parser.parse_args()

    if args.u_batches == 2 and args.execution_mode != "eager":
        parser.error("DeepSeek-V4 U2 currently supports only eager execution")
    if (
        args.connector == "P2pHcclAFDConnector"
        and args.execution_mode != "eager"
    ):
        parser.error("P2pHcclAFDConnector currently supports only eager execution")
    if args.dbo_decode_token_threshold < 0:
        parser.error("--dbo-decode-token-threshold must be non-negative")
    if args.dbo_prefill_token_threshold < 0:
        parser.error("--dbo-prefill-token-threshold must be non-negative")

    for port in (args.attention_port, args.ffn_port, args.afd_port):
        if not _port_is_free(port):
            raise RuntimeError(f"port {port} is already in use")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runtime.json").write_text(
        json.dumps(
            _runtime_manifest(
                connector=args.connector,
                execution_mode=args.execution_mode,
                u_batches=args.u_batches,
                dbo_decode_token_threshold=args.dbo_decode_token_threshold,
                dbo_prefill_token_threshold=args.dbo_prefill_token_threshold,
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
                    "role_ranks": [0],
                    "with_stack": False,
                }
            try:
                ffn_process, ffn_handle = _start_role(
                    "ffn",
                    output_dir=cycle_dir,
                    api_port=args.ffn_port,
                    afd_port=args.afd_port,
                    connector=args.connector,
                    execution_mode=args.execution_mode,
                    u_batches=args.u_batches,
                    dbo_decode_token_threshold=args.dbo_decode_token_threshold,
                    dbo_prefill_token_threshold=args.dbo_prefill_token_threshold,
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
                    connector=args.connector,
                    execution_mode=args.execution_mode,
                    u_batches=args.u_batches,
                    dbo_decode_token_threshold=args.dbo_decode_token_threshold,
                    dbo_prefill_token_threshold=args.dbo_prefill_token_threshold,
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
                cycle_result["ubatch_gate"] = _ubatch_execution_gate(
                    cycle_dir,
                    args.u_batches,
                )
                profile_passed = True
                if profile_dir is not None:
                    profile_gate = _profile_output_gate(profile_dir)
                    cycle_result["profile"]["output_gate"] = profile_gate
                    profile_passed = profile_gate["passed"]
                cycle_result["passed"] = bool(
                    cycle_result.get("passed", False)
                    and cycle_result["shutdown"]["passed"]
                    and cycle_result["log_gate"]["passed"]
                    and cycle_result["ubatch_gate"]["passed"]
                    and profile_passed
                )
                cycle_result["npu_cleanup_gate"] = _wait_for_npu_cleanup(
                    cycle_dir / "npu_after_cleanup.txt"
                )
                cycle_result["passed"] = bool(
                    cycle_result["passed"]
                    and cycle_result["npu_cleanup_gate"]["passed"]
                )
                cycle_result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
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
            "connector": args.connector,
            "u_batches": args.u_batches,
            "dbo_decode_token_threshold": args.dbo_decode_token_threshold,
            "dbo_prefill_token_threshold": args.dbo_prefill_token_threshold,
            "profile": args.profile,
        }
        (args.output_dir / "validation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
