#!/usr/bin/env python3
"""Run the same-budget two-instance native DeepSeek-V4 control."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

RECIPE_DIR = Path(__file__).resolve().parent
REPO_ROOT = RECIPE_DIR.parents[3]
PERFORMANCE_RUNNER_PATH = RECIPE_DIR / "run_performance.py"
NATIVE_SERVICE_SCRIPT = REPO_ROOT / "tools/dsv4/run_v023_native_baseline.sh"
DEFAULT_MODEL = Path("/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp")
NATIVE_PLUGINS = "ascend,ascend_model,ascend_model_loader,ascend_kv_connector"
INSTANCE_COUNT = 2
NPUS_PER_INSTANCE = 8


def _load_performance_runner():
    spec = importlib.util.spec_from_file_location(
        "dsv4_afd_performance_runner",
        PERFORMANCE_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load performance runner: {PERFORMANCE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PERF = _load_performance_runner()
SHARED = PERF.SHARED


def _split_load(total_concurrency: int, total_prompts: int) -> list[tuple[int, int]]:
    """Split one logical workload across two services without changing totals."""
    if total_concurrency <= 0:
        raise ValueError("total concurrency must be positive")
    if total_prompts < total_concurrency:
        raise ValueError("total prompts must cover total concurrency")
    active_instances = min(INSTANCE_COUNT, total_concurrency)
    concurrency = [
        total_concurrency // active_instances
        + (1 if index < total_concurrency % active_instances else 0)
        for index in range(active_instances)
    ]
    prompts = [
        total_prompts // active_instances
        + (1 if index < total_prompts % active_instances else 0)
        for index in range(active_instances)
    ]
    result = list(zip(concurrency, prompts, strict=True))
    result.extend([(0, 0)] * (INSTANCE_COUNT - active_instances))
    return result


def _percentile(values: list[float], percentile: float) -> float:
    """Match NumPy's default linear percentile without importing NumPy here."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _merge_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge detailed client results over their shared monotonic time window."""
    if not results:
        raise ValueError("at least one benchmark result is required")

    detailed_fields = ("input_lens", "output_lens", "ttfts", "itls", "start_times")
    merged: dict[str, list[Any]] = {field: [] for field in detailed_fields}
    errors: list[str] = []
    for result in results:
        lengths = {len(result.get(field, [])) for field in detailed_fields}
        if len(lengths) != 1:
            raise ValueError(f"inconsistent detailed result lengths: {lengths}")
        for field in detailed_fields:
            merged[field].extend(result[field])
        errors.extend(result.get("errors", []))

    request_count = len(merged["input_lens"])
    if len(errors) != request_count:
        raise ValueError("errors and detailed request records have different lengths")

    ttfts = [float(value) for value in merged["ttfts"]]
    e2els = [
        ttft + sum(float(value) for value in itls)
        for ttft, itls in zip(ttfts, merged["itls"], strict=True)
    ]
    tpots = [
        (e2el - ttft) / (int(output_len) - 1)
        for e2el, ttft, output_len in zip(
            e2els,
            ttfts,
            merged["output_lens"],
            strict=True,
        )
        if int(output_len) > 1
    ]
    starts = [float(value) for value in merged["start_times"]]
    ends = [start + e2el for start, e2el in zip(starts, e2els, strict=True)]
    if not starts:
        raise ValueError("benchmark results contain no requests")
    duration = max(ends) - min(starts)
    if duration <= 0:
        raise ValueError(f"invalid merged request window: {duration}")

    completed = sum(int(result.get("completed", 0)) for result in results)
    failed = sum(int(result.get("failed", 0)) for result in results)
    total_input = sum(int(result.get("total_input_tokens", 0)) for result in results)
    total_output = sum(int(result.get("total_output_tokens", 0)) for result in results)
    result: dict[str, Any] = {
        "duration": duration,
        "completed": completed,
        "failed": failed,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": completed / duration,
        "output_throughput": total_output / duration,
        "total_token_throughput": (total_input + total_output) / duration,
        "input_lens": merged["input_lens"],
        "output_lens": merged["output_lens"],
        "ttfts": merged["ttfts"],
        "itls": merged["itls"],
        "start_times": merged["start_times"],
        "errors": errors,
        "instance_durations": [float(item["duration"]) for item in results],
        "merge_window": "earliest_request_start_to_latest_request_end",
    }
    for name, values in (("ttft", ttfts), ("tpot", tpots), ("e2el", e2els)):
        values_ms = [value * 1000.0 for value in values]
        result[f"mean_{name}_ms"] = statistics.fmean(values_ms) if values_ms else 0.0
        result[f"std_{name}_ms"] = (
            statistics.pstdev(values_ms) if len(values_ms) > 1 else 0.0
        )
        for percentile in (50, 90, 99):
            result[f"p{percentile}_{name}_ms"] = _percentile(
                values_ms,
                percentile,
            )
    return result


def _service_environment(args: argparse.Namespace, index: int) -> dict[str, str]:
    tensor_parallel_size = int(getattr(args, "tensor_parallel_size", 1))
    start_device = index * NPUS_PER_INSTANCE
    devices = ",".join(
        str(device) for device in range(start_device, start_device + NPUS_PER_INSTANCE)
    )
    env = os.environ.copy()
    env.update(
        {
            "MODEL_PATH": str(args.model),
            "API_PORT": str(args.api_ports[index]),
            "SERVED_MODEL_NAME": f"dsv4-v023-native-{index}",
            "MAX_MODEL_LEN": str(args.max_model_len),
            "MAX_NUM_BATCHED_TOKENS": str(args.max_num_batched_tokens),
            "MAX_NUM_SEQS": str(args.max_num_seqs),
            "GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
            "DATA_PARALLEL_RPC_PORT": str(args.dp_rpc_ports[index]),
            "MASTER_PORT": str(args.master_ports[index]),
            "ASCEND_RT_VISIBLE_DEVICES": devices,
            "HCCL_IF_BASE_PORT": str(args.hccl_base_ports[index]),
            "VLLM_PLUGINS": NATIVE_PLUGINS,
            "ENABLE_MTP": "1" if args.enable_mtp else "0",
            "MTP_NUM_SPECULATIVE_TOKENS": str(args.mtp_num_speculative_tokens),
            "TENSOR_PARALLEL_SIZE": str(tensor_parallel_size),
            "TORCH_PROFILER_WITH_STACK": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _start_services(
    args: argparse.Namespace,
) -> tuple[dict[str, subprocess.Popen[bytes]], list[Any]]:
    processes: dict[str, subprocess.Popen[bytes]] = {}
    handles: list[Any] = []
    for index in range(INSTANCE_COUNT):
        role = f"native{index}"
        handle = (args.output_dir / f"{role}.log").open("wb")
        process = subprocess.Popen(
            ["bash", str(NATIVE_SERVICE_SCRIPT)],
            cwd=REPO_ROOT,
            env=_service_environment(args, index),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes[role] = process
        handles.append(handle)
    return processes, handles


def _fatal_service_logs(
    processes: dict[str, subprocess.Popen[bytes]],
    log_dir: Path,
) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {}
    for role in processes:
        markers = SHARED._fatal_log_markers(SHARED._log_tail(log_dir / f"{role}.log"))
        if markers:
            failures[role] = markers
    return failures


def _run_pair_benchmark(
    *,
    args: argparse.Namespace,
    service_processes: dict[str, subprocess.Popen[bytes]],
    input_len: int,
    output_len: int,
    num_prompts: int,
    concurrency: int,
    run_kind: str,
    repeat: int,
    stem: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split = _split_load(concurrency, num_prompts)
    clients: dict[str, subprocess.Popen[bytes]] = {}
    handles: list[Any] = []
    result_specs: list[tuple[int, Path, int]] = []
    benchmark_dir = args.output_dir / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, (instance_concurrency, instance_prompts) in enumerate(split):
            if instance_concurrency == 0:
                continue
            result_path = benchmark_dir / f"{stem}_native{index}.json"
            log_path = result_path.with_suffix(".log")
            command = PERF._benchmark_command(
                api_port=args.api_ports[index],
                model_path=args.model,
                result_path=result_path,
                input_len=input_len,
                output_len=output_len,
                num_prompts=instance_prompts,
                concurrency=instance_concurrency,
                u_batches=1,
                run_kind=run_kind,
                repeat=repeat,
                served_model=f"dsv4-v023-native-{index}",
                connector=(
                    "native-dp4-tp2-pair"
                    if int(getattr(args, "tensor_parallel_size", 1)) == 2
                    else "native-dp8-pair"
                ),
            )
            handle = log_path.open("wb")
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            clients[f"native{index}"] = process
            handles.append(handle)
            result_specs.append((index, result_path, instance_prompts))

        deadline = time.monotonic() + args.benchmark_timeout
        while any(process.poll() is None for process in clients.values()):
            exited_services = PERF._exited_services(service_processes)
            if exited_services:
                raise RuntimeError(
                    f"native service exited during benchmark: {exited_services}"
                )
            fatal_logs = _fatal_service_logs(service_processes, args.output_dir)
            if fatal_logs:
                raise RuntimeError(
                    "native service logged a fatal error during benchmark: "
                    f"{fatal_logs}"
                )
            failed_clients = {
                role: process.returncode
                for role, process in clients.items()
                if process.poll() not in (None, 0)
            }
            if failed_clients:
                raise RuntimeError(f"benchmark client failed: {failed_clients}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"native pair benchmark timed out after "
                    f"{args.benchmark_timeout:.0f}s"
                )
            time.sleep(1)

        failed_clients = {
            role: process.returncode
            for role, process in clients.items()
            if process.returncode != 0
        }
        if failed_clients:
            raise RuntimeError(f"benchmark client failed: {failed_clients}")

        instance_records: list[dict[str, Any]] = []
        instance_results: list[dict[str, Any]] = []
        for index, result_path, instance_prompts in result_specs:
            if not result_path.is_file():
                raise RuntimeError(f"benchmark result is missing: {result_path}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            gate = PERF._validate_benchmark_result(
                result,
                input_len=input_len,
                output_len=output_len,
                num_prompts=instance_prompts,
            )
            if not gate["passed"]:
                raise RuntimeError(
                    f"native instance benchmark gate failed: {result_path}"
                )
            instance_results.append(result)
            instance_records.append(
                {
                    "instance": index,
                    "concurrency": split[index][0],
                    "num_prompts": instance_prompts,
                    "result_path": str(result_path),
                    "gate": gate,
                }
            )
        return _merge_results(instance_results), instance_records
    except BaseException:
        for process in clients.values():
            PERF._stop_benchmark_process(process)
        raise
    finally:
        for handle in handles:
            handle.close()


def _service_log_gate(log_dir: Path) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for index in range(INSTANCE_COUNT):
        role = f"native{index}"
        path = log_dir / f"{role}.log"
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            markers = SHARED._fatal_log_markers(text)
            shutdown_at = text.find("[shutdown]")
            exception_positions: list[int] = []
            offset = 0
            while (position := text.find("Exception in thread", offset)) >= 0:
                exception_positions.append(position)
                offset = position + 1
            ignored_shutdown_exceptions = sum(
                position > shutdown_at
                for position in exception_positions
                if shutdown_at >= 0
            )
            if exception_positions and ignored_shutdown_exceptions == len(
                exception_positions
            ):
                markers = [
                    marker for marker in markers if marker != "Exception in thread"
                ]
        else:
            markers = ["<log missing>"]
            ignored_shutdown_exceptions = 0
        roles[role] = {
            "passed": not markers,
            "fatal_markers": markers,
            "ignored_shutdown_thread_exceptions": ignored_shutdown_exceptions,
        }
    return {
        "passed": all(result["passed"] for result in roles.values()),
        "roles": roles,
    }


def _shutdown_services(
    processes: dict[str, subprocess.Popen[bytes]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"order": ["native0", "native1"]}
    for role in result["order"]:
        process = processes.get(role)
        if process is None:
            continue
        SHARED._stop_process(process)
        result[f"{role}_returncode"] = process.returncode
    result["passed"] = all(result.get(f"{role}_returncode") == 0 for role in processes)
    return result


def _runtime_manifest(args: argparse.Namespace) -> dict[str, Any]:
    tensor_parallel_size = int(getattr(args, "tensor_parallel_size", 1))
    data_parallel_size = NPUS_PER_INSTANCE // tensor_parallel_size
    topology = {
        "deployment": "two_independent_native_instances",
        "npu_count": INSTANCE_COUNT * NPUS_PER_INSTANCE,
        "instances": [
            {
                "name": f"native{index}",
                "devices": list(
                    range(index * NPUS_PER_INSTANCE, (index + 1) * NPUS_PER_INSTANCE)
                ),
                "data_parallel_size": data_parallel_size,
                "tensor_parallel_size": tensor_parallel_size,
                "expert_parallel": True,
                "api_port": args.api_ports[index],
                "data_parallel_rpc_port": args.dp_rpc_ports[index],
                "master_port": args.master_ports[index],
                "hccl_if_base_port": args.hccl_base_ports[index],
            }
            for index in range(INSTANCE_COUNT)
        ],
    }
    manifest = SHARED._runtime_manifest(
        connector="none-native-control",
        execution_mode="eager",
        u_batches=1,
        dbo_decode_token_threshold=0,
        dbo_prefill_token_threshold=0,
        profile=False,
        enable_mtp=args.enable_mtp,
        mtp_num_speculative_tokens=args.mtp_num_speculative_tokens,
        topology=topology,
    )
    manifest.update(
        {
            "stage": "A3-P8-native-control",
            "plugins": NATIVE_PLUGINS,
            "service_script": str(NATIVE_SERVICE_SCRIPT.relative_to(REPO_ROOT)),
            "reproducibility_files_sha256": {
                str(path.relative_to(REPO_ROOT)): PERF._file_sha256(path)
                for path in (
                    Path(__file__).resolve(),
                    PERFORMANCE_RUNNER_PATH,
                    NATIVE_SERVICE_SCRIPT,
                )
            },
            "service": {
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_num_seqs": args.max_num_seqs,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "tensor_parallel_size": tensor_parallel_size,
            },
            "workload": {
                "input_len": args.input_len,
                "output_len": args.output_len,
                "concurrencies": args.concurrencies,
                "repeats": args.repeats,
                "prompts_per_concurrency": args.prompts_per_concurrency,
                "min_prompts": args.min_prompts,
                "warmup_input_len": args.warmup_input_len,
                "warmup_output_len": args.warmup_output_len,
                "warmup_prompts": args.warmup_prompts,
                "warmup_concurrency": args.warmup_concurrency,
                "temperature": 0,
                "ignore_eos": True,
                "request_rate": "inf",
                "seed": 1024,
                "merge_window": "earliest_request_start_to_latest_request_end",
                "max_output_throughput_cv": args.max_throughput_cv,
            },
        }
    )
    return manifest


def _run(args: argparse.Namespace) -> dict[str, Any]:
    checked_ports = [*args.api_ports, *args.dp_rpc_ports, *args.master_ports]
    for port in checked_ports:
        if not SHARED._port_is_free(port):
            raise RuntimeError(f"port {port} is already in use")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "runtime.json").write_text(
        json.dumps(_runtime_manifest(args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    SHARED._capture_command(["npu-smi", "info"], args.output_dir / "npu_before.txt")

    processes: dict[str, subprocess.Popen[bytes]] = {}
    handles: list[Any] = []
    monitor: Any = None
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "passed": False,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "deployment": (
            "two_independent_native_dp4_tp2_instances"
            if int(getattr(args, "tensor_parallel_size", 1)) == 2
            else "two_independent_native_dp8_instances"
        ),
        "tensor_parallel_size": int(getattr(args, "tensor_parallel_size", 1)),
        "enable_mtp": args.enable_mtp,
        "mtp_num_speculative_tokens": args.mtp_num_speculative_tokens,
    }
    startup_started = time.monotonic()
    try:
        processes, handles = _start_services(args)
        for index, port in enumerate(args.api_ports):
            SHARED._wait_for_api(
                f"http://127.0.0.1:{port}/v1/models",
                processes,
                args.output_dir,
                args.startup_timeout,
            )
            summary[f"native{index}_ready"] = True
        summary["startup_seconds"] = round(time.monotonic() - startup_started, 3)
        SHARED._capture_command(["npu-smi", "info"], args.output_dir / "npu_ready.txt")

        monitor = PERF._NPUMonitor(
            args.output_dir / "npu_samples.jsonl",
            args.npu_sample_interval,
        )
        monitor.start()

        warmup_result, warmup_instances = _run_pair_benchmark(
            args=args,
            service_processes=processes,
            input_len=args.warmup_input_len,
            output_len=args.warmup_output_len,
            num_prompts=args.warmup_prompts,
            concurrency=args.warmup_concurrency,
            run_kind="warmup",
            repeat=0,
            stem="warmup",
        )
        summary["warmup_instances"] = warmup_instances
        summary["warmup_gate"] = PERF._validate_benchmark_result(
            warmup_result,
            input_len=args.warmup_input_len,
            output_len=args.warmup_output_len,
            num_prompts=args.warmup_prompts,
        )
        if not summary["warmup_gate"]["passed"]:
            raise RuntimeError("native pair warmup gate failed")

        for concurrency in args.concurrencies:
            for repeat in range(1, args.repeats + 1):
                num_prompts = max(
                    args.min_prompts,
                    concurrency * args.prompts_per_concurrency,
                )
                stem = f"c{concurrency}_r{repeat}"
                result, instance_records = _run_pair_benchmark(
                    args=args,
                    service_processes=processes,
                    input_len=args.input_len,
                    output_len=args.output_len,
                    num_prompts=num_prompts,
                    concurrency=concurrency,
                    run_kind="measurement",
                    repeat=repeat,
                    stem=stem,
                )
                gate = PERF._validate_benchmark_result(
                    result,
                    input_len=args.input_len,
                    output_len=args.output_len,
                    num_prompts=num_prompts,
                )
                record = {
                    "concurrency": concurrency,
                    "repeat": repeat,
                    "num_prompts": num_prompts,
                    "split": _split_load(concurrency, num_prompts),
                    "instances": instance_records,
                    "gate": gate,
                    "result": result,
                }
                records.append(record)
                if not gate["passed"]:
                    raise RuntimeError(f"native pair result gate failed: {stem}")
        summary["aggregate"] = PERF._aggregate_results(
            records,
            max_throughput_cv=args.max_throughput_cv,
        )
    except BaseException as error:
        summary["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        raise
    finally:
        if monitor is not None:
            summary["npu_monitor"] = monitor.stop()
        summary["shutdown"] = _shutdown_services(processes)
        for handle in handles:
            handle.close()
        summary["log_gate"] = _service_log_gate(args.output_dir)
        summary["npu_cleanup_gate"] = SHARED._wait_for_npu_cleanup(
            args.output_dir / "npu_after_cleanup.txt"
        )
        summary["records"] = records
        summary["passed"] = all(
            (
                summary.get("warmup_gate", {}).get("passed", False),
                summary.get("aggregate", {}).get("passed", False),
                summary.get("npu_monitor", {}).get("passed", False),
                summary.get("shutdown", {}).get("passed", False),
                summary.get("log_gate", {}).get("passed", False),
                summary.get("npu_cleanup_gate", {}).get("passed", False),
            )
        )
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (args.output_dir / "performance_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--api-ports", type=int, nargs=2, default=[8920, 8921])
    parser.add_argument("--dp-rpc-ports", type=int, nargs=2, default=[29350, 29450])
    parser.add_argument("--master-ports", type=int, nargs=2, default=[29351, 29451])
    parser.add_argument("--hccl-base-ports", type=int, nargs=2, default=[53000, 54000])
    parser.add_argument("--startup-timeout", type=float, default=3600)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        choices=(1, 2),
        default=1,
        help="Use each 8-NPU native instance as DP8/TP1 or DP4/TP2.",
    )
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompts-per-concurrency", type=int, default=4)
    parser.add_argument("--min-prompts", type=int, default=8)
    parser.add_argument("--warmup-input-len", type=int, default=256)
    parser.add_argument("--warmup-output-len", type=int, default=16)
    parser.add_argument("--warmup-prompts", type=int, default=16)
    parser.add_argument("--warmup-concurrency", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--npu-sample-interval", type=float, default=2.0)
    parser.add_argument("--max-throughput-cv", type=float, default=0.10)
    parser.add_argument("--benchmark-timeout", type=float, default=1800)
    parser.add_argument(
        "--enable-mtp",
        action="store_true",
        default=os.environ.get("ENABLE_MTP", "0") == "1",
    )
    parser.add_argument(
        "--mtp-num-speculative-tokens",
        type=int,
        default=int(os.environ.get("MTP_NUM_SPECULATIVE_TOKENS", "1")),
    )
    args = parser.parse_args()

    positive_fields = (
        "startup_timeout",
        "input_len",
        "output_len",
        "repeats",
        "prompts_per_concurrency",
        "min_prompts",
        "warmup_input_len",
        "warmup_output_len",
        "warmup_prompts",
        "warmup_concurrency",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "npu_sample_interval",
        "benchmark_timeout",
        "mtp_num_speculative_tokens",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if any(concurrency <= 0 for concurrency in args.concurrencies):
        parser.error("--concurrencies values must be positive")
    if args.input_len + args.output_len > args.max_model_len:
        parser.error("input length plus output length exceeds --max-model-len")
    if args.warmup_input_len + args.warmup_output_len > args.max_model_len:
        parser.error("warmup lengths exceed --max-model-len")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if not 0 < args.max_throughput_cv <= 1:
        parser.error("--max-throughput-cv must be in (0, 1]")
    all_ports = [
        *args.api_ports,
        *args.dp_rpc_ports,
        *args.master_ports,
        *args.hccl_base_ports,
    ]
    if len(set(all_ports)) != len(all_ports):
        parser.error("all API, DP RPC, master, and HCCL base ports must be distinct")
    if any(port < 1024 or port > 60000 for port in all_ports):
        parser.error("ports must be in [1024, 60000]")
    if args.enable_mtp and args.mtp_num_speculative_tokens != 1:
        parser.error("the A3-P8 MTP comparison requires exactly one speculative token")
    load_points = [
        (
            concurrency,
            max(args.min_prompts, concurrency * args.prompts_per_concurrency),
        )
        for concurrency in args.concurrencies
    ]
    load_points.append((args.warmup_concurrency, args.warmup_prompts))
    for concurrency, prompts in load_points:
        try:
            _split_load(concurrency, prompts)
        except ValueError as error:
            parser.error(str(error))
    return args


def main() -> None:
    args = _parse_args()
    summary = _run(args)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
