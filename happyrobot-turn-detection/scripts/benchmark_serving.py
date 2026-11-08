#!/usr/bin/env python3
"""Measure sequential and concurrent HTTP inference latency."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import mimetypes
import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import httpx
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def percentile(values: List[float], value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(samples: List[Dict[str, Any]], wall_seconds: float, concurrency: int) -> Dict[str, Any]:
    latencies = [sample["latency_ms"] for sample in samples]
    server_feature = [
        sample["response"]["timing"]["feature_extraction_ms"]
        for sample in samples
        if sample["response"] is not None
    ]
    server_model = [
        sample["response"]["timing"]["model_inference_ms"]
        for sample in samples
        if sample["response"] is not None
    ]
    successful = sum(sample["response"] is not None for sample in samples)
    failed = len(samples) - successful
    return {
        "concurrency": concurrency,
        "requests": len(samples),
        "successful_requests": successful,
        "failed_requests": failed,
        "wall_time_seconds": wall_seconds,
        "requests_per_second": successful / wall_seconds if wall_seconds else 0.0,
        "http_request_latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "median": statistics.median(latencies) if latencies else 0.0,
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "server_feature_extraction_ms": {
            "mean": statistics.fmean(server_feature) if server_feature else None,
            "median": statistics.median(server_feature) if server_feature else None,
            "p95": percentile(server_feature, 0.95) if server_feature else None,
            "p99": percentile(server_feature, 0.99) if server_feature else None,
        },
        "server_model_only_inference_ms": {
            "mean": statistics.fmean(server_model) if server_model else None,
            "median": statistics.median(server_model) if server_model else None,
            "p95": percentile(server_model, 0.95) if server_model else None,
            "p99": percentile(server_model, 0.99) if server_model else None,
        },
        "notes": "HTTP latency is measured client-side from request dispatch through response body receipt; server feature/model times are returned by the API.",
    }


def run_one(client: httpx.Client, url: str, filename: str, content: bytes, media_type: str) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.post(
            url,
            files={"audio": (filename, content, media_type)},
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        if response.is_success:
            return {"latency_ms": latency_ms, "response": response.json(), "error": None}
        return {"latency_ms": latency_ms, "response": None, "error": f"HTTP {response.status_code}: {response.text}"}
    except httpx.HTTPError as exc:
        return {
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "response": None,
            "error": str(exc),
        }


def run_batch(
    client: httpx.Client,
    url: str,
    filename: str,
    content: bytes,
    media_type: str,
    count: int,
    concurrency: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    if concurrency == 1:
        samples = [run_one(client, url, filename, content, media_type) for _ in range(count)]
    else:
        samples = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(run_one, client, url, filename, content, media_type)
                for _ in range(count)
            ]
            for future in as_completed(futures):
                samples.append(future.result())
    wall_seconds = time.perf_counter() - started
    result = summarize(samples, wall_seconds, concurrency)
    result["errors"] = [sample["error"] for sample in samples if sample["error"] is not None][:20]
    return result


def write_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def default_report_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return ROOT / "reports" / "serving" / "benchmark_runs" / f"http_{timestamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="WAV or FLAC input file; can be synthetic.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--requests", type=int, default=100, help="Requests per concurrency level.")
    parser.add_argument("--concurrency", default="1,2,4,8", help="Comma-separated worker levels.")
    parser.add_argument(
        "--server-workers",
        type=int,
        default=None,
        help="Uvicorn worker count; recorded as null if the server configuration is unknown.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="New report path (must not already exist). Defaults to a timestamped file under reports/serving/benchmark_runs/.",
    )
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    if args.requests < 1:
        parser.error("--requests must be positive")
    if args.server_workers is not None and args.server_workers < 1:
        parser.error("--server-workers must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        concurrency_levels = sorted({int(value) for value in args.concurrency.split(",")})
    except ValueError:
        parser.error("--concurrency must be a comma-separated list of positive integers")
    if not concurrency_levels or concurrency_levels[0] < 1:
        parser.error("--concurrency values must be positive")
    output_path = args.output
    if output_path is None:
        output_path = default_report_path()
    if output_path.exists():
        parser.error(f"Refusing to overwrite an existing report: {output_path}")

    audio_info = sf.info(str(args.audio))
    content = args.audio.read_bytes()
    media_type = mimetypes.guess_type(args.audio.name)[0] or "application/octet-stream"
    with httpx.Client(timeout=args.timeout, limits=httpx.Limits(max_connections=max(concurrency_levels))) as client:
        health = client.get(args.url.rsplit("/", 1)[0] + "/ready")
        health.raise_for_status()
        sequential = run_batch(client, args.url, args.audio.name, content, media_type, args.requests, 1)
        concurrent = [
            run_batch(
                client,
                args.url,
                args.audio.name,
                content,
                media_type,
                args.requests,
                level,
            )
            for level in concurrency_levels
            if level != 1
        ]

    output = {
        "url": args.url,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or "unspecified",
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "fastapi": importlib.metadata.version("fastapi"),
            "uvicorn": importlib.metadata.version("uvicorn"),
            "httpx": importlib.metadata.version("httpx"),
        },
        "audio": {
            "filename": args.audio.name,
            "size_bytes": len(content),
            "duration_seconds": float(audio_info.duration),
            "sample_rate_hz": int(audio_info.samplerate),
            "channels": int(audio_info.channels),
        },
        "methodology": {
            "sequential_requests": args.requests,
            "requests_per_concurrency_level": args.requests,
            "concurrency_levels": concurrency_levels,
            "warm_connection_pool": True,
            "same_audio_reused": True,
            "server_worker_count": args.server_workers,
        },
        "sequential": sequential,
        "concurrent": concurrent,
    }
    write_report(output_path, output)
    print(json.dumps(output, indent=2, allow_nan=False))
    print(f"\nSaved benchmark report to {output_path}")


if __name__ == "__main__":
    main()
