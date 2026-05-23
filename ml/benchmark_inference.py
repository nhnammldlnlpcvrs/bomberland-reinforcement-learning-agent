"""Benchmark batch=1 CPU inference for the tiny imitation policy."""

import argparse
import statistics
import time

from ml.models.simple_cnn_policy import TORCH_AVAILABLE, load_checkpoint


if TORCH_AVAILABLE:
    import torch
else:  # pragma: no cover - depends on local env
    torch = None


def _percentile(values, pct):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def benchmark(args):
    if not TORCH_AVAILABLE:
        print("PyTorch not installed, skipping inference benchmark.")
        return None

    model, _checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    sample = torch.zeros((1, 12, 13, 13), dtype=torch.float32)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(sample)

        timings = []
        for _ in range(args.runs):
            start = time.perf_counter()
            _ = model(sample)
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)

    mean_ms = statistics.mean(timings)
    p95_ms = _percentile(timings, 95)
    p99_ms = _percentile(timings, 99)
    print(f"checkpoint: {args.checkpoint}")
    print(f"runs: {args.runs}")
    print(f"batch: 1")
    print(f"mean_ms: {mean_ms:.4f}")
    print(f"p95_ms: {p95_ms:.4f}")
    print(f"p99_ms: {p99_ms:.4f}")
    if mean_ms > 10.0:
        print("WARNING: average inference time is above 10ms.")
    if p99_ms > 50.0:
        print("WARNING: p99 inference time is above 50ms.")
    return {"mean_ms": mean_ms, "p95_ms": p95_ms, "p99_ms": p99_ms}


def main():
    parser = argparse.ArgumentParser(description="Benchmark tiny imitation policy inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
