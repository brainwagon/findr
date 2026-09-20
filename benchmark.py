"""Benchmark the Solver backends over the test frames.

For each Solver backend, loads it once and times a solve of every image in
test-images/, then prints a Markdown table of the results.

Usage:
    python3 benchmark.py [--images DIR] [--repeat N]
"""

import argparse
import os
import platform
import time
from dataclasses import dataclass
from statistics import mean

from solver import BACKENDS, LibrarySolver

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGES = os.path.join(BASE_DIR, "test-images")


@dataclass
class Timing:
    """One timed solve of one frame."""

    frame: str
    seconds: float
    solved: bool


def image_paths(directory):
    """Every JPEG in the directory, sorted by name."""
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith((".jpg", ".jpeg"))
    )


def benchmark_backend(key, frames, repeat=1, warmup=True):
    """Time a backend over every frame.

    Returns (load_seconds, timings) where timings maps a frame name to its
    list of per-solve times.
    """
    start = time.perf_counter()
    solver = LibrarySolver(BACKENDS[key])
    load_seconds = time.perf_counter() - start

    if warmup and frames:
        solver.solve(frames[0])

    timings = {}
    for frame in frames:
        name = os.path.basename(frame)
        timings[name] = []
        for _ in range(repeat):
            start = time.perf_counter()
            result = solver.solve(frame)
            timings[name].append(
                Timing(
                    frame=name,
                    seconds=time.perf_counter() - start,
                    solved=result is not None,
                )
            )
    return load_seconds, timings


def cpu_model():
    """The CPU model name, or a fallback."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except IOError:
        pass
    return platform.processor() or "unknown CPU"


def memory_gib():
    """Total RAM in GiB, or None."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024 ** 2), 1)
    except IOError:
        pass
    return None


def os_name():
    """A short OS description, noting WSL when present."""
    pretty = platform.platform()
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    pretty = line.split("=", 1)[1].strip().strip('"')
    except IOError:
        pass
    suffix = " (WSL2)" if "microsoft" in platform.release().lower() else ""
    return f"{pretty}{suffix}"


def describe_machine():
    """A one-line description of the machine running the benchmark."""
    parts = [
        cpu_model(),
        f"{os.cpu_count()} threads",
    ]
    memory = memory_gib()
    if memory is not None:
        parts.append(f"{memory} GiB RAM")
    parts.append(os_name())
    parts.append(f"Python {platform.python_version()}")
    return " · ".join(parts)


def format_table(keys, results, repeat):
    """Render the per-frame timings as a Markdown table (milliseconds)."""
    frames = list(results[keys[0]][1].keys())
    lines = [
        "| Frame | " + " | ".join(keys) + " |",
        "| --- | " + " | ".join("---:" for _ in keys) + " |",
    ]
    for frame in frames:
        cells = [
            f"{mean(t.seconds for t in results[key][1][frame]) * 1000:.1f}"
            for key in keys
        ]
        lines.append(f"| `{frame}` | " + " | ".join(cells) + " |")

    totals = []
    means = []
    solved = []
    for key in keys:
        groups = list(results[key][1].values())
        per_frame = [mean(t.seconds for t in group) for group in groups]
        totals.append(f"**{sum(per_frame) * 1000:.0f}**")
        means.append(f"**{mean(per_frame) * 1000:.1f}**")
        solved.append(
            f"{sum(1 for group in groups if any(t.solved for t in group))}/{len(groups)}"
        )
    lines.append("| **Total** | " + " | ".join(totals) + " |")
    lines.append("| **Mean** | " + " | ".join(means) + " |")

    loads = [f"{results[key][0]:.2f} s" for key in keys]
    lines.append("| **Database load** | " + " | ".join(loads) + " |")
    lines.append("| **Frames solved** | " + " | ".join(solved) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=DEFAULT_IMAGES,
                        help="directory of test frames")
    parser.add_argument("--repeat", type=int, default=3,
                        help="solves per frame; per-frame times are the mean")
    args = parser.parse_args()

    frames = image_paths(args.images)
    if not frames:
        raise SystemExit(f"No images found in {args.images}")

    print(f"Machine: {describe_machine()}")
    print(f"Frames: {len(frames)} · repeat: {args.repeat}")
    print()

    results = {}
    for key in BACKENDS:
        print(f"Benchmarking {key}...", flush=True)
        results[key] = benchmark_backend(key, frames, repeat=args.repeat)

    print()
    print(format_table(list(BACKENDS), results, args.repeat))


if __name__ == "__main__":
    main()
