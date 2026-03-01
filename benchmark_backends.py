"""Benchmark: Torch vs MLX backend inference on real clips.

Usage:
    uv run python benchmark_backends.py [--frames N] [--warmup N]
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from CorridorKeyModule.backend import create_engine

# --- Defaults matching "spam enter after selecting sRGB" ---
INPUT_IS_LINEAR = False
DESPILL_STRENGTH = 1.0  # 10/10
AUTO_DESPECKLE = True
DESPECKLE_SIZE = 400
REFINER_SCALE = 1.0

CLIP_INPUT = "ClipsForInference/fire_BASE/Input.mp4"
CLIP_ALPHA = "ClipsForInference/fire_BASE/AlphaHint/fire_MASK.mp4"


def load_frames(n_frames: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read n_frames of (image_srgb, mask_linear) from the test clip."""
    input_cap = cv2.VideoCapture(CLIP_INPUT)
    alpha_cap = cv2.VideoCapture(CLIP_ALPHA)

    total_input = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_alpha = int(alpha_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    available = min(total_input, total_alpha)
    n = min(n_frames, available)

    frames = []
    for _ in range(n):
        ok1, bgr = input_cap.read()
        ok2, mask_bgr = alpha_cap.read()
        if not ok1 or not ok2:
            break
        img_srgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask_linear = mask_bgr[:, :, 2].astype(np.float32) / 255.0
        frames.append((img_srgb, mask_linear))

    input_cap.release()
    alpha_cap.release()
    print(f"Loaded {len(frames)} frames ({frames[0][0].shape[1]}x{frames[0][0].shape[0]})")
    return frames


def bench_backend(
    backend: str,
    frames: list[tuple[np.ndarray, np.ndarray]],
    warmup: int,
    device: str | None = None,
) -> list[float]:
    """Run inference on all frames, return per-frame times (excluding warmup)."""
    engine = create_engine(backend=backend, device=device)

    # Warmup
    for i in range(min(warmup, len(frames))):
        img, mask = frames[i]
        engine.process_frame(
            img, mask,
            input_is_linear=INPUT_IS_LINEAR,
            despill_strength=DESPILL_STRENGTH,
            auto_despeckle=AUTO_DESPECKLE,
            despeckle_size=DESPECKLE_SIZE,
            refiner_scale=REFINER_SCALE,
        )
    print(f"  Warmup: {warmup} frame(s)")

    # Timed run
    times = []
    results = []
    for i, (img, mask) in enumerate(frames):
        t0 = time.perf_counter()
        res = engine.process_frame(
            img, mask,
            input_is_linear=INPUT_IS_LINEAR,
            despill_strength=DESPILL_STRENGTH,
            auto_despeckle=AUTO_DESPECKLE,
            despeckle_size=DESPECKLE_SIZE,
            refiner_scale=REFINER_SCALE,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        results.append(res)
        print(f"  Frame {i}: {elapsed:.3f}s", end="\r")

    print()
    return times, results


def print_stats(name: str, times: list[float]):
    arr = np.array(times)
    total = arr.sum()
    fps = len(arr) / total if total > 0 else 0
    print(f"\n{'=' * 40}")
    print(f"  {name}")
    print(f"{'=' * 40}")
    print(f"  Frames:  {len(arr)}")
    print(f"  Total:   {total:.2f}s")
    print(f"  Mean:    {arr.mean():.3f}s")
    print(f"  Median:  {np.median(arr):.3f}s")
    print(f"  Std:     {arr.std():.3f}s")
    print(f"  Min:     {arr.min():.3f}s")
    print(f"  Max:     {arr.max():.3f}s")
    print(f"  FPS:     {fps:.2f}")


def compare_outputs(torch_results: list[dict], mlx_results: list[dict]):
    """Compare output similarity between backends."""
    print(f"\n{'=' * 40}")
    print("  Output Comparison (Torch vs MLX)")
    print(f"{'=' * 40}")

    for key in ("alpha", "fg", "comp", "processed"):
        diffs = []
        for tr, mr in zip(torch_results, mlx_results):
            diff = np.abs(tr[key].astype(np.float64) - mr[key].astype(np.float64))
            diffs.append(diff.mean())
        mean_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        print(f"  {key:>10s}  mean_abs_diff={mean_diff:.6f}  max_frame_diff={max_diff:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Torch vs MLX backends")
    parser.add_argument("--frames", type=int, default=50, help="frames to benchmark (default 50)")
    parser.add_argument("--warmup", type=int, default=2, help="warmup frames (default 2)")
    args = parser.parse_args()

    frames = load_frames(args.frames)

    backends = [
        ("Torch (MPS)", "torch", "mps"),
        ("Torch (CPU)", "torch", "cpu"),
        ("MLX", "mlx", None),
    ]

    all_times = {}
    all_results = {}

    for label, backend, device in backends:
        print(f"\n--- {label} ---")
        times, results = bench_backend(backend, frames, args.warmup, device=device)
        print_stats(label, times)
        all_times[label] = times
        all_results[label] = results

    # Speedup summary
    print(f"\n{'=' * 40}")
    print("  Speedup Summary (mean per frame)")
    print(f"{'=' * 40}")
    means = {k: np.mean(v) for k, v in all_times.items()}
    slowest = max(means.values())
    for label, mean in means.items():
        ratio = slowest / mean
        print(f"  {label:<15s}  {mean:.3f}s  ({ratio:.2f}x vs slowest)")

    # Output similarity (Torch MPS vs MLX)
    compare_outputs(all_results["Torch (MPS)"], all_results["MLX"])


if __name__ == "__main__":
    main()
