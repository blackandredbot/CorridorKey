#!/usr/bin/env python3
"""Quick diagnostic visualizations for a clip's Input/ frames.

Outputs into a Diagnostics/ subfolder:
  - temporal_median.png       — per-pixel median across all frames (naive clean plate)
  - temporal_variance.png     — per-pixel variance heatmap (hot = motion)
  - frame_diff_timeline.png   — mean absolute difference from the median per frame,
                                stacked vertically as a timeline strip
  - frame_diff_values.txt     — per-frame mean diff values as TSV
  - CleanPlates/              — per-frame clean plate estimate (leave-one-out median)

Usage:
    python diagnose_clip.py /path/to/clip
    python diagnose_clip.py /path/to/clip --max-frames 100
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


_IMAGE_EXTS = frozenset((".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff"))


def _load_frames(input_dir: str, max_frames: int | None = None) -> list[np.ndarray]:
    files = sorted(
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
    )
    if max_frames:
        files = files[:max_frames]

    frames = []
    for f in files:
        img = cv2.imread(os.path.join(input_dir, f), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        frames.append(img)

    return frames


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _apply_heatmap(gray_float: np.ndarray) -> np.ndarray:
    """Convert a [0,1] float grayscale to a BGR heatmap via COLORMAP_INFERNO."""
    gray_u8 = np.clip(gray_float * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray_u8, cv2.COLORMAP_INFERNO)


def compute_temporal_median(stack: np.ndarray) -> np.ndarray:
    """Per-pixel median across frames. stack: [N, H, W, 3]."""
    return np.median(stack, axis=0).astype(np.float32)


def compute_temporal_variance(stack: np.ndarray) -> np.ndarray:
    """Per-pixel variance (mean across channels). stack: [N, H, W, 3]."""
    var = np.var(stack, axis=0).mean(axis=2)  # [H, W]
    return var.astype(np.float32)


def compute_frame_diffs(stack: np.ndarray, median: np.ndarray) -> np.ndarray:
    """Mean absolute diff from median per frame. Returns [N, H, W]."""
    diffs = np.abs(stack - median[np.newaxis]).mean(axis=3)  # [N, H, W]
    return diffs.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a clip's temporal behavior")
    parser.add_argument("clip_dir", help="Path to clip directory containing Input/")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames loaded")
    args = parser.parse_args()

    input_dir = os.path.join(args.clip_dir, "Input")
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading frames from {input_dir}...")
    frames = _load_frames(input_dir, args.max_frames)
    if len(frames) < 2:
        print(f"Error: need at least 2 frames, found {len(frames)}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(frames)} frames, shape {frames[0].shape}")
    stack = np.stack(frames, axis=0)  # [N, H, W, 3]

    out_dir = os.path.join(args.clip_dir, "Diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Temporal median (naive clean plate)
    print("Computing temporal median...")
    median = compute_temporal_median(stack)
    cv2.imwrite(os.path.join(out_dir, "temporal_median.png"), _to_uint8(median))

    # 2. Variance heatmap
    print("Computing variance heatmap...")
    variance = compute_temporal_variance(stack)
    # Normalize to [0, 1] for visualization (cap at 99th percentile)
    p99 = np.percentile(variance, 99)
    if p99 > 0:
        var_norm = np.clip(variance / p99, 0, 1)
    else:
        var_norm = variance
    cv2.imwrite(os.path.join(out_dir, "temporal_variance.png"), _apply_heatmap(var_norm))

    # 3. Per-frame diff from median — timeline strip
    print("Computing per-frame diffs...")
    diffs = compute_frame_diffs(stack, median)  # [N, H, W]
    # For each frame, collapse to a single row (mean across height)
    # Result: [N, W] — a timeline where x=pixel column, y=frame index
    timeline = diffs.mean(axis=1)  # [N, W]
    p99_diff = np.percentile(timeline, 99)
    if p99_diff > 0:
        timeline_norm = np.clip(timeline / p99_diff, 0, 1)
    else:
        timeline_norm = timeline
    # Scale up vertically so it's visible
    scale_y = max(1, 400 // len(frames))
    timeline_scaled = np.repeat(timeline_norm, scale_y, axis=0)
    cv2.imwrite(
        os.path.join(out_dir, "frame_diff_timeline.png"),
        _apply_heatmap(timeline_scaled),
    )

    # 4. Also save the raw per-frame mean diff as a simple line plot (text)
    frame_means = diffs.mean(axis=(1, 2))  # [N]
    with open(os.path.join(out_dir, "frame_diff_values.txt"), "w") as f:
        f.write("frame_idx\tmean_diff_from_median\n")
        for i, v in enumerate(frame_means):
            f.write(f"{i}\t{v:.6f}\n")

    # 5. Per-frame clean plate estimates (leave-one-out median)
    print("Computing per-frame clean plate estimates...")
    plate_dir = os.path.join(out_dir, "CleanPlates")
    os.makedirs(plate_dir, exist_ok=True)
    n = stack.shape[0]
    for i in range(n):
        # Leave-one-out: median of all frames except frame i
        if n > 2:
            others = np.concatenate([stack[:i], stack[i + 1:]], axis=0)
            plate_i = np.median(others, axis=0).astype(np.float32)
        else:
            # Only 2 frames — use the other frame as the plate
            plate_i = stack[1 - i]
        cv2.imwrite(
            os.path.join(plate_dir, f"clean_plate_{i:04d}.png"),
            _to_uint8(plate_i),
        )

    print(f"Done. Outputs in {out_dir}/")
    print(f"  temporal_median.png      — naive clean plate (median of all frames)")
    print(f"  temporal_variance.png    — variance heatmap (hot = motion)")
    print(f"  frame_diff_timeline.png  — per-frame diff from median (y=time, x=columns)")
    print(f"  frame_diff_values.txt    — per-frame mean diff values")
    print(f"  CleanPlates/             — per-frame clean plate estimate (leave-one-out median)")
    print()
    print(f"Tip: If temporal_median.png shows ghosting, the foreground is")
    print(f"     present in too many frames for a simple median to reject it.")
    print(f"     The variance heatmap shows where the action is.")


if __name__ == "__main__":
    main()
