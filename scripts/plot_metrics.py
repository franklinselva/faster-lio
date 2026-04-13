#!/usr/bin/env python3
"""Plot performance and computation metrics from evaluate_lio / run_mapping output.

Reads the timer log (CSV) and trajectory files to produce:
  - Per-frame computation time breakdown (stacked area)
  - FPS over time
  - Time distribution histogram
  - Trajectory comparison and ATE over time (if ground truth provided)

Usage:
    python scripts/plot_metrics.py --time_log ./Log/time.log
    python scripts/plot_metrics.py --time_log ./Log/time.log --traj ./Log/traj.txt --gt ./Log/ground_truth_tum.txt
    python scripts/plot_metrics.py --time_log ./Log/time.log --save ./Log/metrics.pdf
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_time_log(path: str) -> dict[str, np.ndarray]:
    """Load timer CSV: first row is comma-separated names, subsequent rows are values."""
    records: dict[str, list[float]] = {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        names = [h.strip() for h in header if h.strip()]
        for name in names:
            records[name] = []
        for row in reader:
            for i, name in enumerate(names):
                if i < len(row) and row[i].strip():
                    try:
                        records[name].append(float(row[i].strip()))
                    except ValueError:
                        records[name].append(0.0)
                else:
                    records[name].append(0.0)
    return {k: np.array(v) for k, v in records.items() if len(v) > 0}


def load_tum_trajectory(path: str):
    """Load TUM format: timestamp x y z qx qy qz qw"""
    timestamps, positions = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                timestamps.append(float(parts[0]))
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(timestamps), np.array(positions)


def align_umeyama(src, dst):
    """Umeyama alignment (rotation + translation, no scale)."""
    assert src.shape == dst.shape
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.T @ dst_c / n
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(3)
    if d < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    t = dst_mean - R @ src_mean
    return R, t


def find_nearest(gt_ts, query_ts, max_dt=0.05):
    """For each query timestamp, find nearest ground truth index."""
    indices = np.searchsorted(gt_ts, query_ts)
    matches = []
    for i, idx in enumerate(indices):
        best_idx, best_dt = -1, max_dt
        for candidate in [idx - 1, idx]:
            if 0 <= candidate < len(gt_ts):
                dt = abs(gt_ts[candidate] - query_ts[i])
                if dt < best_dt:
                    best_dt = dt
                    best_idx = candidate
        matches.append(best_idx)
    return matches


def plot_computation(records: dict[str, np.ndarray], save_path=None):
    """Plot computation time breakdown and FPS."""
    total_key = "Laser Mapping Single Run"
    has_total = total_key in records

    # Identify sub-timers (indented names or all non-total)
    sub_keys = [k for k in records if k != total_key]
    # Sort sub-keys by mean time descending for better visualization
    sub_keys.sort(key=lambda k: records[k].mean(), reverse=True)

    n_frames = max(len(v) for v in records.values())
    frames = np.arange(n_frames)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Faster-LIO Performance Metrics", fontsize=14, fontweight="bold")

    # --- Plot 1: Per-frame total computation time ---
    ax = axes[0, 0]
    if has_total:
        total = records[total_key]
        ax.plot(frames[:len(total)], total, linewidth=0.8, color="steelblue", alpha=0.7)
        mean_t = total.mean()
        median_t = np.median(total)
        ax.axhline(mean_t, color="red", linestyle="--", linewidth=1, label=f"Mean: {mean_t:.2f} ms")
        ax.axhline(median_t, color="orange", linestyle=":", linewidth=1, label=f"Median: {median_t:.2f} ms")
        ax.legend(fontsize=9)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-Frame Total Computation Time")
    ax.grid(True, alpha=0.3)

    # --- Plot 2: FPS over time ---
    ax = axes[0, 1]
    if has_total:
        total = records[total_key]
        fps = 1000.0 / np.maximum(total, 0.1)  # avoid div by zero
        # Smoothed FPS (rolling window)
        window = min(20, len(fps) // 4) if len(fps) > 8 else 1
        if window > 1:
            kernel = np.ones(window) / window
            fps_smooth = np.convolve(fps, kernel, mode="valid")
            offset = (len(fps) - len(fps_smooth)) // 2
            ax.plot(frames[:len(fps)], fps, linewidth=0.5, alpha=0.3, color="steelblue", label="Raw")
            ax.plot(frames[offset:offset + len(fps_smooth)], fps_smooth, linewidth=1.5,
                    color="steelblue", label=f"Smoothed (w={window})")
        else:
            ax.plot(frames[:len(fps)], fps, linewidth=1, color="steelblue")
        mean_fps = fps.mean()
        ax.axhline(mean_fps, color="red", linestyle="--", linewidth=1, label=f"Mean: {mean_fps:.1f} FPS")
        ax.axhline(10, color="gray", linestyle=":", linewidth=0.8, alpha=0.5, label="10 Hz (real-time)")
        ax.legend(fontsize=9)
    ax.set_xlabel("Frame")
    ax.set_ylabel("FPS")
    ax.set_title("Processing Rate (FPS)")
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Time breakdown (stacked bar / area) ---
    ax = axes[1, 0]
    if sub_keys:
        # Pick top sub-timers by mean time
        top_n = min(6, len(sub_keys))
        top_keys = sub_keys[:top_n]
        bottom = np.zeros(n_frames)
        colors = plt.cm.Set2(np.linspace(0, 1, top_n))
        for i, key in enumerate(top_keys):
            vals = records[key]
            padded = np.zeros(n_frames)
            padded[:len(vals)] = vals
            label = key.strip()
            ax.bar(frames, padded, bottom=bottom, width=1.0, color=colors[i],
                   alpha=0.8, label=f"{label} ({vals.mean():.1f} ms)")
            bottom += padded
        ax.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Computation Breakdown (Top Components)")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Plot 4: Distribution histogram ---
    ax = axes[1, 1]
    if has_total:
        total = records[total_key]
        ax.hist(total, bins=min(50, len(total) // 2 + 1), color="steelblue",
                edgecolor="white", alpha=0.8)
        ax.axvline(total.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {total.mean():.2f} ms")
        p95 = np.percentile(total, 95)
        ax.axvline(p95, color="orange", linestyle=":", linewidth=1.5,
                   label=f"P95: {p95:.2f} ms")
        ax.legend(fontsize=9)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Computation Time Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved computation plot to: {save_path}")
    return fig


def plot_trajectory(est_ts, est_pos, gt_ts=None, gt_pos=None, time_offset=0.0, save_path=None):
    """Plot trajectory comparison and ATE if ground truth is available."""
    has_gt = gt_ts is not None and gt_pos is not None and len(gt_ts) > 0

    n_plots = 3 if has_gt else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("Trajectory Analysis", fontsize=14, fontweight="bold")

    # --- Trajectory 2D (XY) ---
    ax = axes[0]
    ax.plot(est_pos[:, 0], est_pos[:, 1], linewidth=1.2, color="steelblue", label="Estimated")
    if has_gt:
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], linewidth=1.2, color="coral", alpha=0.7, label="Ground Truth")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory (XY)")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if has_gt:
        # Match and align
        query_ts = est_ts + time_offset
        matches = find_nearest(gt_ts, query_ts)
        valid = [(i, m) for i, m in enumerate(matches) if m >= 0]

        if len(valid) >= 3:
            est_matched = np.array([est_pos[i] for i, _ in valid])
            gt_matched = np.array([gt_pos[m] for _, m in valid])

            R, t = align_umeyama(est_matched, gt_matched)
            est_aligned = (R @ est_matched.T).T + t

            ate_errors = np.linalg.norm(est_aligned - gt_matched, axis=1)
            matched_ts = np.array([est_ts[i] for i, _ in valid])
            rel_ts = matched_ts - matched_ts[0]

            # --- ATE over time ---
            ax = axes[1]
            ax.plot(rel_ts, ate_errors, linewidth=0.8, color="steelblue", alpha=0.7)
            # Smoothed
            window = min(20, len(ate_errors) // 4) if len(ate_errors) > 8 else 1
            if window > 1:
                kernel = np.ones(window) / window
                smooth = np.convolve(ate_errors, kernel, mode="valid")
                offset = (len(ate_errors) - len(smooth)) // 2
                ax.plot(rel_ts[offset:offset + len(smooth)], smooth, linewidth=1.5,
                        color="red", label=f"Smoothed (w={window})")
            rmse = np.sqrt(np.mean(ate_errors ** 2))
            ax.axhline(rmse, color="orange", linestyle="--", linewidth=1,
                       label=f"RMSE: {rmse:.4f} m")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("ATE (m)")
            ax.set_title("Absolute Trajectory Error Over Time")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            # --- Aligned trajectory ---
            ax = axes[2]
            est_full_aligned = (R @ est_pos.T).T + t
            ax.plot(est_full_aligned[:, 0], est_full_aligned[:, 1], linewidth=1.2,
                    color="steelblue", label="Estimated (aligned)")
            ax.plot(gt_pos[:, 0], gt_pos[:, 1], linewidth=1.2, color="coral",
                    alpha=0.7, label="Ground Truth")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title("Aligned Trajectory (XY)")
            ax.set_aspect("equal")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, f"Only {len(valid)} matches\n(need >= 3)",
                         ha="center", va="center", transform=axes[1].transAxes)
            axes[2].text(0.5, 0.5, "Insufficient matches", ha="center", va="center",
                         transform=axes[2].transAxes)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved trajectory plot to: {save_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot faster-LIO performance and computation metrics")
    parser.add_argument("--time_log", default="./Log/time.log", help="Timer CSV from DumpIntoFile")
    parser.add_argument("--traj", default=None, help="Estimated trajectory (TUM format)")
    parser.add_argument("--gt", default=None, help="Ground truth trajectory (TUM format)")
    parser.add_argument("--time_offset", type=float, default=0.0,
                        help="Time offset for GT matching (seconds)")
    parser.add_argument("--save", default="./results/metrics",
                        help="Save plots to file (e.g. ./results/metrics). "
                             "Adds _comp and _traj suffixes.")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots interactively")
    args = parser.parse_args()

    time_log_path = Path(args.time_log)
    if not time_log_path.exists():
        print(f"Error: time log not found: {time_log_path}", file=sys.stderr)
        sys.exit(1)

    records = load_time_log(str(time_log_path))
    print(f"Loaded {len(records)} timer records from {time_log_path}")
    for name, vals in records.items():
        print(f"  {name.strip():40s}  mean={vals.mean():.2f} ms  std={vals.std():.2f} ms  "
              f"n={len(vals)}  p95={np.percentile(vals, 95):.2f} ms")

    save_comp = f"{args.save}_comp.pdf" if args.save else None
    plot_computation(records, save_path=save_comp)

    if args.traj and Path(args.traj).exists():
        est_ts, est_pos = load_tum_trajectory(args.traj)
        print(f"Loaded {len(est_ts)} estimated poses from {args.traj}")

        gt_ts, gt_pos = None, None
        if args.gt and Path(args.gt).exists():
            gt_ts, gt_pos = load_tum_trajectory(args.gt)
            print(f"Loaded {len(gt_ts)} ground truth poses from {args.gt}")

        save_traj = f"{args.save}_traj.pdf" if args.save else None
        plot_trajectory(est_ts, est_pos, gt_ts, gt_pos,
                        time_offset=args.time_offset, save_path=save_traj)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
