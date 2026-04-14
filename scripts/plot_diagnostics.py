#!/usr/bin/env python3
"""Plot per-frame LIO diagnostics CSV.

Usage:
    uv run --with matplotlib --with numpy --with pandas python3 plot_diagnostics.py \\
        --csv build/results_diag/diagnostics.csv \\
        --output build/results_diag

Generates a 4×3 grid covering algorithm health, state evolution, AND
per-frame compute / memory / CPU usage so you can see hot spots directly.

The CSV is produced by LaserMapping::EnableDiagnostics when faster-lio is
compiled with -DFASTER_LIO_ENABLE_DIAGNOSTICS=ON (default in non-Release).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_diagnostics(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Elapsed seconds for readable x-axis
    t = df["timestamp"] - df["timestamp"].iloc[0]

    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(f"LIO Diagnostics — {csv_path.name}",
                 fontsize=14, fontweight="bold")

    # ── Row 0: feature health ───────────────────────────────────────────
    ax = axes[0][0]
    ax.plot(t, df["scan_undistort_pts"], label="undistorted", alpha=0.7)
    ax.plot(t, df["scan_down_pts"], label="after voxel", alpha=0.7)
    ax.plot(t, df["effect_feat"], label="effective (IEKF)", linewidth=2)
    ax.set_title("Point counts")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("points")
    ax.set_yscale("log"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(t, df["effect_ratio"] * 100)
    ax.axhline(20, color="orange", linestyle="--", alpha=0.5, label="20% (starving)")
    ax.set_title("Effective feature ratio")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("% of down-sampled")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0][2]
    ax.plot(t, df["curv_min_ms"], label="min", alpha=0.6)
    ax.plot(t, df["curv_max_ms"], label="max", alpha=0.6)
    ax.axhspan(80, 110, color="green", alpha=0.1, label="expected 10Hz range")
    ax.set_title("Per-point timing span (curvature)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("ms from scan start")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Row 1: residuals / fit / map ───────────────────────────────────
    ax = axes[1][0]
    ax.plot(t, df["residual_mean"] * 100, label="mean", alpha=0.7)
    ax.plot(t, df["residual_rms"] * 100, label="rms", alpha=0.7)
    ax.plot(t, df["residual_max"] * 100, label="max", alpha=0.5)
    ax.set_title("Point-to-plane residuals")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("cm")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.plot(t, df["fit_quality_mean"] * 100, label="mean")
    ax.plot(t, df["fit_quality_rms"] * 100, label="rms")
    ax.set_title("k-NN plane fit RMS (fit_quality)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("cm")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1][2]
    ax.plot(t, df["map_grids"])
    ax.set_title("iVox valid grid count")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("grids")
    ax.grid(alpha=0.3)

    # ── Row 2: state evolution ─────────────────────────────────────────
    ax = axes[2][0]
    for axis in "xyz":
        ax.plot(t, df[f"pos_{axis}"], label=axis, alpha=0.8)
    ax.set_title("Position (world frame)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("m")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2][1]
    for axis, c in zip("xyz", ["C0", "C1", "C2"]):
        ax.plot(t, df[f"bg_{axis}"], label=f"gyr_b_{axis}", alpha=0.7, color=c)
    for axis, c in zip("xyz", ["C3", "C4", "C5"]):
        ax.plot(t, df[f"ba_{axis}"], label=f"acc_b_{axis}", alpha=0.4,
                color=c, linestyle="--")
    ax.set_title("IMU biases")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    ax = axes[2][2]
    for axis in "xyz":
        ax.plot(t, df[f"ext_T_{axis}"], label=axis)
    ax.set_title("LiDAR-IMU extrinsic translation (online)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("m")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Row 3: compute / memory / CPU observability ────────────────────
    ax = axes[3][0]
    if "t_run_total_us" in df.columns:
        # Stacked area: undistort + downsample + iekf + mapinc
        comps = ["t_undistort_us", "t_downsample_us", "t_iekf_us", "t_mapinc_us"]
        labels = ["undistort", "downsample", "IEKF", "mapincr"]
        # ms for readability
        stacks = [df[c] / 1000.0 for c in comps]
        ax.stackplot(t, *stacks, labels=labels, alpha=0.75)
        ax.plot(t, df["t_run_total_us"] / 1000.0, color="black",
                linewidth=1, label="Run() total", alpha=0.6)
        ax.set_title("Per-frame compute breakdown")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("ms / frame")
        ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Timing columns absent\n(rebuild with diagnostics ON)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    ax = axes[3][1]
    if "rss_mb" in df.columns:
        ax.plot(t, df["rss_mb"], color="C2", linewidth=2)
        ax.set_title("Resident memory (RSS)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("MB")
        ax.grid(alpha=0.3)
        # Annotate growth
        delta = df["rss_mb"].iloc[-1] - df["rss_mb"].iloc[0]
        ax.text(0.02, 0.95, f"Δ = {delta:+.1f} MB", transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    else:
        ax.set_axis_off()

    ax = axes[3][2]
    if "cpu_delta_us" in df.columns:
        # Convert to ms; first row is 0 (no prev sample), skip it for plot
        cpu_ms = df["cpu_delta_us"] / 1000.0
        ax.plot(t.iloc[1:], cpu_ms.iloc[1:], color="C3", alpha=0.6, linewidth=1)
        # Rolling mean smooths the spikes
        win = max(5, len(df) // 50)
        ax.plot(t, cpu_ms.rolling(win, min_periods=1).mean(),
                color="darkred", linewidth=2,
                label=f"rolling mean (w={win})")
        ax.set_title("CPU time per frame (delta)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("ms")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    else:
        ax.set_axis_off()

    plt.tight_layout()
    out_path = out_dir / "diagnostics.png"
    plt.savefig(out_path, dpi=110)
    print(f"Saved: {out_path}")

    # ── Console summary ────────────────────────────────────────────────
    print("\nSummary (first/last 10% windows):")
    n = max(1, len(df) // 10)
    head = df.head(n); tail = df.tail(n)
    print(f"  effect_feat:     {head['effect_feat'].mean():>6.0f}  →  {tail['effect_feat'].mean():>6.0f}")
    print(f"  effect_ratio:    {head['effect_ratio'].mean()*100:>5.1f}%  →  {tail['effect_ratio'].mean()*100:>5.1f}%")
    print(f"  residual_rms:    {head['residual_rms'].mean()*100:>5.2f}cm  →  {tail['residual_rms'].mean()*100:>5.2f}cm")
    print(f"  curv_max:        {head['curv_max_ms'].mean():>5.1f}ms  →  {tail['curv_max_ms'].mean():>5.1f}ms")
    if "t_run_total_us" in df.columns:
        print(f"  Run() time:      {df['t_run_total_us'].mean()/1000:>5.2f}ms (mean)  "
              f"max={df['t_run_total_us'].max()/1000:.1f}ms")
        for col, label in [("t_undistort_us", "  undistort:    "),
                            ("t_downsample_us", "  downsample:   "),
                            ("t_iekf_us", "  IEKF:         "),
                            ("t_mapinc_us", "  mapincr:      ")]:
            print(f"  {label}{df[col].mean()/1000:>5.2f}ms  "
                  f"({df[col].sum()/df['t_run_total_us'].sum()*100:>4.1f}% of Run)")
    if "rss_mb" in df.columns:
        print(f"  RSS:             {df['rss_mb'].iloc[0]:>6.1f}MB  →  {df['rss_mb'].iloc[-1]:>6.1f}MB")
    if "cpu_delta_us" in df.columns:
        cpu_total_s = df["cpu_delta_us"].sum() / 1e6
        wall_s = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])
        print(f"  CPU total:       {cpu_total_s:.1f}s   wall (per-scan ts span): {wall_s:.1f}s   "
              f"util: {cpu_total_s/wall_s*100:.0f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--output", default=None, type=Path)
    args = p.parse_args()
    plot_diagnostics(args.csv, args.output or args.csv.parent)


if __name__ == "__main__":
    main()
