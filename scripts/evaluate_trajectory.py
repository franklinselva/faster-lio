#!/usr/bin/env python3
"""Evaluate and visualize LIO trajectory against ground truth using evo.

Reads TUM-format trajectory files and runs evo_ape / evo_rpe for metrics and plots.
"""

import argparse
import subprocess
import sys


def run_cmd(cmd: list[str], desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectory with evo")
    parser.add_argument("--estimated", required=True, help="Estimated trajectory (TUM format)")
    parser.add_argument("--ground_truth", required=True, help="Ground truth trajectory (TUM format)")
    parser.add_argument("--plot", action="store_true", help="Show plots")
    parser.add_argument("--save_plot", default=None, help="Save plot to file")
    parser.add_argument("--time_offset", type=float, default=0.0,
                        help="Time offset (seconds) to add to estimated timestamps")
    parser.add_argument("--eval_start", type=float, default=0.0,
                        help="Skip poses before this many seconds from trajectory start")
    args = parser.parse_args()

    # Apply time offset and/or trim to adjusted file
    estimated = args.estimated
    if args.time_offset != 0.0 or args.eval_start > 0.0:
        from pathlib import Path
        adjusted = Path(args.estimated).with_suffix(".adjusted.txt")
        first_ts = None
        with open(args.estimated) as fin, open(adjusted, "w") as fout:
            for line in fin:
                if line.startswith("#"):
                    fout.write(line)
                    continue
                parts = line.strip().split()
                if len(parts) >= 8:
                    orig_ts = float(parts[0])
                    if first_ts is None:
                        first_ts = orig_ts
                    if orig_ts < first_ts + args.eval_start:
                        continue
                    ts = orig_ts + args.time_offset
                    fout.write(f"{ts:.6f} {' '.join(parts[1:])}\n")
        estimated = str(adjusted)
        info = []
        if args.time_offset != 0.0:
            info.append(f"time_offset={args.time_offset:.3f}s")
        if args.eval_start > 0.0:
            info.append(f"eval_start={args.eval_start:.1f}s")
        print(f"Adjusted trajectory ({', '.join(info)}) -> {estimated}")

    # ATE (Absolute Trajectory Error)
    ate_cmd = ["evo_ape", "tum", args.ground_truth, estimated, "-va", "--align"]
    if args.plot:
        ate_cmd.append("--plot")
    if args.save_plot:
        ate_cmd.extend(["--save_plot", f"{args.save_plot}_ate.pdf"])
    rc = run_cmd(ate_cmd, "Absolute Trajectory Error (ATE)")
    if rc != 0:
        print("evo_ape failed", file=sys.stderr)

    # RPE (Relative Pose Error)
    rpe_cmd = ["evo_rpe", "tum", args.ground_truth, estimated, "-va", "--align",
               "--delta", "1", "--delta_unit", "m"]
    if args.plot:
        rpe_cmd.append("--plot")
    if args.save_plot:
        rpe_cmd.extend(["--save_plot", f"{args.save_plot}_rpe.pdf"])
    rc = run_cmd(rpe_cmd, "Relative Pose Error (RPE)")
    if rc != 0:
        print("evo_rpe failed", file=sys.stderr)

    # Trajectory plot
    if args.plot or args.save_plot:
        traj_cmd = ["evo_traj", "tum", estimated, args.ground_truth,
                     "--align", "--ref", args.ground_truth]
        if args.plot:
            traj_cmd.append("--plot")
        if args.save_plot:
            traj_cmd.extend(["--save_plot", f"{args.save_plot}_traj.pdf"])
        run_cmd(traj_cmd, "Trajectory Comparison")


if __name__ == "__main__":
    main()
