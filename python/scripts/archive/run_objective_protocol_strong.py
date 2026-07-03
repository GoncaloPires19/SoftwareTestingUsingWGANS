#!/usr/bin/env python3
"""Run a thesis-grade objective protocol: multi-seed WGAN comparison + Gaussian baseline + dashboard.

This script orchestrates existing project tools so results remain reproducible and minimal.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def latest_experiment_dir(exp_root: Path, prefix: str) -> Path:
    candidates = [p for p in exp_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No experiment directory found in {exp_root} for prefix '{prefix}'")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def latest_run_dir(outputs_root: Path, run_prefix: str, profile: str) -> Path:
    pattern = f"{run_prefix}_{profile}_r01_{profile}_*"
    candidates = list(outputs_root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No run folder found for pattern: {pattern}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strong objective WGAN protocol")
    parser.add_argument("--profiles", nargs="+", default=["balanced", "optimized", "experimental"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 999])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--data-path", type=str, default="python/data/junit_data_cleaned_strict.csv")
    parser.add_argument("--real-data", type=str, default="python/data/junit_data_cleaned_strict.csv")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default="objective_wgan_strong")
    parser.add_argument("--baseline-samples", type=int, default=1000)
    parser.add_argument("--baseline-seed", type=int, default=42)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # 1) Multi-seed WGAN replication + stats
    replicate_cmd = [
        py,
        "python/scripts/replicate_and_compare.py",
        "--profiles",
        *args.profiles,
        "--seeds",
        *[str(s) for s in args.seeds],
        "--device",
        args.device,
        "--data-path",
        args.data_path,
        "--real-data",
        args.real_data,
        "--output-root",
        args.output_root,
        "--experiment-name",
        args.experiment_name,
    ]
    if args.skip_training:
        replicate_cmd.append("--skip-training")

    run(replicate_cmd, cwd=repo_root)

    # 2) Resolve aggregate folder produced by replicate_and_compare
    aggregate_root = output_root / "experimental"
    aggregate_dir = latest_experiment_dir(aggregate_root, f"{args.experiment_name}_")

    # 3) Baseline generation (Gaussian)
    baseline_csv = aggregate_dir / "baseline_gaussian.csv"
    run(
        [
            py,
            "python/scripts/generate_baseline_normal.py",
            "--real-data",
            args.real_data,
            "--samples",
            str(args.baseline_samples),
            "--seed",
            str(args.baseline_seed),
            "--output",
            str(baseline_csv).replace("\\", "/"),
        ],
        cwd=repo_root,
    )

    # 4) Baseline objective analysis against real data
    baseline_analysis_dir = aggregate_dir / "baseline_analysis"
    baseline_analysis_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            py,
            "python/scripts/experimental_analysis.py",
            "--training-history",
            str(latest_run_dir(output_root, args.experiment_name, args.profiles[0]) / "training_history.csv").replace("\\", "/"),
            "--real-data",
            args.real_data,
            "--synthetic-data",
            str(baseline_csv).replace("\\", "/"),
            "--output-dir",
            str(baseline_analysis_dir).replace("\\", "/"),
        ],
        cwd=repo_root,
    )

    # 5) Dashboard artifacts from aggregate outputs
    run(
        [
            py,
            "python/scripts/generate_quality_dashboard.py",
            "--aggregate-dir",
            str(aggregate_dir).replace("\\", "/"),
        ],
        cwd=repo_root,
    )

    print("=" * 80)
    print("Strong objective protocol complete")
    print(f"Aggregate dir: {aggregate_dir}")
    print(f"Baseline CSV:  {baseline_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
