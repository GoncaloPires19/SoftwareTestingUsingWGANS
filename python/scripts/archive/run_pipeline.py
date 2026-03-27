"""
Unified pipeline runner for JUnit and SF110 datasets.

Examples:
  python python/scripts/run_pipeline.py --dataset junit --profile balanced --epochs 80
  python python/scripts/run_pipeline.py --dataset sf110 --profile balanced --epochs 80 --sf110-output-name sf110_full_v2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    """Run command with Windows-friendly fallback resolution for Maven."""
    try:
        subprocess.run(cmd, check=True)
        return
    except FileNotFoundError as exc:
        # On Windows, Maven is typically exposed as mvn.cmd / mvn.bat.
        if os.name == "nt" and cmd and cmd[0] == "mvn":
            for alt in ("mvn.cmd", "mvn.bat"):
                try:
                    subprocess.run([alt, *cmd[1:]], check=True)
                    return
                except FileNotFoundError:
                    continue
            raise RuntimeError(
                "Maven command was not found. Install Maven and ensure one of these is in PATH: "
                "mvn, mvn.cmd, mvn.bat"
            ) from exc
        raise


def _latest_run_dir(root: Path, run_name: str, profile: str) -> Path:
    pattern = f"{run_name}_{profile}_*"
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No run directory found for pattern: {pattern}")
    return matches[0]


def run_junit_pipeline(args) -> None:
    if args.run_java_tests:
        _run(["mvn", "clean", "test"])

    run_name = args.run_name or "pipeline_junit"
    train_cmd = [
        sys.executable,
        "python/scripts/train_basic_profiles.py",
        "--profile", args.profile,
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--device", args.device,
        "--data-path", args.junit_data_path,
        "--output-root", args.output_root,
        "--run-name", run_name,
        "--auto-generate-synthetic",
        "--synthetic-checkpoint", args.synthetic_checkpoint,
        "--synthetic-samples", str(args.synthetic_samples),
    ]
    _run(train_cmd)

    run_dir = _latest_run_dir(Path(args.output_root), run_name, args.profile)
    synthetic_name = "synthetic_samples_final.csv" if args.synthetic_checkpoint == "final" else "synthetic_samples_best.csv"

    if args.run_analysis:
        analysis_cmd = [
            sys.executable,
            "python/scripts/experimental_analysis.py",
            "--training-history", str(run_dir / "training_history.csv").replace("\\", "/"),
            "--real-data", args.junit_data_path,
            "--synthetic-data", str(run_dir / synthetic_name).replace("\\", "/"),
            "--output-dir", str(run_dir / "analysis").replace("\\", "/"),
        ]
        _run(analysis_cmd)

    print("=" * 70)
    print("✅ JUnit pipeline complete")
    print(f"   Run: {run_dir}")
    print("=" * 70)


def run_sf110_pipeline(args) -> None:
    if args.run_sf110_java_tests:
        java_test_cmd = [
            sys.executable,
            "python/scripts/run_sf110_java_tests.py",
            "--max-projects", str(args.sf110_java_max_projects),
            "--report-path", args.sf110_java_report,
            "--javac-source", args.sf110_java_javac_source,
            "--javac-target", args.sf110_java_javac_target,
            "--continue-on-error",
        ]
        if args.sf110_java_project_filter:
            java_test_cmd.extend(["--project-filter", args.sf110_java_project_filter])
        if args.sf110_java_target:
            java_test_cmd.extend(["--ant-target", args.sf110_java_target])
        _run(java_test_cmd)

    sf_cmd = [
        sys.executable,
        "python/scripts/run_sf110_hypotheses.py",
        "--profile", args.profile,
        "--epochs", str(args.epochs),
        "--device", args.device,
        "--output-name", args.sf110_output_name,
        "--hypotheses", "H1",
        "--validate",
        "--generate-synthetic",
        "--synthetic-samples", str(args.synthetic_samples),
        "--evaluate",
    ]

    if args.sf110_subset_projects is not None:
        sf_cmd.extend(["--subset-projects", str(args.sf110_subset_projects)])

    if args.sf110_skip_prepare:
        sf_cmd.append("--skip-prepare")

    _run(sf_cmd)

    print("=" * 70)
    print("✅ SF110 pipeline complete")
    print(f"   Summary: outputs/sf110_hypotheses/summaries/{args.sf110_output_name}_summary.csv")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete pipeline on JUnit or SF110")
    parser.add_argument("--dataset", choices=["junit", "sf110"], required=True)

    parser.add_argument("--profile", choices=["aggressive", "experimental", "conservative", "balanced"], default="balanced")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--synthetic-samples", type=int, default=1000)
    parser.add_argument("--synthetic-checkpoint", choices=["auto", "best", "final"], default="auto")

    parser.add_argument("--output-root", type=str, default="outputs/pipeline_runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-analysis", action="store_true")

    parser.add_argument("--run-java-tests", action="store_true", help="For junit mode: run 'mvn clean test' before training")
    parser.add_argument("--junit-data-path", type=str, default="python/data/junit_data_cleaned_strict.csv")

    parser.add_argument("--sf110-output-name", type=str, default="sf110_full_v2")
    parser.add_argument("--sf110-subset-projects", type=int, default=0)
    parser.add_argument("--sf110-skip-prepare", action="store_true")
    parser.add_argument("--run-sf110-java-tests", action="store_true", help="For sf110 mode: run Java tests/build across SF110 projects before training")
    parser.add_argument("--sf110-java-max-projects", type=int, default=20, help="For sf110 Java tests: maximum projects to execute (0=all)")
    parser.add_argument("--sf110-java-project-filter", type=str, default=None, help="For sf110 Java tests: include only projects containing this text")
    parser.add_argument("--sf110-java-target", type=str, default=None, help="For sf110 Java tests: force Ant target")
    parser.add_argument("--sf110-java-javac-source", type=str, default="1.8", help="For sf110 Java tests: override compile.source")
    parser.add_argument("--sf110-java-javac-target", type=str, default="1.8", help="For sf110 Java tests: override compile.target")
    parser.add_argument("--sf110-java-report", type=str, default="outputs/sf110_hypotheses/summaries/sf110_java_test_report.json")

    args = parser.parse_args()

    if args.dataset == "junit":
        run_junit_pipeline(args)
    else:
        run_sf110_pipeline(args)


if __name__ == "__main__":
    main()
