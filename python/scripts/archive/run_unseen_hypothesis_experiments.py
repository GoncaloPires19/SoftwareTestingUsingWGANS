"""
Run unseen-data generalization experiments for SF110 hypotheses.

For each selected hypothesis (H1/H2/H3), this runner:
1) loads the prepared strict dataset,
2) splits it into train/test (holdout never seen during training),
3) trains WGAN on train split only,
4) generates synthetic samples,
5) evaluates synthetic vs holdout test split using experimental_analysis.py,
6) saves a consolidated summary CSV/JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class ExperimentPaths:
    base_dir: Path
    data_dir: Path
    runs_dir: Path
    analysis_dir: Path
    summary_csv: Path
    summary_json: Path


def _load_strict_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path, header=None, names=["methods", "branches", "statements"])
    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")
    return df


def _split_train_test(df: pd.DataFrame, test_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_size = max(1, int(round(len(shuffled) * test_ratio)))
    if test_size >= len(shuffled):
        test_size = len(shuffled) - 1

    test_df = shuffled.iloc[:test_size].copy()
    train_df = shuffled.iloc[test_size:].copy()

    if train_df.empty or test_df.empty:
        raise RuntimeError("Invalid split generated empty train or test set")

    return train_df, test_df


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _latest_run_dir(runs_root: Path, run_name: str, profile: str) -> Path:
    pattern = f"{run_name}_{profile}_*"
    matches = sorted(runs_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No run directory found for pattern: {pattern}")
    return matches[0]


def _resolve_synthetic_csv(run_dir: Path) -> Path:
    best_csv = run_dir / "synthetic_samples_best.csv"
    final_csv = run_dir / "synthetic_samples_final.csv"
    if best_csv.exists():
        return best_csv
    if final_csv.exists():
        return final_csv
    raise FileNotFoundError(f"Synthetic CSV not found in {run_dir}")


def _load_experimental_report(report_path: Path) -> dict:
    if not report_path.exists():
        raise FileNotFoundError(f"Missing experimental analysis report: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _profile_for_hypothesis(hypothesis: str) -> str:
    # Keep consistency with your previous baseline settings.
    if hypothesis == "H1":
        return "generalization-stress"
    return "optimized"


def _dataset_path_for_hypothesis(output_name: str, hypothesis: str) -> Path:
    root = Path("python/data/processed")
    mapping = {
        "H1": root / f"{output_name}_H1_sf110_only_strict.csv",
        "H2": root / f"{output_name}_H2_sf110_plus_junit_strict.csv",
        "H3": root / f"{output_name}_H3_simulated_combined_strict.csv",
    }
    return mapping[hypothesis]


def _build_paths(base_output_dir: Path, stamp: str) -> ExperimentPaths:
    base_dir = base_output_dir / f"unseen_eval_{stamp}"
    data_dir = base_dir / "data"
    runs_dir = base_dir / "runs"
    analysis_dir = base_dir / "analysis"

    data_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    return ExperimentPaths(
        base_dir=base_dir,
        data_dir=data_dir,
        runs_dir=runs_dir,
        analysis_dir=analysis_dir,
        summary_csv=base_dir / "unseen_eval_summary.csv",
        summary_json=base_dir / "unseen_eval_summary.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unseen-data WGAN experiments for H1/H2/H3")
    parser.add_argument("--output-name", type=str, default="sf110_full_v2",
                        help="Prepared dataset prefix (e.g., sf110_full_v2)")
    parser.add_argument("--hypotheses", nargs="+", default=["H1", "H2", "H3"], choices=["H1", "H2", "H3"],
                        help="Hypotheses to evaluate on unseen holdout")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Holdout ratio for unseen test split")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split and training")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Training epochs per hypothesis")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Training device")
    parser.add_argument("--eval-samples", type=int, default=1000,
                        help="Deterministic evaluation samples per epoch")
    parser.add_argument("--warmup-epochs", type=int, default=10,
                        help="Best-checkpoint warmup epochs")
    parser.add_argument("--batch-log-interval", type=int, default=20,
                        help="Batch-level progress heartbeat interval")
    parser.add_argument("--base-output-dir", type=str, default="outputs/sf110_hypotheses",
                        help="Where to store unseen-eval experiment artifacts")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = _build_paths(Path(args.base_output_dir), stamp)

    summary_rows = []

    for hypothesis in args.hypotheses:
        profile = _profile_for_hypothesis(hypothesis)
        source_csv = _dataset_path_for_hypothesis(args.output_name, hypothesis)

        df = _load_strict_dataset(source_csv)
        train_df, test_df = _split_train_test(df, args.test_ratio, args.seed)

        hyp_data_dir = paths.data_dir / hypothesis
        hyp_data_dir.mkdir(parents=True, exist_ok=True)
        train_csv = hyp_data_dir / "train_unseen_split.csv"
        test_csv = hyp_data_dir / "test_unseen_holdout.csv"
        train_df.to_csv(train_csv, index=False, header=False)
        test_df.to_csv(test_csv, index=False, header=False)

        run_name = f"unseen_{hypothesis.lower()}"
        train_cmd = [
            sys.executable,
            "python/scripts/train_basic_profiles.py",
            "--profile", profile,
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
            "--device", args.device,
            "--data-path", str(train_csv).replace("\\", "/"),
            "--output-root", str(paths.runs_dir).replace("\\", "/"),
            "--run-name", run_name,
            "--auto-generate-synthetic",
            "--synthetic-checkpoint", "auto",
            "--synthetic-samples", str(len(test_df)),
            "--eval-samples", str(args.eval_samples),
            "--warmup-epochs", str(args.warmup_epochs),
            "--batch-log-interval", str(args.batch_log_interval),
        ]
        _run(train_cmd)

        run_dir = _latest_run_dir(paths.runs_dir, run_name, profile)
        synthetic_csv = _resolve_synthetic_csv(run_dir)

        hyp_analysis_dir = paths.analysis_dir / hypothesis
        hyp_analysis_dir.mkdir(parents=True, exist_ok=True)
        analysis_cmd = [
            sys.executable,
            "python/scripts/experimental_analysis.py",
            "--training-history", str((run_dir / "training_history.csv")).replace("\\", "/"),
            "--real-data", str(test_csv).replace("\\", "/"),
            "--synthetic-data", str(synthetic_csv).replace("\\", "/"),
            "--output-dir", str(hyp_analysis_dir).replace("\\", "/"),
        ]
        _run(analysis_cmd)

        report = _load_experimental_report(hyp_analysis_dir / "experimental_analysis_report.json")
        feature_metrics = report.get("distribution_comparison", {}).get("features", {})
        overfit = report.get("overfitting_detection", {}).get("detected_overfitting", False)

        row = {
            "hypothesis": hypothesis,
            "profile": profile,
            "source_dataset": str(source_csv).replace("\\", "/"),
            "train_samples": len(train_df),
            "test_samples_unseen": len(test_df),
            "run_dir": str(run_dir).replace("\\", "/"),
            "quality_score": report.get("quality_score"),
            "overfitting_detected": overfit,
        }

        for feature in ["methods", "branches", "statements"]:
            fm = feature_metrics.get(feature, {})
            row[f"{feature}_ks_stat"] = fm.get("ks_statistic")
            row[f"{feature}_wasserstein"] = fm.get("wasserstein_distance")
            row[f"{feature}_coverage_pct"] = fm.get("coverage_pct")
            row[f"{feature}_mean_error_pct"] = fm.get("mean_error_pct")

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(paths.summary_csv, index=False)

    with open(paths.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print("=" * 90)
    print("DONE: Unseen-data experiments completed")
    print(f"Results folder: {paths.base_dir}")
    print(f"Summary CSV:    {paths.summary_csv}")
    print(f"Summary JSON:   {paths.summary_json}")
    print("=" * 90)


if __name__ == "__main__":
    main()
