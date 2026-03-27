"""Run repeated WGAN profile experiments and compute objective statistics.

This script automates:
1) repeated training runs per profile,
2) per-run experimental analysis,
3) aggregate summary with confidence intervals,
4) pairwise statistical tests across profiles.
"""

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def _ci95(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "cv": math.nan}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sem = std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    margin = 1.96 * sem
    cv = float(std / mean) if mean != 0 else math.nan
    return {
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - margin),
        "ci95_high": float(mean + margin),
        "cv": cv,
    }


def _safe_float(value, default=math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _find_latest_run_dir(output_root: Path, prefix: str) -> Path:
    matches = [p for p in output_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not matches:
        raise FileNotFoundError(f"No run directory found for prefix '{prefix}' in {output_root}")
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _call_python(command: List[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def _extract_run_metrics(report_path: Path, history_path: Path, profile: str, repetition: int, seed: int, run_dir: Path) -> Dict[str, float]:
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    history = pd.read_csv(history_path)

    quality_score = _safe_float(report.get("quality_score"))
    overfitting = bool(report.get("overfitting_detection", {}).get("detected_overfitting", False))

    metrics = report.get("training_analysis", {}).get("metrics", {})
    gen_convergence = _safe_float(metrics.get("gen_loss", {}).get("convergence_score"))
    critic_convergence = _safe_float(metrics.get("critic_loss", {}).get("convergence_score"))

    feature_stats = report.get("distribution_comparison", {}).get("features", {})
    ks_values = []
    wasserstein_values = []
    coverage_values = []
    for feat_data in feature_stats.values():
        ks_values.append(_safe_float(feat_data.get("ks_statistic")))
        wasserstein_values.append(_safe_float(feat_data.get("wasserstein_distance")))
        coverage_values.append(_safe_float(feat_data.get("coverage_pct")))

    avg_ks = float(np.nanmean(ks_values)) if ks_values else math.nan
    avg_wasserstein = float(np.nanmean(wasserstein_values)) if wasserstein_values else math.nan
    avg_coverage = float(np.nanmean(coverage_values)) if coverage_values else math.nan

    final_coverage = _safe_float(history["coverage"].iloc[-1]) if "coverage" in history.columns and len(history) else math.nan
    final_coverage_p95 = _safe_float(history["coverage_p95"].iloc[-1]) if "coverage_p95" in history.columns and len(history) else math.nan
    final_coverage_p99 = _safe_float(history["coverage_p99"].iloc[-1]) if "coverage_p99" in history.columns and len(history) else math.nan
    final_robust_v2 = _safe_float(history["robust_v2_score"].iloc[-1]) if "robust_v2_score" in history.columns and len(history) else math.nan

    return {
        "profile": profile,
        "repetition": repetition,
        "seed": seed,
        "run_dir": str(run_dir).replace("\\", "/"),
        "quality_score": quality_score,
        "avg_ks": avg_ks,
        "avg_wasserstein": avg_wasserstein,
        "avg_feature_coverage": avg_coverage,
        "final_coverage": final_coverage,
        "final_coverage_p95": final_coverage_p95,
        "final_coverage_p99": final_coverage_p99,
        "final_robust_v2_score": final_robust_v2,
        "gen_convergence_score": gen_convergence,
        "critic_convergence_score": critic_convergence,
        "overfitting_detected": overfitting,
    }


def _pairwise_tests(df: pd.DataFrame, metric: str, alpha: float = 0.05) -> List[Dict[str, float]]:
    rows = []
    profiles = sorted(df["profile"].unique())
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            p1 = profiles[i]
            p2 = profiles[j]
            a = df[df["profile"] == p1][metric].dropna().to_numpy()
            b = df[df["profile"] == p2][metric].dropna().to_numpy()
            if len(a) < 2 or len(b) < 2:
                rows.append({
                    "profile_a": p1,
                    "profile_b": p2,
                    "metric": metric,
                    "test": "insufficient_data",
                    "statistic": math.nan,
                    "p_value": math.nan,
                    "significant": False,
                    "effect_size_cliffs_delta": math.nan,
                })
                continue

            u_stat, mw_p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            ks_stat, ks_p = scipy_stats.ks_2samp(a, b)

            # Cliff's delta for non-parametric effect size.
            gt = sum(x > y for x in a for y in b)
            lt = sum(x < y for x in a for y in b)
            cliffs_delta = (gt - lt) / (len(a) * len(b))

            rows.append({
                "profile_a": p1,
                "profile_b": p2,
                "metric": metric,
                "test": "mann_whitney_u",
                "statistic": float(u_stat),
                "p_value": float(mw_p),
                "significant": bool(mw_p < alpha),
                "effect_size_cliffs_delta": float(cliffs_delta),
            })
            rows.append({
                "profile_a": p1,
                "profile_b": p2,
                "metric": metric,
                "test": "ks_2samp",
                "statistic": float(ks_stat),
                "p_value": float(ks_p),
                "significant": bool(ks_p < alpha),
                "effect_size_cliffs_delta": float(cliffs_delta),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated WGAN profile experiments and compare results objectively")
    parser.add_argument("--profiles", nargs="+", default=["balanced", "optimized", "experimental"],
                        help="Profiles from train_basic_profiles.py")
    parser.add_argument("--runs", type=int, default=10, help="Number of repetitions per profile")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed; each repetition uses base_seed + repetition")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Explicit seed list (overrides --runs/--base-seed), e.g. --seeds 42 123 456 789 999")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--data-path", type=str, default="python/data/junit_data_cleaned_strict.csv")
    parser.add_argument("--norm-stats-path", type=str, default=None)
    parser.add_argument("--real-data", type=str, default="python/data/junit_data_cleaned_strict.csv")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--best-metric", choices=["robust", "robust_v2", "coverage"], default="robust_v2")
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--skip-training", action="store_true", help="Only aggregate existing results")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Optional name for this batch, used for result folder naming")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.experiment_name or "objective_wgan"
    aggregate_dir = output_root / "experimental" / f"{exp_name}_{stamp}"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, float]] = []

    if args.seeds:
        run_plan = [(idx + 1, seed) for idx, seed in enumerate(args.seeds)]
    else:
        run_plan = [(rep, args.base_seed + rep) for rep in range(1, args.runs + 1)]

    for profile in args.profiles:
        for rep, run_seed in run_plan:
            run_prefix = f"{exp_name}_{profile}_r{rep:02d}"

            if not args.skip_training:
                train_cmd = [
                    sys.executable,
                    "python/scripts/train_basic_profiles.py",
                    "--profile", profile,
                    "--seed", str(run_seed),
                    "--device", args.device,
                    "--data-path", args.data_path,
                    "--output-root", args.output_root,
                    "--run-name", run_prefix,
                    "--auto-generate-synthetic",
                    "--synthetic-checkpoint", "auto",
                    "--synthetic-output-name", "synthetic_samples_best.csv",
                    "--eval-samples", str(args.eval_samples),
                    "--eval-seed", str(args.eval_seed),
                    "--best-metric", args.best_metric,
                    "--warmup-epochs", str(args.warmup_epochs),
                ]
                if args.epochs is not None:
                    train_cmd.extend(["--epochs", str(args.epochs)])
                if args.norm_stats_path:
                    train_cmd.extend(["--norm-stats-path", args.norm_stats_path])
                _call_python(train_cmd, cwd=repo_root)

            run_dir = _find_latest_run_dir(output_root, run_prefix)
            history_path = run_dir / "training_history.csv"
            synthetic_path = run_dir / "synthetic_samples_best.csv"
            if not synthetic_path.exists():
                # Fallback when user chose final checkpoint output naming.
                synthetic_path = run_dir / "synthetic_samples_final.csv"

            analysis_dir = aggregate_dir / "per_run_analysis" / run_dir.name
            analysis_dir.mkdir(parents=True, exist_ok=True)

            analysis_cmd = [
                sys.executable,
                "python/scripts/experimental_analysis.py",
                "--training-history", str(history_path).replace("\\", "/"),
                "--real-data", args.real_data,
                "--synthetic-data", str(synthetic_path).replace("\\", "/"),
                "--output-dir", str(analysis_dir).replace("\\", "/"),
            ]
            _call_python(analysis_cmd, cwd=repo_root)

            report_path = analysis_dir / "experimental_analysis_report.json"
            row = _extract_run_metrics(report_path, history_path, profile, rep, run_seed, run_dir)
            all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No runs found/processed. Nothing to aggregate.")

    df = pd.DataFrame(all_rows)
    master_csv = aggregate_dir / "MASTER_COMPARISON.csv"
    df.to_csv(master_csv, index=False)

    profile_summary_rows = []
    tracked_metrics = [
        "quality_score",
        "avg_ks",
        "avg_wasserstein",
        "avg_feature_coverage",
        "final_coverage",
        "final_coverage_p95",
        "final_coverage_p99",
        "final_robust_v2_score",
        "gen_convergence_score",
        "critic_convergence_score",
    ]

    for profile, group in df.groupby("profile"):
        row = {"profile": profile, "runs": int(len(group)), "overfitting_rate_pct": float(group["overfitting_detected"].mean() * 100)}
        for metric in tracked_metrics:
            ci = _ci95(group[metric].dropna().tolist())
            for key, val in ci.items():
                row[f"{metric}_{key}"] = val
        profile_summary_rows.append(row)

    summary_df = pd.DataFrame(profile_summary_rows).sort_values(by="quality_score_mean", ascending=False)
    summary_csv = aggregate_dir / "PROFILE_SUMMARY.csv"
    summary_df.to_csv(summary_csv, index=False)

    tests = []
    for metric in ["quality_score", "final_coverage", "final_robust_v2_score", "avg_ks", "avg_wasserstein"]:
        tests.extend(_pairwise_tests(df, metric=metric))
    tests_df = pd.DataFrame(tests)
    tests_csv = aggregate_dir / "PAIRWISE_STAT_TESTS.csv"
    tests_df.to_csv(tests_csv, index=False)

    result = {
        "created_at": datetime.now().isoformat(),
        "experiment": {
            "name": exp_name,
            "profiles": args.profiles,
            "runs_per_profile": len(run_plan),
            "base_seed": args.base_seed,
            "seeds": [seed for _, seed in run_plan],
            "device": args.device,
            "epochs": args.epochs,
        },
        "files": {
            "master_comparison_csv": str(master_csv).replace("\\", "/"),
            "profile_summary_csv": str(summary_csv).replace("\\", "/"),
            "pairwise_stat_tests_csv": str(tests_csv).replace("\\", "/"),
        },
        "ranking": summary_df[["profile", "quality_score_mean", "quality_score_ci95_low", "quality_score_ci95_high", "overfitting_rate_pct"]].to_dict(orient="records"),
    }

    result_json = aggregate_dir / "summary.json"
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=" * 80)
    print("Objective experiment completed")
    print(f"Aggregate folder: {aggregate_dir}")
    print(f"MASTER_COMPARISON: {master_csv}")
    print(f"PROFILE_SUMMARY:   {summary_csv}")
    print(f"STAT_TESTS:        {tests_csv}")
    print(f"SUMMARY_JSON:      {result_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
