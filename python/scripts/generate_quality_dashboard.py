"""Generate static dashboard artifacts from objective WGAN experiment results."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _save_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df, x="profile", y="quality_score", palette="Set2")
    sns.stripplot(data=df, x="profile", y="quality_score", color="black", alpha=0.45, size=4)
    ax.set_title("Quality Score Distribution by WGAN Profile")
    ax.set_xlabel("Profile")
    ax.set_ylabel("Quality Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_timeline(df: pd.DataFrame, out_path: Path) -> None:
    timeline = df.copy()
    timeline["run_index"] = timeline.groupby("profile").cumcount() + 1
    plt.figure(figsize=(11, 6))
    for profile, group in timeline.groupby("profile"):
        group_sorted = group.sort_values("run_index")
        plt.plot(group_sorted["run_index"], group_sorted["quality_score"], marker="o", label=profile)
    plt.title("Quality Score Evolution Across Repetitions")
    plt.xlabel("Repetition")
    plt.ylabel("Quality Score")
    plt.legend(title="Profile")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_heatmap(summary_df: pd.DataFrame, out_path: Path) -> None:
    heatmap_cols = [
        "quality_score_mean",
        "final_coverage_mean",
        "final_robust_v2_score_mean",
        "avg_ks_mean",
        "avg_wasserstein_mean",
        "gen_convergence_score_mean",
        "critic_convergence_score_mean",
        "overfitting_rate_pct",
    ]
    available = [c for c in heatmap_cols if c in summary_df.columns]
    matrix = summary_df.set_index("profile")[available].copy()

    # Normalize per column for visual comparability.
    normalized = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-8)

    plt.figure(figsize=(12, 6))
    sns.heatmap(normalized, annot=matrix.round(3), fmt="", cmap="YlGnBu", cbar_kws={"label": "Normalized intensity"})
    plt.title("Profile vs Metric Heatmap (normalized with raw values annotated)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_overview_card(summary_df: pd.DataFrame, tests_df: pd.DataFrame, out_path: Path) -> dict:
    best = summary_df.sort_values("quality_score_mean", ascending=False).iloc[0]
    significant = tests_df[(tests_df["metric"] == "quality_score") & (tests_df["test"] == "mann_whitney_u") & (tests_df["significant"] == True)]

    lines = [
        "Objective WGAN Performance Summary",
        f"Generated at: {datetime.now().isoformat()}",
        "",
        f"Best profile by quality_score_mean: {best['profile']}",
        f"Quality score mean (95% CI): {best['quality_score_mean']:.3f} [{best['quality_score_ci95_low']:.3f}, {best['quality_score_ci95_high']:.3f}]",
        f"Overfitting rate: {best['overfitting_rate_pct']:.2f}%",
        "",
        f"Significant pairwise differences (quality_score, Mann-Whitney): {len(significant)}",
    ]

    if len(significant):
        for _, row in significant.iterrows():
            lines.append(
                f"- {row['profile_a']} vs {row['profile_b']}: p={row['p_value']:.4g}, Cliff's delta={row['effect_size_cliffs_delta']:.3f}"
            )

    payload = {
        "best_profile": str(best["profile"]),
        "best_quality_score_mean": float(best["quality_score_mean"]),
        "best_quality_score_ci95_low": float(best["quality_score_ci95_low"]),
        "best_quality_score_ci95_high": float(best["quality_score_ci95_high"]),
        "best_overfitting_rate_pct": float(best["overfitting_rate_pct"]),
        "significant_quality_pairwise_tests": int(len(significant)),
        "notes": lines,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static dashboard from objective WGAN outputs")
    parser.add_argument("--aggregate-dir", type=str, required=True,
                        help="Path to folder containing MASTER_COMPARISON.csv, PROFILE_SUMMARY.csv, PAIRWISE_STAT_TESTS.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output folder for dashboard files. Default: <aggregate-dir>/quality_dashboard")
    args = parser.parse_args()

    aggregate_dir = Path(args.aggregate_dir).resolve()
    master_csv = aggregate_dir / "MASTER_COMPARISON.csv"
    summary_csv = aggregate_dir / "PROFILE_SUMMARY.csv"
    tests_csv = aggregate_dir / "PAIRWISE_STAT_TESTS.csv"

    _ensure_file(master_csv)
    _ensure_file(summary_csv)
    _ensure_file(tests_csv)

    dashboard_dir = Path(args.output_dir).resolve() if args.output_dir else aggregate_dir / "quality_dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    master_df = pd.read_csv(master_csv)
    summary_df = pd.read_csv(summary_csv)
    tests_df = pd.read_csv(tests_csv)

    boxplot_path = dashboard_dir / "quality_score_boxplot.png"
    timeline_path = dashboard_dir / "quality_score_timeline.png"
    heatmap_path = dashboard_dir / "profile_metric_heatmap.png"

    _save_boxplot(master_df, boxplot_path)
    _save_timeline(master_df, timeline_path)
    _save_heatmap(summary_df, heatmap_path)

    # Export a compact table ready for thesis/report insertion.
    compact_cols = [
        "profile",
        "runs",
        "quality_score_mean",
        "quality_score_ci95_low",
        "quality_score_ci95_high",
        "final_coverage_mean",
        "final_robust_v2_score_mean",
        "overfitting_rate_pct",
    ]
    compact_cols = [c for c in compact_cols if c in summary_df.columns]
    compact = summary_df[compact_cols].sort_values(by="quality_score_mean", ascending=False)
    compact_csv = dashboard_dir / "ranking_table.csv"
    compact.to_csv(compact_csv, index=False)

    overview_json = dashboard_dir / "dashboard_summary.json"
    overview = _save_overview_card(summary_df, tests_df, overview_json)

    print("=" * 80)
    print("Static quality dashboard generated")
    print(f"Dashboard dir: {dashboard_dir}")
    print(f"Boxplot:       {boxplot_path}")
    print(f"Timeline:      {timeline_path}")
    print(f"Heatmap:       {heatmap_path}")
    print(f"Ranking table: {compact_csv}")
    print(f"Summary JSON:  {overview_json}")
    print(f"Best profile:  {overview['best_profile']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
