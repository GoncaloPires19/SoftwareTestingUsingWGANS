"""
Run SF110 hypothesis experiments:
- H1: SF110 only
- H2: SF110 + local JUnit cleaned data
- H3: simulated combined (same as H2 by user decision)

This runner can:
1) prepare SF110 automatically from EvoSuite SF110 page
2) build strict training CSVs for each hypothesis
3) run train_basic_profiles.py with configurable profile/epochs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from ast import literal_eval
from shutil import copyfile

import numpy as np
import pandas as pd

from prepare_sf110_dataset import prepare_sf110_dataset, SF110_URL

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_validation import DataValidator
from evaluation import (
    CoverageExtractor,
    DatasetComparator,
    DiversityAnalyzer,
    EvaluationReport,
    QualityMetrics,
)


def _ensure_local_junit_strict(max_value: int = 1000) -> Path:
    data_dir = Path("python/data")
    candidates = [
        data_dir / "junit_data_cleaned_strict.csv",
        data_dir / "junit_data_cleaned_v2.csv",
        data_dir / "junit_data_cleaned.csv",
        data_dir / "archive" / "junit_data_cleaned_strict.csv",
        data_dir / "archive" / "junit_data_cleaned.csv",
    ]

    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(
            "Local JUnit dataset not found. Tried: " + ", ".join(str(p) for p in candidates)
        )

    # Prefer the first available candidate and normalize into strict output path.
    strict_path = data_dir / "junit_data_cleaned_strict.csv"
    fallback = existing[0]
    if fallback == strict_path and strict_path.exists():
        return strict_path

    df = pd.read_csv(fallback, header=None, names=["methods", "branches", "statements"])
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()
    for c in ["methods", "branches", "statements"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df = df[(df[["methods", "branches", "statements"]] >= 0).all(axis=1)]
    df = df[(df[["methods", "branches", "statements"]] <= max_value).all(axis=1)]
    df = df.drop_duplicates().reset_index(drop=True)

    strict_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(strict_path, index=False, header=False)
    return strict_path


def _merge_strict_csvs(input_csvs: list[Path], output_csv: Path, max_value: int = 1000) -> Path:
    frames = []
    for csv_path in input_csvs:
        df = pd.read_csv(csv_path, header=None, names=["methods", "branches", "statements"])
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.replace([float("inf"), float("-inf")], pd.NA).dropna()
    for c in ["methods", "branches", "statements"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged = merged.dropna()
    merged = merged[(merged[["methods", "branches", "statements"]] >= 0).all(axis=1)]
    merged = merged[(merged[["methods", "branches", "statements"]] <= max_value).all(axis=1)]
    merged = merged.drop_duplicates().reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False, header=False)
    return output_csv


def _sanitize_strict_csv(
    input_csv: Path,
    output_csv: Path,
    max_value: int = 1000,
    deduplicate: bool = False,
) -> Path:
    """Normalize one strict CSV without forcing dedup by default."""
    df = pd.read_csv(input_csv, header=None, names=["methods", "branches", "statements"])
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()
    for c in ["methods", "branches", "statements"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df = df[(df[["methods", "branches", "statements"]] >= 0).all(axis=1)]
    df = df[(df[["methods", "branches", "statements"]] <= max_value).all(axis=1)]
    if deduplicate:
        df = df.drop_duplicates().reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, header=False)
    return output_csv


def _build_synthetic_dataset_from_source(
    source_csv: Path,
    output_csv: Path,
    random_seed: int = 42,
    target_size: int | None = None,
    jitter_scale: float = 0.05,
    max_value: int = 1000,
) -> Path:
    """Create a synthetic-only dataset by bootstrap + noise from a source dataset."""
    rng = np.random.default_rng(random_seed)
    source_df = pd.read_csv(source_csv, header=None, names=["methods", "branches", "statements"])
    if target_size is None or target_size <= 0:
        target_size = len(source_df)

    sampled = source_df.sample(n=target_size, replace=True, random_state=random_seed).reset_index(drop=True)
    std = sampled.std().replace(0.0, 1.0)

    for col in ["methods", "branches", "statements"]:
        noise = rng.normal(0.0, float(std[col]) * jitter_scale, size=target_size)
        sampled[col] = sampled[col].astype(float) + noise

    sampled = sampled.clip(lower=0.0, upper=float(max_value)).round(3)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_csv, index=False, header=False)
    return output_csv


def _run_training(
    profile: str,
    epochs: int,
    dataset_path: Path,
    run_name: str,
    device: str,
    output_root: Path,
    best_metric: str,
    warmup_epochs: int,
    eval_seed: int,
    eval_samples: int,
    batch_log_interval: int,
) -> None:
    cmd = [
        sys.executable,
        "python/scripts/train_basic_profiles.py",
        "--profile", profile,
        "--epochs", str(epochs),
        "--device", device,
        "--data-path", str(dataset_path).replace("\\", "/"),
        "--output-root", str(output_root).replace("\\", "/"),
        "--run-name", run_name,
        "--best-metric", best_metric,
        "--warmup-epochs", str(warmup_epochs),
        "--eval-seed", str(eval_seed),
        "--eval-samples", str(eval_samples),
        "--batch-log-interval", str(batch_log_interval),
    ]
    subprocess.run(cmd, check=True)


def _validate_dataset(csv_path: Path, validation_dir: Path) -> Path:
    validator = DataValidator(validation_log_dir=validation_dir)
    is_valid, report_path = validator.validate_and_report(csv_path)
    if not is_valid:
        raise RuntimeError(f"Dataset validation failed for: {csv_path}. Report: {report_path}")
    return report_path


def _generate_synthetic_for_run(
    run_dir: Path,
    norm_stats_path: Path,
    num_samples: int,
    device: str,
    best_metric: str,
) -> Path:
    if best_metric == "robust_v2":
        best_model = run_dir / "models" / "generator_best_by_robust_v2.pth"
    elif best_metric == "robust":
        best_model = run_dir / "models" / "generator_best_by_robust.pth"
    else:
        best_model = run_dir / "models" / "generator_best_by_coverage.pth"

    model_path = best_model if best_model.exists() else (run_dir / "models" / "generator_final.pth")
    output_path = run_dir / "synthetic_samples.csv"

    cmd = [
        sys.executable,
        "python/scripts/generate_synthetic.py",
        "--model", str(model_path).replace("\\", "/"),
        "--num-samples", str(num_samples),
        "--output", str(output_path).replace("\\", "/"),
        "--norm-stats", str(norm_stats_path).replace("\\", "/"),
        "--device", device,
    ]
    subprocess.run(cmd, check=True)
    return output_path


def _evaluate_run(
    hypothesis: str,
    output_name: str,
    dataset_path: Path,
    synthetic_csv: Path,
    evaluation_dir: Path,
) -> Path:
    real_df = pd.read_csv(dataset_path, header=None, names=["methods", "branches", "statements"])
    synthetic_df = pd.read_csv(synthetic_csv)
    synthetic_features = synthetic_df[["methods", "branches", "statements"]]

    dataset_comparison = DatasetComparator.compare_datasets(real_df, synthetic_features)
    diversity = DiversityAnalyzer.analyze_dataset(synthetic_features)
    coverage = CoverageExtractor().extract_from_surefire()

    quality = QualityMetrics(
        synthetic_quality_score=0.75,
        mutation_score=0.0,
        assertion_rate=0.0,
        test_complexity=float(synthetic_features.mean().mean()),
        method_mean=float(synthetic_features["methods"].mean()),
        branch_mean=float(synthetic_features["branches"].mean()),
        statement_mean=float(synthetic_features["statements"].mean()),
    )

    report_gen = EvaluationReport(output_dir=str(evaluation_dir))
    report = report_gen.generate_comparison_report(
        name=f"{output_name}_{hypothesis}",
        coverage=coverage,
        diversity=diversity,
        quality=quality,
        dataset_comparison=dataset_comparison,
    )

    report_path = report_gen.save_report(report, filename=f"evaluation_{output_name}_{hypothesis}.json")
    return report_path


def _latest_run_dir(output_root: Path, run_name: str, profile: str) -> Path:
    pattern = f"{run_name}_{profile}_*"
    matches = sorted(output_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No output folder found for pattern: {pattern}")
    return matches[0]


def _collect_run_metrics(run_dir: Path) -> dict:
    history_path = run_dir / "training_history.csv"
    config_path = run_dir / "config.json"

    if not history_path.exists():
        raise FileNotFoundError(f"Missing training history: {history_path}")

    history_df = pd.read_csv(history_path)
    final_row = history_df.iloc[-1]
    best_idx = history_df["coverage"].idxmax()
    best_row = history_df.iloc[best_idx]
    selected_row = best_row
    best_epoch = int(best_row["epoch"])
    best_coverage = float(best_row["coverage"])
    selected_metric = "coverage"
    selected_metric_score = float(best_row["coverage"])

    result = {
        "run_dir": str(run_dir).replace("\\", "/"),
        "epochs": int(len(history_df)),
        "best_epoch": best_epoch,
        "best_coverage": best_coverage,
        "final_coverage": float(final_row["coverage"]),
        "final_diversity": float(final_row["diversity"]),
        "final_max_methods": float(final_row["max_methods"]),
        "final_max_branches": float(final_row["max_branches"]),
        "final_max_statements": float(final_row["max_statements"]),
    }

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        result["dataset_path"] = cfg.get("dataset_path")
        result["norm_stats_path"] = cfg.get("norm_stats_path")
        result["best_metric"] = cfg.get("best_metric")
        result["warmup_epochs"] = cfg.get("warmup_epochs")
        result["best_by_coverage"] = cfg.get("best_by_coverage")
        result["best_by_robust"] = cfg.get("best_by_robust")
        result["best_by_robust_v2"] = cfg.get("best_by_robust_v2")

        def _parse_best_meta(meta_value):
            if isinstance(meta_value, str):
                try:
                    meta_value = literal_eval(meta_value)
                except (ValueError, SyntaxError):
                    return None
            return meta_value if isinstance(meta_value, dict) else None

        selected_metric = str(cfg.get("best_metric") or "coverage")
        metric_to_key = {
            "coverage": "best_by_coverage",
            "robust": "best_by_robust",
            "robust_v2": "best_by_robust_v2",
        }
        selected_key = metric_to_key.get(selected_metric, "best_by_coverage")
        selected_meta = _parse_best_meta(cfg.get(selected_key))

        if isinstance(selected_meta, dict):
            meta_epoch = selected_meta.get("epoch")
            meta_score = selected_meta.get("score")
            if isinstance(meta_epoch, (int, float)):
                result["best_epoch"] = int(meta_epoch)
                matched = history_df[history_df["epoch"] == int(meta_epoch)]
                if not matched.empty:
                    selected_row = matched.iloc[-1]
            if isinstance(meta_score, (int, float)):
                selected_metric_score = float(meta_score)

        # Keep best_coverage as the selected metric score for reporting consistency.
        result["best_coverage"] = selected_metric_score
        result["selected_metric"] = selected_metric
        result["selected_epoch"] = int(result["best_epoch"])
        result["selected_metric_score"] = selected_metric_score

    dataset_path = Path(result.get("dataset_path", "")) if result.get("dataset_path") else None
    if dataset_path and dataset_path.exists():
        ref_df = pd.read_csv(dataset_path, header=None, names=["methods", "branches", "statements"])
        max_methods = float(ref_df["methods"].max()) if len(ref_df) else 0.0
        max_branches = float(ref_df["branches"].max()) if len(ref_df) else 0.0
        max_lines = float(ref_df["statements"].max()) if len(ref_df) else 0.0
        eps = 1e-8

        result["real_max_methods"] = max_methods
        result["real_max_branches"] = max_branches
        result["real_max_lines"] = max_lines

        result["final_method_coverage_pct"] = float(result["final_max_methods"] / max(max_methods, eps) * 100)
        result["final_branch_coverage_pct"] = float(result["final_max_branches"] / max(max_branches, eps) * 100)
        result["final_line_coverage_pct"] = float(result["final_max_statements"] / max(max_lines, eps) * 100)

        result["best_method_coverage_pct"] = float(selected_row["max_methods"] / max(max_methods, eps) * 100)
        result["best_branch_coverage_pct"] = float(selected_row["max_branches"] / max(max_branches, eps) * 100)
        result["best_line_coverage_pct"] = float(selected_row["max_statements"] / max(max_lines, eps) * 100)
        result["delta_method_final_minus_best_pct"] = float(result["final_method_coverage_pct"] - result["best_method_coverage_pct"])
        result["delta_branch_final_minus_best_pct"] = float(result["final_branch_coverage_pct"] - result["best_branch_coverage_pct"])
        result["delta_line_final_minus_best_pct"] = float(result["final_line_coverage_pct"] - result["best_line_coverage_pct"])

    return result


def _save_summary(experiment_root: Path, output_name: str, metrics: dict) -> tuple[Path, Path]:
    summary_dir = experiment_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for hypothesis, values in metrics.items():
        row = {"hypothesis": hypothesis}
        row.update(values)
        rows.append(row)

    summary_csv = summary_dir / f"{output_name}_summary.csv"
    summary_json = summary_dir / f"{output_name}_summary.json"
    coverage_table_csv = summary_dir / f"{output_name}_coverage_table.csv"

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_csv, index=False)

    coverage_cols = [
        "hypothesis",
        "dataset_label",
        "selected_metric",
        "best_epoch",
        "selected_metric_score",
        "final_method_coverage_pct",
        "final_branch_coverage_pct",
        "final_line_coverage_pct",
        "best_method_coverage_pct",
        "best_branch_coverage_pct",
        "best_line_coverage_pct",
        "delta_method_final_minus_best_pct",
        "delta_branch_final_minus_best_pct",
        "delta_line_final_minus_best_pct",
    ]
    available = [c for c in coverage_cols if c in summary_df.columns]
    if available:
        summary_df[available].to_csv(coverage_table_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return summary_csv, summary_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SF110 H1/H2/H3 experiments")
    parser.add_argument("--profile", type=str, default="balanced", choices=["aggressive", "experimental", "conservative", "balanced", "optimized", "generalization-stress"])
    parser.add_argument("--epochs", type=int, default=40, help="Epochs per run")
    parser.add_argument("--subset-projects", type=int, default=10, help="Use first N SF110 projects (0=all)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-value", type=int, default=1000)
    parser.add_argument("--output-name", type=str, default="sf110_subset")
    parser.add_argument("--url", type=str, default=SF110_URL)
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare data, do not train")
    parser.add_argument(
        "--hypotheses",
        type=str,
        nargs="+",
        default=["H1", "H1A", "H2", "H3"],
        choices=["H1", "H1A", "H2", "H3"],
        help="Hypotheses to run (default: H1 H1A H2 H3)",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip SF110 download/prepare and reuse existing processed CSV files",
    )
    parser.add_argument("--validate", action="store_true", help="Validate hypothesis datasets via data_validation.py logic")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic samples for each executed hypothesis")
    parser.add_argument("--synthetic-samples", type=int, default=1000, help="Synthetic samples per hypothesis")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation report per hypothesis (requires synthetic generation)")
    parser.add_argument(
        "--best-metric",
        type=str,
        default="coverage",
        choices=["robust_v2", "robust", "coverage"],
        help="Best-checkpoint metric forwarded to train_basic_profiles.py",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Do not update best checkpoints before this epoch (exclusive)",
    )
    parser.add_argument("--eval-seed", type=int, default=777, help="Deterministic latent seed for per-epoch evaluation")
    parser.add_argument("--eval-samples", type=int, default=1000, help="Number of deterministic synthetic samples per epoch")
    parser.add_argument("--batch-log-interval", type=int, default=20,
                        help="Batch-level training log heartbeat interval passed to train_basic_profiles.py")
    parser.add_argument("--synthetic-jitter", type=float, default=0.05,
                        help="Jitter scale for synthetic-only H3 dataset generation")
    parser.add_argument("--synthetic-size", type=int, default=0,
                        help="Synthetic-only H3 dataset size (0 = same as combined size)")
    parser.add_argument(
        "--h1-min-size",
        type=int,
        default=500,
        help="Minimum target size for H1 training dataset (uses bootstrap+jitter if source is smaller)",
    )
    parser.add_argument(
        "--h1-jitter",
        type=float,
        default=0.03,
        help="Jitter scale for H1 expansion when enforcing --h1-min-size",
    )
    parser.add_argument(
        "--h1a-size",
        type=int,
        default=500,
        help="Target size for H1A augmented dataset",
    )
    parser.add_argument(
        "--h1a-jitter",
        type=float,
        default=0.05,
        help="Jitter scale for H1A augmented dataset",
    )

    args = parser.parse_args()

    experiment_root = Path("outputs") / "sf110_hypotheses"
    prepared_dir = Path("python/data/processed")

    sf110_strict = prepared_dir / f"{args.output_name}_strict.csv"
    if not args.skip_prepare:
        sf110_result = prepare_sf110_dataset(
            sf110_url=args.url,
            subset_projects=args.subset_projects,
            output_name=args.output_name,
            force_download=False,
            max_value=args.max_value,
        )
        sf110_strict = Path(sf110_result.strict_csv)
    elif not sf110_strict.exists():
        raise FileNotFoundError(
            f"--skip-prepare was provided but strict SF110 dataset was not found: {sf110_strict}"
        )

    junit_strict = _ensure_local_junit_strict(max_value=args.max_value)

    h1_pure_csv = prepared_dir / f"{args.output_name}_H1_pure_sf110_only_strict.csv"
    h1_csv = prepared_dir / f"{args.output_name}_H1_sf110_only_strict.csv"
    h1a_csv = prepared_dir / f"{args.output_name}_H1A_sf110_augmented_strict.csv"
    h2_csv = prepared_dir / f"{args.output_name}_H2_sf110_plus_junit_strict.csv"
    h3_csv = prepared_dir / f"{args.output_name}_H3_simulated_combined_strict.csv"

    # Preserve pure H1 without forced global deduplication.
    _sanitize_strict_csv(sf110_strict, h1_pure_csv, max_value=args.max_value, deduplicate=False)

    h1_pure_rows = len(pd.read_csv(h1_pure_csv, header=None))
    if h1_pure_rows >= args.h1_min_size:
        copyfile(h1_pure_csv, h1_csv)
    else:
        _build_synthetic_dataset_from_source(
            source_csv=h1_pure_csv,
            output_csv=h1_csv,
            random_seed=42,
            target_size=args.h1_min_size,
            jitter_scale=args.h1_jitter,
            max_value=args.max_value,
        )

    # H1A: explicit augmented variant from H1 pure base.
    _build_synthetic_dataset_from_source(
        source_csv=h1_pure_csv,
        output_csv=h1a_csv,
        random_seed=42,
        target_size=args.h1a_size,
        jitter_scale=args.h1a_jitter,
        max_value=args.max_value,
    )

    _merge_strict_csvs([sf110_strict, junit_strict], h2_csv, max_value=args.max_value)
    synthetic_size = args.synthetic_size if args.synthetic_size > 0 else None
    _build_synthetic_dataset_from_source(
        source_csv=h2_csv,
        output_csv=h3_csv,
        random_seed=42,
        target_size=synthetic_size,
        jitter_scale=args.synthetic_jitter,
        max_value=args.max_value,
    )

    if args.validate:
        validation_root = experiment_root / "validation"
        _validate_dataset(h1_csv, validation_root / "H1")
        _validate_dataset(h1a_csv, validation_root / "H1A")
        _validate_dataset(h2_csv, validation_root / "H2")
        _validate_dataset(h3_csv, validation_root / "H3")

    print("=" * 70)
    print("DONE: Hypothesis datasets ready")
    print(f"   H1 Pure (SF110 raw-clean): {h1_pure_csv} ({h1_pure_rows} rows)")
    print(f"   H1 (SF110): {h1_csv}")
    print(f"   H1A (Augmented): {h1a_csv}")
    print(f"   H2 (Combined): {h2_csv}")
    print(f"   H3 (Synthetic): {h3_csv}")
    print("=" * 70)

    if args.prepare_only:
        return

    hypothesis_to_dataset = {
        "H1": (h1_csv, "H1_sf110_only", "sf110"),
        "H1A": (h1a_csv, "H1A_sf110_augmented", "sf110_augmented"),
        "H2": (h2_csv, "H2_sf110_plus_junit", "combined"),
        "H3": (h3_csv, "H3_simulated_combined", "synthetic"),
    }

    evaluation_dir = experiment_root / "summaries"
    metrics = {}
    for hypothesis in args.hypotheses:
        dataset_path, run_name, dataset_label = hypothesis_to_dataset[hypothesis]
        _run_training(
            args.profile,
            args.epochs,
            dataset_path,
            run_name,
            args.device,
            experiment_root,
            args.best_metric,
            args.warmup_epochs,
            args.eval_seed,
            args.eval_samples,
            args.batch_log_interval,
        )
        run_dir = _latest_run_dir(experiment_root, run_name, args.profile)
        metrics[hypothesis] = _collect_run_metrics(run_dir)
        metrics[hypothesis]["dataset_label"] = dataset_label

        synthetic_path = None
        if args.generate_synthetic:
            norm_stats_path = Path(metrics[hypothesis]["norm_stats_path"])
            synthetic_path = _generate_synthetic_for_run(
                run_dir=run_dir,
                norm_stats_path=norm_stats_path,
                num_samples=args.synthetic_samples,
                device=args.device,
                best_metric=args.best_metric,
            )
            metrics[hypothesis]["synthetic_csv"] = str(synthetic_path).replace("\\", "/")

        if args.evaluate:
            if synthetic_path is None:
                raise ValueError("--evaluate requires --generate-synthetic")
            report_path = _evaluate_run(
                hypothesis=hypothesis,
                output_name=args.output_name,
                dataset_path=dataset_path,
                synthetic_csv=synthetic_path,
                evaluation_dir=evaluation_dir,
            )
            metrics[hypothesis]["evaluation_report"] = str(report_path).replace("\\", "/")

    summary_csv, summary_json = _save_summary(experiment_root, args.output_name, metrics)

    print("=" * 70)
    print("DONE: All hypothesis runs completed")
    print(f"   Outputs root: {experiment_root}")
    print(f"   Summary CSV: {summary_csv}")
    print(f"   Summary JSON: {summary_json}")
    coverage_table_csv = experiment_root / "summaries" / f"{args.output_name}_coverage_table.csv"
    if coverage_table_csv.exists():
        print(f"   Coverage Table CSV: {coverage_table_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
