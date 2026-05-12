"""
Create a quantile-aware augmented training dataset for WGAN experiments.

This script expands a 3-feature count dataset (methods, branches, statements)
using stratified bootstrap + log-space jitter. The goal is to preserve the
source distribution, especially the tail-heavy quantiles that plain bootstrap
often underrepresents during WGAN training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_validation import DataValidator


FEATURES = ["methods", "branches", "statements"]
DEFAULT_QUANTILES = [0.0, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]


def _load_source_dataset(csv_path: Path, max_value: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=FEATURES)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df[FEATURES] >= 0).all(axis=1)]
    df = df[(df[FEATURES] <= max_value).all(axis=1)]
    return df.reset_index(drop=True)


def _compute_feature_thresholds(df: pd.DataFrame, quantiles: list[float]) -> dict[str, np.ndarray]:
    thresholds: dict[str, np.ndarray] = {}
    for feature in FEATURES:
        values = df[feature].quantile(quantiles).to_numpy(dtype=np.float64)
        values[0] = float(df[feature].min())
        values[-1] = float(df[feature].max())
        thresholds[feature] = values
    return thresholds


def _assign_strata(df: pd.DataFrame, thresholds: dict[str, np.ndarray]) -> np.ndarray:
    severity = np.zeros(len(df), dtype=np.int64)
    for feature in FEATURES:
        bins = thresholds[feature][1:-1]
        feature_codes = np.searchsorted(bins, df[feature].to_numpy(dtype=np.float64), side="right")
        severity = np.maximum(severity, feature_codes)
    return severity


def _allocate_target_counts(strata: np.ndarray, target_size: int, tail_amplification: float) -> dict[int, int]:
    unique_strata, counts = np.unique(strata, return_counts=True)
    max_stratum = int(unique_strata.max()) if len(unique_strata) else 0

    weights = counts.astype(np.float64)
    if abs(tail_amplification - 1.0) > 1e-9 and max_stratum > 0:
        severity_scale = unique_strata.astype(np.float64) / float(max_stratum)
        weights *= 1.0 + (tail_amplification - 1.0) * severity_scale

    weights /= weights.sum()
    raw = weights * target_size
    base = np.floor(raw).astype(int)
    remainder = target_size - int(base.sum())
    if remainder > 0:
        order = np.argsort(raw - base)[::-1]
        for idx in order[:remainder]:
            base[idx] += 1

    return {int(stratum): int(count) for stratum, count in zip(unique_strata, base)}


def _jitter_group(
    sampled_group: pd.DataFrame,
    source_group: pd.DataFrame,
    rng: np.random.Generator,
    jitter_scale: float,
    max_value: int,
) -> pd.DataFrame:
    augmented = sampled_group.copy()
    for feature in FEATURES:
        source_log = np.log1p(source_group[feature].to_numpy(dtype=np.float64))
        sampled_log = np.log1p(sampled_group[feature].to_numpy(dtype=np.float64))
        log_std = float(source_log.std())
        if log_std <= 1e-12:
            global_scale = np.log1p(max(float(source_group[feature].std()), 1.0))
            log_std = max(global_scale, 1e-3)

        noise = rng.normal(0.0, log_std * jitter_scale, size=len(sampled_group))
        jittered = np.expm1(np.clip(sampled_log + noise, 0.0, None))
        augmented[feature] = np.clip(jittered, 0.0, float(max_value))

    return augmented


def build_quantile_augmented_dataset(
    source_csv: Path,
    output_csv: Path,
    report_path: Path,
    target_size: int,
    random_seed: int,
    jitter_scale: float,
    tail_amplification: float,
    max_value: int,
    quantiles: list[float],
) -> tuple[Path, Path]:
    source_df = _load_source_dataset(source_csv, max_value=max_value)
    if target_size <= len(source_df):
        raise ValueError(
            f"target_size must be larger than source size ({len(source_df)}). Got {target_size}."
        )

    rng = np.random.default_rng(random_seed)
    thresholds = _compute_feature_thresholds(source_df, quantiles)
    source_strata = _assign_strata(source_df, thresholds)
    target_counts = _allocate_target_counts(source_strata, target_size, tail_amplification)

    source_df = source_df.copy()
    source_df["_stratum"] = source_strata
    groups: list[pd.DataFrame] = []
    augmented_strata_counts: dict[int, int] = {}

    for stratum, requested_count in sorted(target_counts.items()):
        source_group = source_df[source_df["_stratum"] == stratum][FEATURES].reset_index(drop=True)
        if source_group.empty or requested_count <= 0:
            continue

        sampled_group = source_group.sample(
            n=requested_count,
            replace=True,
            random_state=int(rng.integers(0, 2**31 - 1)),
        ).reset_index(drop=True)
        augmented_group = _jitter_group(
            sampled_group=sampled_group,
            source_group=source_group,
            rng=rng,
            jitter_scale=jitter_scale,
            max_value=max_value,
        )
        groups.append(augmented_group)
        augmented_strata_counts[int(stratum)] = len(augmented_group)

    augmented_df = pd.concat(groups, ignore_index=True)
    augmented_df = augmented_df[FEATURES].clip(lower=0.0, upper=float(max_value)).round(3)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(output_csv, index=False, header=False)

    report = {
        "source_csv": str(source_csv),
        "output_csv": str(output_csv),
        "source_rows": int(len(source_df)),
        "augmented_rows": int(len(augmented_df)),
        "target_size": int(target_size),
        "random_seed": int(random_seed),
        "jitter_scale": float(jitter_scale),
        "tail_amplification": float(tail_amplification),
        "quantiles": quantiles,
        "strata": {
            "source": {str(int(k)): int(v) for k, v in pd.Series(source_strata).value_counts().sort_index().items()},
            "augmented": {str(k): int(v) for k, v in augmented_strata_counts.items()},
        },
        "features": {},
    }

    for feature in FEATURES:
        source_series = source_df[feature]
        augmented_series = augmented_df[feature]
        source_quantiles = source_series.quantile(quantiles).to_numpy(dtype=np.float64)
        augmented_quantiles = augmented_series.quantile(quantiles).to_numpy(dtype=np.float64)
        rel_error = np.abs(augmented_quantiles - source_quantiles) / np.maximum(source_quantiles, 1e-8)
        report["features"][feature] = {
            "source_mean": float(source_series.mean()),
            "augmented_mean": float(augmented_series.mean()),
            "source_std": float(source_series.std()),
            "augmented_std": float(augmented_series.std()),
            "source_max": float(source_series.max()),
            "augmented_max": float(augmented_series.max()),
            "source_quantiles": [float(x) for x in source_quantiles],
            "augmented_quantiles": [float(x) for x in augmented_quantiles],
            "quantile_relative_error": [float(x) for x in rel_error],
            "mean_quantile_relative_error": float(rel_error.mean()),
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return output_csv, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV with 3 columns (methods, branches, statements)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV path for the augmented dataset",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1200,
        help="Number of rows in the augmented dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and jitter",
    )
    parser.add_argument(
        "--jitter-scale",
        type=float,
        default=0.05,
        help="Gaussian jitter scale applied in log space within each stratum",
    )
    parser.add_argument(
        "--tail-amplification",
        type=float,
        default=1.20,
        help="Relative oversampling factor for higher-quantile strata (1.0 disables)",
    )
    parser.add_argument(
        "--max-value",
        type=int,
        default=1000,
        help="Hard upper bound applied after jittering",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the augmented CSV after generation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input)
    output_csv = Path(args.output)
    report_path = output_csv.with_suffix("")
    report_path = report_path.parent / f"{report_path.name}_augmentation_report.json"

    output_csv, report_path = build_quantile_augmented_dataset(
        source_csv=input_csv,
        output_csv=output_csv,
        report_path=report_path,
        target_size=args.target_size,
        random_seed=args.seed,
        jitter_scale=args.jitter_scale,
        tail_amplification=args.tail_amplification,
        max_value=args.max_value,
        quantiles=DEFAULT_QUANTILES,
    )

    print("=" * 70)
    print("DONE: Quantile-aware augmented dataset created")
    print(f"   Input:  {input_csv}")
    print(f"   Output: {output_csv}")
    print(f"   Report: {report_path}")

    if args.validate:
        validator = DataValidator(validation_log_dir=Path("outputs") / "validation" / output_csv.stem)
        is_valid, validation_report = validator.validate_csv_structure(output_csv)
        print(f"   Validation: {'OK' if is_valid else 'FAILED'}")
        if validation_report['warnings']:
            print(f"   Warnings: {len(validation_report['warnings'])}")

    print("=" * 70)


if __name__ == "__main__":
    main()