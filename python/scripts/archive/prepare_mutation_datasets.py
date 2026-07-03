#!/usr/bin/env python3
"""Prepare mutation-focused datasets from validated JUnit feature data."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["methods", "branches", "statements"]


def _load_valid_dataset(data_path: Path) -> pd.DataFrame:
	df = pd.read_csv(data_path, header=None)
	df = df.iloc[:, :3]
	df.columns = FEATURE_COLUMNS

	# Strict numeric validation and safety range expected by this project.
	df = df.apply(pd.to_numeric, errors="coerce")
	df = df.replace([np.inf, -np.inf], np.nan).dropna()
	df = df[(df[FEATURE_COLUMNS] >= 0).all(axis=1) & (df[FEATURE_COLUMNS] <= 1000).all(axis=1)]

	if df.empty:
		raise ValueError(f"No valid rows after filtering in {data_path}")

	return df.reset_index(drop=True)


def _build_quality_filtered(df: pd.DataFrame, top_ratio: float) -> pd.DataFrame:
	ratio = min(max(top_ratio, 0.01), 1.0)

	normalized = (df[FEATURE_COLUMNS] - df[FEATURE_COLUMNS].min()) / (df[FEATURE_COLUMNS].max() - df[FEATURE_COLUMNS].min() + 1e-12)
	quality_score = (
		0.15 * normalized["methods"]
		+ 0.40 * normalized["branches"]
		+ 0.45 * normalized["statements"]
	)

	ranked = df.copy()
	ranked["quality_score"] = quality_score

	keep = max(1, int(len(ranked) * ratio))
	filtered = ranked.sort_values("quality_score", ascending=False).head(keep)
	return filtered[FEATURE_COLUMNS].reset_index(drop=True)


def _build_deduplicated(df: pd.DataFrame) -> pd.DataFrame:
	return df.drop_duplicates().reset_index(drop=True)


def main() -> int:
	parser = argparse.ArgumentParser(description="Prepare EXP B/C mutation datasets from clean feature data")
	parser.add_argument("--input", type=str, default="python/data/junit_data_cleaned_strict.csv",
						help="Input CSV path (headerless 3-column features)")
	parser.add_argument("--output-b", type=str, default="python/data/exp_b_high_quality.csv",
						help="Output path for high-quality filtered dataset (EXP B)")
	parser.add_argument("--output-c", type=str, default="python/data/exp_c_low_redundancy.csv",
						help="Output path for deduplicated dataset (EXP C)")
	parser.add_argument("--top-ratio", type=float, default=0.2,
						help="Top ratio to keep for EXP B (default: 0.2)")
	parser.add_argument("--summary", type=str, default="outputs/mutation_experiments/dataset_preparation_summary.json",
						help="Where to save preparation summary JSON")
	args = parser.parse_args()

	data_path = Path(args.input)
	output_b = Path(args.output_b)
	output_c = Path(args.output_c)
	summary_path = Path(args.summary)

	if not data_path.exists():
		raise FileNotFoundError(f"Input dataset not found: {data_path}")

	df = _load_valid_dataset(data_path)
	df_b = _build_quality_filtered(df, args.top_ratio)
	df_c = _build_deduplicated(df)

	output_b.parent.mkdir(parents=True, exist_ok=True)
	output_c.parent.mkdir(parents=True, exist_ok=True)
	summary_path.parent.mkdir(parents=True, exist_ok=True)

	df_b.to_csv(output_b, index=False, header=False)
	df_c.to_csv(output_c, index=False, header=False)

	summary = {
		"input": str(data_path),
		"input_rows": int(len(df)),
		"exp_b_rows": int(len(df_b)),
		"exp_c_rows": int(len(df_c)),
		"top_ratio": float(args.top_ratio),
		"exp_b_output": str(output_b),
		"exp_c_output": str(output_c),
	}
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	print(f"Generated EXP B dataset: {len(df_b)} samples -> {output_b}")
	print(f"Generated EXP C dataset: {len(df_c)} samples -> {output_c}")
	print(f"Saved summary: {summary_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
