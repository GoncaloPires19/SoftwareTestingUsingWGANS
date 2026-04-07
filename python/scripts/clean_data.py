#!/usr/bin/env python3
"""Compatibility cleaner that performs strict numeric cleaning for 3-feature JUnit CSV files."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def clean_dataset(input_path: Path, output_path: Path, min_value: float = 0.0, max_value: float = 1000.0) -> int:
    df = pd.read_csv(input_path, header=None)
    df = df.iloc[:, :3].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df >= min_value).all(axis=1) & (df <= max_value).all(axis=1)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, header=False, index=False)
    return len(df)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean contaminated JUnit metric data into strict 3-column CSV")
    parser.add_argument("--input", type=str, default="python/data/junit_data.csv")
    parser.add_argument("--output", type=str, default="python/data/junit_data_cleaned.csv")
    parser.add_argument("--min", dest="min_value", type=float, default=0.0)
    parser.add_argument("--max", dest="max_value", type=float, default=1000.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 1

    rows = clean_dataset(input_path, output_path, min_value=args.min_value, max_value=args.max_value)
    print(f"Cleaned dataset saved to {output_path} ({rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
