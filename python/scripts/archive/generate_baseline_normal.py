#!/usr/bin/env python3
"""Generate a non-GAN Gaussian baseline synthetic dataset from real data statistics.

This baseline is useful to objectively compare WGAN against a simple statistical sampler.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gaussian baseline synthetic samples")
    parser.add_argument("--real-data", type=str, required=True,
                        help="Path to real CSV data (3 columns: methods, branches, statements)")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of synthetic baseline samples to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path")
    args = parser.parse_args()

    np.random.seed(args.seed)

    real_path = Path(args.real_data)
    if not real_path.exists():
        raise FileNotFoundError(f"Real data not found: {real_path}")

    real_df = pd.read_csv(real_path, header=None)
    real = real_df.iloc[:, :3].to_numpy(dtype=float)

    mean = real.mean(axis=0)
    std = real.std(axis=0, ddof=1)
    std = np.where(std <= 0, 1e-6, std)

    synthetic = np.random.normal(loc=mean, scale=std, size=(args.samples, 3))
    synthetic = np.clip(synthetic, a_min=0.0, a_max=None)

    out_df = pd.DataFrame(synthetic, columns=["methods", "branches", "statements"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("=" * 80)
    print("Baseline dataset generated")
    print(f"real_data: {real_path}")
    print(f"samples:   {args.samples}")
    print(f"seed:      {args.seed}")
    print(f"output:    {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
