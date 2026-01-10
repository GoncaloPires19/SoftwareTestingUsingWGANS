"""
Combine real and synthetic datasets for cascading synthesis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def combine_datasets(
    real_path: Path,
    synthetic_path: Path,
    output_path: Path,
    ratio: float = None
) -> pd.DataFrame:
    """
    Combine real and synthetic datasets
    
    Args:
        real_path: Path to real test data CSV
        synthetic_path: Path to synthetic data CSV
        output_path: Output path for combined data
        ratio: If specified, limit synthetic to N% of real (e.g., 0.5 = 50% synthetic)
    """
    
    logger.info("=" * 70)
    logger.info("DATASET COMBINATION FOR CASCADING SYNTHESIS")
    logger.info("=" * 70)
    
    # Load data
    logger.info(f"\n📂 Loading datasets...")
    
    real_df = pd.read_csv(real_path)
    synthetic_df = pd.read_csv(synthetic_path)
    
    logger.info(f"✅ Real data: {len(real_df)} samples from {real_path}")
    logger.info(f"   Columns: {list(real_df.columns)}")
    logger.info(f"✅ Synthetic data: {len(synthetic_df)} samples from {synthetic_path}")
    logger.info(f"   Columns: {list(synthetic_df.columns)}")
    
    # Normalize column names for real data if numeric
    if all(isinstance(col, str) and col.replace('.', '').isdigit() for col in real_df.columns):
        logger.info(f"\n🔄 Converting numeric column names to feature names...")
        real_df.columns = ['methods', 'branches', 'statements']
    
    # Ensure same columns
    real_cols = set(real_df.columns)
    synthetic_cols = set(synthetic_df.columns)
    
    common_cols = real_cols & synthetic_cols
    if not common_cols:
        raise ValueError("No common columns between datasets!")
    
    logger.info(f"\n📊 Common columns: {sorted(common_cols)}")
    
    # Select common columns
    real_df = real_df[list(common_cols)].copy()
    synthetic_df = synthetic_df[list(common_cols)].copy()
    
    # Apply ratio if specified
    if ratio is not None:
        max_synthetic = int(len(real_df) * ratio)
        if len(synthetic_df) > max_synthetic:
            logger.info(f"\n⚙️ Limiting synthetic to {ratio*100:.0f}% of real ({max_synthetic} samples)")
            synthetic_df = synthetic_df.sample(n=max_synthetic, random_state=42)
    
    # Combine
    combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
    
    # Add source label for tracking
    combined_df['source'] = 'real'
    combined_df.loc[len(real_df):, 'source'] = 'synthetic'
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    logger.info(f"\n✅ Combined dataset saved to {output_path}")
    logger.info(f"   Total samples: {len(combined_df)}")
    logger.info(f"   Real: {len(real_df)} ({len(real_df)/len(combined_df)*100:.1f}%)")
    logger.info(f"   Synthetic: {len(synthetic_df)} ({len(synthetic_df)/len(combined_df)*100:.1f}%)")
    
    # Statistics
    logger.info(f"\n📊 Combined Dataset Statistics:")
    for col in real_cols & synthetic_cols - {'source'}:
        if col in combined_df.columns:
            logger.info(f"   {col}: {combined_df[col].mean():.2f} ± {combined_df[col].std():.2f}")
    
    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine real and synthetic datasets")
    parser.add_argument("--real", type=Path, default=Path("python/data/junit_data_cleaned.csv"),
                       help="Real test data path")
    parser.add_argument("--synthetic", type=Path, default=Path("python/outputs/synthetic_100epochs.csv"),
                       help="Synthetic data path")
    parser.add_argument("--output", type=Path, default=Path("python/data/combined_augmented.csv"),
                       help="Output combined data path")
    parser.add_argument("--ratio", type=float, default=None,
                       help="Limit synthetic to N% of real (e.g., 0.5 for 50%)")
    
    args = parser.parse_args()
    
    combined = combine_datasets(
        real_path=args.real,
        synthetic_path=args.synthetic,
        output_path=args.output,
        ratio=args.ratio
    )
    
    logger.info("\n✅ Dataset combination complete!")
