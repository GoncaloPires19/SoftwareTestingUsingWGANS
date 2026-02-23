"""
Strict Data Cleaning Script

Removes corrupted data from junit_data.csv:
- Integer overflow values (±2^31)
- Extreme outliers (< -1e9 or > 1e9)
- NaN and Inf values

Generates junit_data_cleaned.csv with only valid test data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def clean_data_strict(input_path: str, output_path: str, threshold: float = 1e9):
    """
    Strictly clean test data by removing corrupted values.
    
    Args:
        input_path: Path to raw junit_data.csv
        output_path: Path to save cleaned junit_data_cleaned.csv
        threshold: Maximum absolute value allowed (default 1e9)
    
    Returns:
        dict: Statistics about the cleaning process
    """
    logger.info(f"Loading data from {input_path}...")
    
    # Load raw data (no header, 3 columns: methods, branches, statements)
    try:
        df = pd.read_csv(input_path, header=None, names=['methods', 'branches', 'statements'])
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    initial_count = len(df)
    logger.info(f"Initial data: {initial_count} rows")
    
    # Track removed rows for reporting
    removed_stats = {
        'initial_count': initial_count,
        'nan_inf_removed': 0,
        'overflow_removed': 0,
        'outlier_removed': 0,
        'final_count': 0
    }
    
    # Step 1: Remove NaN and Inf values
    nan_inf_mask = df.isna().any(axis=1) | np.isinf(df).any(axis=1)
    removed_stats['nan_inf_removed'] = nan_inf_mask.sum()
    df = df[~nan_inf_mask]
    logger.info(f"Removed {removed_stats['nan_inf_removed']} rows with NaN/Inf")
    
    # Step 2: Remove integer overflow values (exactly ±2147483648)
    overflow_values = [2147483647, -2147483648, 2147483648, -2147483647]
    overflow_mask = df.isin(overflow_values).any(axis=1)
    removed_stats['overflow_removed'] = overflow_mask.sum()
    df = df[~overflow_mask]
    logger.info(f"Removed {removed_stats['overflow_removed']} rows with integer overflow")
    
    # Step 3: Remove extreme outliers (beyond threshold)
    outlier_mask = (df.abs() > threshold).any(axis=1)
    removed_stats['outlier_removed'] = outlier_mask.sum()
    df = df[~outlier_mask]
    logger.info(f"Removed {removed_stats['outlier_removed']} rows with values > {threshold}")
    
    # Final count
    removed_stats['final_count'] = len(df)
    removed_stats['total_removed'] = (
        removed_stats['nan_inf_removed'] + 
        removed_stats['overflow_removed'] + 
        removed_stats['outlier_removed']
    )
    
    logger.info(f"Final data: {removed_stats['final_count']} rows")
    logger.info(f"Total removed: {removed_stats['total_removed']} rows "
                f"({100 * removed_stats['total_removed'] / initial_count:.1f}%)")
    
    # Data quality check
    if removed_stats['final_count'] < 50:
        logger.warning(f"WARNING: Only {removed_stats['final_count']} samples remain. "
                       f"This may be too few for effective WGAN training.")
    
    # Save cleaned data
    logger.info(f"Saving cleaned data to {output_path}...")
    df.to_csv(output_path, index=False, header=False)
    
    # Display statistics
    logger.info("\nCleaned Data Statistics:")
    logger.info(f"  Methods    - min: {df['methods'].min():.2f}, max: {df['methods'].max():.2f}, mean: {df['methods'].mean():.2f}")
    logger.info(f"  Branches   - min: {df['branches'].min():.2f}, max: {df['branches'].max():.2f}, mean: {df['branches'].mean():.2f}")
    logger.info(f"  Statements - min: {df['statements'].min():.2f}, max: {df['statements'].max():.2f}, mean: {df['statements'].mean():.2f}")
    
    return removed_stats


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "python" / "data"
    input_path = data_dir / "junit_data.csv"
    output_path = data_dir / "junit_data_cleaned.csv"
    
    # Check if input file exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Please run Java tests first: mvn clean test")
        return
    
    # Clean the data
    try:
        stats = clean_data_strict(str(input_path), str(output_path))
        logger.info("\n✅ Data cleaning completed successfully!")
        logger.info(f"Cleaned data saved to: {output_path}")
        
        # Warn if too much data was removed
        if stats['total_removed'] > stats['initial_count'] * 0.5:
            logger.warning(f"\n⚠️ WARNING: More than 50% of data was removed!")
            logger.warning(f"Consider regenerating test data (mvn clean test)")
        
    except Exception as e:
        logger.error(f"\n❌ Data cleaning failed: {e}")
        raise


if __name__ == "__main__":
    main()
