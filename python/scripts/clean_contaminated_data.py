"""
Strict Data Cleaning Script - Remove Contaminated Outliers and Negative Values

This script addresses the critical data quality issues identified:
1. Extreme outliers (values up to 3.1M) from arithmetic test cases
2. Negative values from subtraction operations
3. Corrupted normalization statistics

Strategy: Keep only samples where ALL features are in [0, 1000]
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Configuration
INPUT_FILE = Path(__file__).parent.parent / "data" / "junit_data_cleaned.csv"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "junit_data_cleaned_strict.csv"
NORM_STATS_FILE = Path(__file__).parent.parent / "data" / "junit_data_cleaned_norm_stats.json"
REPORT_FILE = Path(__file__).parent.parent / "data" / "cleaning_report.json"

MIN_BOUND = 0
MAX_BOUND = 1000
MIN_SAMPLES_REQUIRED = 500

# Column names (these represent test characteristics)
FEATURE_NAMES = ['methods', 'branches', 'statements']


def load_data(file_path):
    """Load CSV data without headers"""
    print(f"📂 Loading data from: {file_path}")
    df = pd.read_csv(file_path, header=None, names=FEATURE_NAMES)
    print(f"   Loaded {len(df)} samples")
    return df


def analyze_contamination(df):
    """Analyze and report data quality issues"""
    print("\n🔍 Analyzing data contamination...")
    
    issues = {
        'total_samples': len(df),
        'negative_values': {},
        'extreme_outliers': {},
        'invalid_samples': 0,
        'valid_samples': 0
    }
    
    for col in FEATURE_NAMES:
        # Check for negative values
        neg_count = (df[col] < MIN_BOUND).sum()
        issues['negative_values'][col] = int(neg_count)
        
        # Check for extreme outliers
        outlier_count = (df[col] > MAX_BOUND).sum()
        issues['extreme_outliers'][col] = int(outlier_count)
        
        print(f"   {col}: {neg_count} negative, {outlier_count} outliers (>{MAX_BOUND})")
    
    # Count rows with ANY invalid value
    invalid_mask = ((df < MIN_BOUND).any(axis=1) | (df > MAX_BOUND).any(axis=1))
    issues['invalid_samples'] = int(invalid_mask.sum())
    issues['valid_samples'] = int((~invalid_mask).sum())
    
    print(f"\n   Total invalid samples: {issues['invalid_samples']}")
    print(f"   Total valid samples: {issues['valid_samples']}")
    
    return issues


def clean_data(df):
    """Remove samples with ANY value outside [MIN_BOUND, MAX_BOUND]"""
    print(f"\n🧹 Cleaning data (keeping only [{MIN_BOUND}, {MAX_BOUND}])...")
    
    original_count = len(df)
    
    # Filter: keep only rows where ALL values are in bounds
    clean_mask = ((df >= MIN_BOUND).all(axis=1) & (df <= MAX_BOUND).all(axis=1))
    df_clean = df[clean_mask].copy()
    
    removed_count = original_count - len(df_clean)
    retention_rate = (len(df_clean) / original_count) * 100
    
    print(f"   Removed: {removed_count} samples ({100-retention_rate:.1f}%)")
    print(f"   Retained: {len(df_clean)} samples ({retention_rate:.1f}%)")
    
    # Verify no invalid values remain
    assert (df_clean >= MIN_BOUND).all().all(), "ERROR: Negative values still present!"
    assert (df_clean <= MAX_BOUND).all().all(), "ERROR: Outliers still present!"
    assert not df_clean.isna().any().any(), "ERROR: NaN values present!"
    
    print("   ✅ Verification passed: No invalid values remain")
    
    return df_clean, removed_count, retention_rate


def compute_statistics(df, label="Dataset"):
    """Compute and display statistics"""
    print(f"\n📊 {label} Statistics:")
    stats = {}
    
    for col in FEATURE_NAMES:
        col_stats = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std()),
            'q25': float(df[col].quantile(0.25)),
            'q75': float(df[col].quantile(0.75)),
            'q95': float(df[col].quantile(0.95))
        }
        stats[col] = col_stats
        
        print(f"   {col}:")
        print(f"      Range: [{col_stats['min']:.1f}, {col_stats['max']:.1f}]")
        print(f"      Mean/Median: {col_stats['mean']:.1f} / {col_stats['median']:.1f}")
        print(f"      Std: {col_stats['std']:.1f}")
        print(f"      Q25/Q75/Q95: {col_stats['q25']:.1f} / {col_stats['q75']:.1f} / {col_stats['q95']:.1f}")
    
    return stats


def generate_normalization_stats(df):
    """Generate normalization parameters for WGAN training"""
    print(f"\n🔢 Generating normalization statistics...")
    
    norm_stats = {
        'min': df.min().tolist(),
        'max': df.max().tolist(),
        'range': (df.max() - df.min()).tolist(),
        'mean': df.mean().tolist(),
        'std': df.std().tolist()
    }
    
    # Also compute feature-wise stats (for instrumentation.py)
    feature_stats = {}
    for i, col in enumerate(FEATURE_NAMES):
        feature_stats[f'feature_{i}'] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'range': float(df[col].max() - df[col].min()),
            'std': float(df[col].std())
        }
    
    norm_stats['features'] = feature_stats
    
    print(f"   Min values: {norm_stats['min']}")
    print(f"   Max values: {norm_stats['max']}")
    print(f"   Ranges: {norm_stats['range']}")
    
    return norm_stats


def save_results(df_clean, norm_stats, issues, removed_count, retention_rate):
    """Save cleaned data, normalization stats, and cleaning report"""
    
    # Save cleaned data (without headers to match original format)
    print(f"\n💾 Saving cleaned data to: {OUTPUT_FILE}")
    df_clean.to_csv(OUTPUT_FILE, index=False, header=False)
    print(f"   ✅ Saved {len(df_clean)} clean samples")
    
    # Save normalization stats
    print(f"\n💾 Saving normalization stats to: {NORM_STATS_FILE}")
    with open(NORM_STATS_FILE, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"   ✅ Saved normalization parameters")
    
    # Generate cleaning report
    report = {
        'input_file': str(INPUT_FILE),
        'output_file': str(OUTPUT_FILE),
        'filter_bounds': [MIN_BOUND, MAX_BOUND],
        'original_samples': issues['total_samples'],
        'removed_samples': removed_count,
        'retained_samples': len(df_clean),
        'retention_rate_percent': round(retention_rate, 2),
        'contamination_issues': {
            'negative_values': issues['negative_values'],
            'extreme_outliers': issues['extreme_outliers']
        },
        'normalization_stats': norm_stats
    }
    
    print(f"\n💾 Saving cleaning report to: {REPORT_FILE}")
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"   ✅ Saved cleaning report")


def main():
    """Main cleaning pipeline"""
    print("=" * 70)
    print("STRICT DATA CLEANING - Remove Contaminated Outliers")
    print("=" * 70)
    
    # 1. Load original data
    df_original = load_data(INPUT_FILE)
    
    # 2. Analyze contamination
    issues = analyze_contamination(df_original)
    
    # 3. Show original statistics
    original_stats = compute_statistics(df_original, "Original Data")
    
    # 4. Clean data
    df_clean, removed_count, retention_rate = clean_data(df_original)
    
    # 5. Verify minimum samples requirement
    if len(df_clean) < MIN_SAMPLES_REQUIRED:
        print(f"\n❌ ERROR: Only {len(df_clean)} samples after cleaning")
        print(f"   Required minimum: {MIN_SAMPLES_REQUIRED} samples")
        print(f"   Consider relaxing MAX_BOUND or investigating data source")
        return False
    
    # 6. Show cleaned statistics
    clean_stats = compute_statistics(df_clean, "Cleaned Data")
    
    # 7. Generate normalization stats
    norm_stats = generate_normalization_stats(df_clean)
    
    # 8. Save results
    save_results(df_clean, norm_stats, issues, removed_count, retention_rate)
    
    # 9. Final summary
    print("\n" + "=" * 70)
    print("✅ DATA CLEANING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"   Original samples: {issues['total_samples']}")
    print(f"   Clean samples: {len(df_clean)}")
    print(f"   Retention rate: {retention_rate:.1f}%")
    print(f"\n   Next steps:")
    print(f"   1. Review cleaning_report.json for details")
    print(f"   2. Update config.py to use junit_data_cleaned_strict.csv")
    print(f"   3. Run training with clean data")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
