"""
Pre-Training Validation Script

Validates the entire training pipeline is ready before starting WGAN training.
This script should be run BEFORE train_profile_aware.py to catch issues early.

Checks:
1. Dataset exists and has minimum required samples
2. No outliers > 1000 or negative values in dataset
3. Normalization stats are reasonable (not contaminated by outliers)
4. Models directory exists and optionally has backups
5. All critical dependencies are available
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import torch


# Configuration
MIN_SAMPLES = 500
MAX_VALUE_THRESHOLD = 1000
MIN_VALUE_THRESHOLD = 0
REQUIRED_COLUMNS = 3

# Paths
DATA_PATH = Path("python/data/junit_data_cleaned_strict.csv")
NORM_STATS_PATH = Path("python/data/junit_data_cleaned_norm_stats.json")
MODELS_DIR = Path("python/models")
CLEANING_REPORT = Path("python/data/cleaning_report.json")

FEATURE_NAMES = ['methods', 'branches', 'statements']


class ValidationError(Exception):
    """Raised when validation fails"""
    pass


def print_section(title: str):
    """Print section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check_file_exists(path: Path, description: str) -> bool:
    """Check if file exists"""
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists


def validate_dataset():
    """Validate dataset quality"""
    print_section("1. DATASET VALIDATION")
    
    # Check file exists
    if not check_file_exists(DATA_PATH, "Cleaned dataset"):
        raise ValidationError(f"Dataset not found: {DATA_PATH}")
    
    # Load data
    print(f"\n📂 Loading dataset...")
    df = pd.read_csv(DATA_PATH, header=None, names=FEATURE_NAMES)
    print(f"   Loaded {len(df)} samples with {df.shape[1]} features")
    
    # Check minimum samples
    if len(df) < MIN_SAMPLES:
        raise ValidationError(f"Insufficient samples: {len(df)} < {MIN_SAMPLES}")
    print(f"✅ Sufficient samples: {len(df)} >= {MIN_SAMPLES}")
    
    # Check for negative values
    negatives = (df < MIN_VALUE_THRESHOLD).sum()
    if negatives.any():
        print(f"❌ Dataset contains negative values:")
        for col, count in negatives.items():
            if count > 0:
                print(f"   {col}: {count} negative values")
        raise ValidationError("Dataset contains negative values")
    print(f"✅ No negative values")
    
    # Check for outliers
    outliers = (df > MAX_VALUE_THRESHOLD).sum()
    if outliers.any():
        print(f"❌ Dataset contains outliers (>{MAX_VALUE_THRESHOLD}):")
        for col, count in outliers.items():
            if count > 0:
                print(f"   {col}: {count} outliers")
        raise ValidationError(f"Dataset contains outliers > {MAX_VALUE_THRESHOLD}")
    print(f"✅ No outliers (all values <= {MAX_VALUE_THRESHOLD})")
    
    # Check for NaN/Inf
    if df.isna().any().any():
        raise ValidationError("Dataset contains NaN values")
    if np.isinf(df.values).any():
        raise ValidationError("Dataset contains Inf values")
    print(f"✅ No NaN or Inf values")
    
    # Display statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Shape: {df.shape}")
    for col in FEATURE_NAMES:
        print(f"   {col}: min={df[col].min():.1f}, max={df[col].max():.1f}, "
              f"mean={df[col].mean():.1f}, median={df[col].median():.1f}")
    
    return df


def validate_normalization_stats():
    """Validate normalization statistics"""
    print_section("2. NORMALIZATION STATS VALIDATION")
    
    # Check file exists
    if not check_file_exists(NORM_STATS_PATH, "Normalization stats"):
        raise ValidationError(f"Normalization stats not found: {NORM_STATS_PATH}")
    
    # Load stats
    print(f"\n📂 Loading normalization stats...")
    with open(NORM_STATS_PATH, 'r') as f:
        norm_stats = json.load(f)
    
    # Check structure
    required_keys = ['min', 'max', 'range']
    missing = [k for k in required_keys if k not in norm_stats]
    if missing:
        raise ValidationError(f"Missing keys in norm_stats: {missing}")
    print(f"✅ Contains required keys: {required_keys}")
    
    # Check values are reasonable (not contaminated by outliers)
    max_values = norm_stats['max']
    print(f"\n📊 Max values in normalization stats:")
    for i, (feature, max_val) in enumerate(zip(FEATURE_NAMES, max_values)):
        print(f"   {feature}: {max_val}")
        if max_val > MAX_VALUE_THRESHOLD * 1.1:  # Allow 10% buffer
            raise ValidationError(
                f"Normalization max for {feature} is too high: {max_val} > {MAX_VALUE_THRESHOLD}"
            )
    print(f"✅ All max values are reasonable (<= {MAX_VALUE_THRESHOLD * 1.1})")
    
    # Check min values are non-negative
    min_values = norm_stats['min']
    for i, (feature, min_val) in enumerate(zip(FEATURE_NAMES, min_values)):
        if min_val < MIN_VALUE_THRESHOLD:
            raise ValidationError(f"Normalization min for {feature} is negative: {min_val}")
    print(f"✅ All min values are non-negative")
    
    return norm_stats


def validate_cleaning_report():
    """Validate cleaning report exists and shows good results"""
    print_section("3. CLEANING REPORT VALIDATION")
    
    if not check_file_exists(CLEANING_REPORT, "Cleaning report"):
        print(f"⚠️  Cleaning report not found (optional)")
        return None
    
    # Load report
    with open(CLEANING_REPORT, 'r') as f:
        report = json.load(f)
    
    print(f"\n📊 Cleaning Report Summary:")
    print(f"   Original samples: {report['original_samples']}")
    print(f"   Retained samples: {report['retained_samples']}")
    print(f"   Retention rate: {report['retention_rate_percent']}%")
    print(f"   Removed issues:")
    
    issues = report['contamination_issues']
    for issue_type, feature_counts in issues.items():
        total = sum(feature_counts.values())
        if total > 0:
            print(f"      {issue_type}: {total} samples")
    
    if report['retention_rate_percent'] < 80:
        print(f"⚠️  Low retention rate: {report['retention_rate_percent']}%")
        print(f"   Consider investigating data quality")
    else:
        print(f"✅ Good retention rate: {report['retention_rate_percent']}%")
    
    return report


def validate_environment():
    """Validate Python environment and dependencies"""
    print_section("4. ENVIRONMENT VALIDATION")
    
    # Check PyTorch
    print(f"   PyTorch version: {torch.__version__}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    if device.type == 'cuda':
        print(f"   CUDA version: {torch.version.cuda}")
    print(f"✅ PyTorch available")
    
    # Check pandas/numpy
    print(f"   Pandas version: {pd.__version__}")
    print(f"   NumPy version: {np.__version__}")
    print(f"✅ Data libraries available")
    
    # Check models directory
    if not MODELS_DIR.exists():
        print(f"⚠️  Models directory doesn't exist, creating: {MODELS_DIR}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    else:
        # Check if there are existing models
        existing_models = list(MODELS_DIR.glob("*.pth"))
        if existing_models:
            print(f"⚠️  Found {len(existing_models)} existing model files:")
            for model in existing_models[:5]:  # Show first 5
                print(f"      {model.name}")
            if len(existing_models) > 5:
                print(f"      ... and {len(existing_models) - 5} more")
            print(f"   Consider backing up or moving old models before training")
    print(f"✅ Models directory exists: {MODELS_DIR}")
    
    return True


def validate_synthetic_generation_config():
    """Validate synthetic generation will work correctly"""
    print_section("5. SYNTHETIC GENERATION CONFIG")
    
    # Check generate_synthetic.py uses correct norm stats path
    generate_script = Path("python/scripts/generate_synthetic.py")
    if generate_script.exists():
        with open(generate_script, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'junit_data_cleaned_norm_stats.json' in content:
                print(f"✅ generate_synthetic.py configured correctly")
            else:
                print(f"⚠️  generate_synthetic.py may have incorrect norm stats path")
    else:
        print(f"⚠️  generate_synthetic.py not found")
    
    return True


def main():
    """Run all validations"""
    print("=" * 70)
    print("PRE-TRAINING VALIDATION")
    print("=" * 70)
    print("This script validates the training pipeline is ready.")
    print("Run this BEFORE starting WGAN training.")
    
    try:
        # Run validations
        df = validate_dataset()
        norm_stats = validate_normalization_stats()
        report = validate_cleaning_report()
        validate_environment()
        validate_synthetic_generation_config()
        
        # Final summary
        print_section("VALIDATION SUMMARY")
        print("✅ ALL VALIDATIONS PASSED")
        print()
        print("Your training pipeline is ready!")
        print()
        print("Next steps:")
        print("  1. Run training:")
        print("     python python/scripts/train_profile_aware.py --profile hybrid-realistic --epochs 150")
        print("  2. Monitor coverage in outputs/*/coverage_trajectory.csv")
        print("  3. Generate synthetic samples:")
        print("     python python/scripts/generate_synthetic.py --num-samples 1000")
        print()
        print("=" * 70)
        
        return 0
        
    except ValidationError as e:
        print_section("VALIDATION FAILED")
        print(f"❌ Error: {e}")
        print()
        print("Fix the issues above before training.")
        print("=" * 70)
        return 1
    
    except Exception as e:
        print_section("UNEXPECTED ERROR")
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
