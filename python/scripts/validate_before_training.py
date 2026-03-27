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
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import torch


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


def validate_dataset(data_path: Path, min_samples: int, max_value_threshold: float, min_value_threshold: float):
    """Validate dataset quality"""
    print_section("1. DATASET VALIDATION")
    
    # Check file exists
    if not check_file_exists(data_path, "Cleaned dataset"):
        raise ValidationError(f"Dataset not found: {data_path}")
    
    # Load data
    print(f"\n📂 Loading dataset...")
    df = pd.read_csv(data_path, header=None, names=FEATURE_NAMES)
    print(f"   Loaded {len(df)} samples with {df.shape[1]} features")
    
    # Check minimum samples
    if len(df) < min_samples:
        raise ValidationError(f"Insufficient samples: {len(df)} < {min_samples}")
    print(f"✅ Sufficient samples: {len(df)} >= {min_samples}")
    
    # Check for negative values
    negatives = (df < min_value_threshold).sum()
    if negatives.any():
        print(f"❌ Dataset contains negative values:")
        for col, count in negatives.items():
            if count > 0:
                print(f"   {col}: {count} negative values")
        raise ValidationError("Dataset contains negative values")
    print(f"✅ No negative values")
    
    # Check for outliers
    outliers = (df > max_value_threshold).sum()
    if outliers.any():
        print(f"❌ Dataset contains outliers (>{max_value_threshold}):")
        for col, count in outliers.items():
            if count > 0:
                print(f"   {col}: {count} outliers")
        raise ValidationError(f"Dataset contains outliers > {max_value_threshold}")
    print(f"✅ No outliers (all values <= {max_value_threshold})")
    
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


def validate_normalization_stats(norm_stats_path: Path, max_value_threshold: float, min_value_threshold: float):
    """Validate normalization statistics"""
    print_section("2. NORMALIZATION STATS VALIDATION")
    
    # Check file exists
    if not check_file_exists(norm_stats_path, "Normalization stats"):
        raise ValidationError(f"Normalization stats not found: {norm_stats_path}")
    
    # Load stats
    print(f"\n📂 Loading normalization stats...")
    with open(norm_stats_path, 'r', encoding='utf-8') as f:
        norm_stats = json.load(f)

    # Support both legacy schema and current JUnitDataset schema.
    if "features" in norm_stats:
        max_values = [norm_stats["features"][f"feature_{i}"]["max"] for i in range(len(FEATURE_NAMES))]
        min_values = [norm_stats["features"][f"feature_{i}"]["min"] for i in range(len(FEATURE_NAMES))]
        print("✅ Detected current norm stats schema (features/feature_i)")
    else:
        required_keys = ['min', 'max']
        missing = [k for k in required_keys if k not in norm_stats]
        if missing:
            raise ValidationError(f"Missing keys in norm_stats: {missing}")
        max_values = norm_stats['max']
        min_values = norm_stats['min']
        print("✅ Detected legacy norm stats schema (min/max arrays)")

    # Check values are reasonable (not contaminated by outliers)
    print(f"\n📊 Max values in normalization stats:")
    for i, (feature, max_val) in enumerate(zip(FEATURE_NAMES, max_values)):
        print(f"   {feature}: {max_val}")
        if max_val > max_value_threshold * 1.1:  # Allow 10% buffer
            raise ValidationError(
                f"Normalization max for {feature} is too high: {max_val} > {max_value_threshold}"
            )
    print(f"✅ All max values are reasonable (<= {max_value_threshold * 1.1})")
    
    # Check min values are non-negative
    for i, (feature, min_val) in enumerate(zip(FEATURE_NAMES, min_values)):
        if min_val < min_value_threshold:
            raise ValidationError(f"Normalization min for {feature} is negative: {min_val}")
    print(f"✅ All min values are non-negative")
    
    return norm_stats


def validate_cleaning_report(cleaning_report: Path):
    """Validate cleaning report exists and shows good results"""
    print_section("3. CLEANING REPORT VALIDATION")
    
    if not check_file_exists(cleaning_report, "Cleaning report"):
        print(f"⚠️  Cleaning report not found (optional)")
        return None
    
    # Load report
    with open(cleaning_report, 'r', encoding='utf-8') as f:
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


def validate_environment(models_dir: Path):
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
    if not models_dir.exists():
        print(f"⚠️  Models directory doesn't exist, creating: {models_dir}")
        models_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Check if there are existing models
        existing_models = list(models_dir.glob("*.pth"))
        if existing_models:
            print(f"⚠️  Found {len(existing_models)} existing model files:")
            for model in existing_models[:5]:  # Show first 5
                print(f"      {model.name}")
            if len(existing_models) > 5:
                print(f"      ... and {len(existing_models) - 5} more")
            print(f"   Consider backing up or moving old models before training")
    print(f"✅ Models directory exists: {models_dir}")
    
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
    parser = argparse.ArgumentParser(description="Validate dataset and environment before training")
    parser.add_argument("--data-path", type=str, default="python/data/junit_data_cleaned_strict.csv")
    parser.add_argument("--norm-stats-path", type=str, default=None)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--max-value-threshold", type=float, default=1000)
    parser.add_argument("--min-value-threshold", type=float, default=0)
    parser.add_argument("--models-dir", type=str, default="python/models")
    parser.add_argument("--cleaning-report", type=str, default="python/data/cleaning_report.json")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if args.norm_stats_path:
        norm_stats_path = Path(args.norm_stats_path)
    else:
        norm_stats_path = data_path.parent / f"{data_path.stem}_norm_stats.json"

    models_dir = Path(args.models_dir)
    cleaning_report = Path(args.cleaning_report)

    print("=" * 70)
    print("PRE-TRAINING VALIDATION")
    print("=" * 70)
    print("This script validates the training pipeline is ready.")
    print("Run this BEFORE starting WGAN training.")
    
    try:
        # Run validations
        df = validate_dataset(
            data_path=data_path,
            min_samples=args.min_samples,
            max_value_threshold=args.max_value_threshold,
            min_value_threshold=args.min_value_threshold,
        )
        norm_stats = validate_normalization_stats(
            norm_stats_path=norm_stats_path,
            max_value_threshold=args.max_value_threshold,
            min_value_threshold=args.min_value_threshold,
        )
        report = validate_cleaning_report(cleaning_report)
        validate_environment(models_dir)
        validate_synthetic_generation_config()
        
        # Final summary
        print_section("VALIDATION SUMMARY")
        print("✅ ALL VALIDATIONS PASSED")
        print()
        print("Your training pipeline is ready!")
        print()
        print("Next steps:")
        print("  1. Run training:")
        print("     python python/scripts/train_basic_profiles.py --profile balanced --data-path <your_csv>")
        print("  2. Monitor coverage in outputs/*/training_history.csv")
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
