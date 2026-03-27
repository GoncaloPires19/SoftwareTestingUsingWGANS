"""
Data Validation and Integrity Checking Module

Ensures data extraction, cleaning, and pipeline integrity:
- Validates CSV structure (must have exactly 3 columns)
- Detects and reports data quality issues
- Identifies outliers and corruptions
- Generates validation reports
- Logs all statistics for reproducibility
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _json_default(value):
    """JSON serializer for numpy/pandas objects used in validation reports."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.dtype):
        return str(value)
    return str(value)


class DataValidator:
    """
    Comprehensive data validation and integrity checking.
    
    Purpose: Ensure data reliability throughout the pipeline
    - Input CSV must have exactly 3 columns (methods, branches, statements)
    - No missing or infinite values
    - Reasonable ranges for each feature
    - Data consistency across processing stages
    """
    
    # Expected feature names and typical ranges based on software testing
    EXPECTED_COLUMNS = 3
    FEATURE_NAMES = ['methods', 'branches', 'statements']
    
    # Reasonable ranges (but will be computed from actual data)
    # These are SANITY CHECKS only - actual min/max computed from data
    EXPECTED_RANGES = {
        'methods': (0, 10000),           # Methods per test
        'branches': (0, 50000),          # Code branches per test
        'statements': (0, 50000),        # Statements per test
    }
    
    # Outlier thresholds (IQR-based)
    OUTLIER_IQR_MULTIPLIER = 3.0  # Flag values beyond 3*IQR as outliers
    
    def __init__(self, validation_log_dir: Path = Path("logs/validation")):
        """Initialize validator with output directory"""
        self.validation_log_dir = Path(validation_log_dir)
        self.validation_log_dir.mkdir(parents=True, exist_ok=True)
        self.validation_report = {}
    
    def validate_csv_structure(self, csv_path: Path) -> Tuple[bool, Dict]:
        """
        Validate that CSV has correct structure (3 columns, proper values).
        
        Returns:
            (is_valid, report_dict)
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 VALIDATING CSV STRUCTURE: {csv_path.name}")
        logger.info(f"{'='*70}")
        
        report = {
            'file': str(csv_path),
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Try reading with error handling for malformed CSVs
            df = pd.read_csv(csv_path, header=None, on_bad_lines='skip', engine='python')
        except Exception as e:
            report['valid'] = False
            report['errors'].append(f"CSV parsing failed: {str(e)}")
            logger.error(f"❌ CSV parsing failed: {e}")
            logger.error(f"   (Trying with on_bad_lines='skip' to recover)")
            
            # Final attempt with skip_bad_lines
            try:
                df = pd.read_csv(csv_path, header=None, on_bad_lines='skip', engine='python')
                logger.info(f"   Recovered {len(df)} valid rows with on_bad_lines='skip'")
            except Exception as e2:
                logger.error(f"❌ Recovery failed: {e2}")
                return False, report
            return False, report
        
        logger.info(f"   Rows read: {len(df)}")
        logger.info(f"   Columns in file: {df.shape[1]}")
        
        # Check consistent column count
        if df.shape[1] != self.EXPECTED_COLUMNS:
            report['valid'] = False
            report['errors'].append(
                f"Expected {self.EXPECTED_COLUMNS} columns, got {df.shape[1]}"
            )
            logger.error(f"❌ Column mismatch: Expected 3, got {df.shape[1]}")
            
            # Analyze column distribution
            col_counts = df.apply(lambda x: x.notna().sum(), axis=1)
            logger.warning(f"   Column count distribution:")
            for count, freq in col_counts.value_counts().items():
                logger.warning(f"      {count} columns: {freq} rows")
            return False, report
        
        logger.info(f"   ✓ Column count correct (3 columns)")
        
        # Check for NaN/Inf
        nan_count = df.isna().sum().sum()
        inf_count = np.isinf(df.values).sum()
        
        if nan_count > 0:
            report['warnings'].append(f"Found {nan_count} NaN values")
            logger.warning(f"   ⚠ Found {nan_count} NaN values")
        
        if inf_count > 0:
            report['warnings'].append(f"Found {inf_count} Inf values")
            logger.warning(f"   ⚠ Found {inf_count} Inf values")
        
        # Collect statistics
        stats = {
            'total_rows': len(df),
            'total_columns': df.shape[1],
            'nan_count': int(nan_count),
            'inf_count': int(inf_count),
            'data_types': df.dtypes.to_dict(),
            'features': {}
        }
        
        # Per-feature statistics
        for i, col in enumerate(df.columns):
            feature_name = self.FEATURE_NAMES[i] if i < len(self.FEATURE_NAMES) else f"feature_{i}"
            feature_stats = self._get_feature_stats(df[col], feature_name)
            stats['features'][feature_name] = feature_stats
            
            logger.info(f"\n   Feature {i} ({feature_name}):")
            logger.info(f"      Min: {feature_stats['min']}")
            logger.info(f"      Max: {feature_stats['max']}")
            logger.info(f"      Mean: {feature_stats['mean']:.2f}")
            logger.info(f"      Std: {feature_stats['std']:.2f}")
            logger.info(f"      Outliers (>3*IQR): {feature_stats['outlier_count']}")
        
        report['statistics'] = stats
        
        if report['valid']:
            logger.info(f"\n✅ CSV structure validation PASSED")
        else:
            logger.error(f"\n❌ CSV structure validation FAILED")
        
        return report['valid'], report
    
    def _get_feature_stats(self, series: pd.Series, name: str) -> Dict:
        """Compute detailed statistics for a feature"""
        valid_data = series.dropna()
        
        if len(valid_data) == 0:
            return {
                'name': name,
                'count': 0,
                'min': None,
                'max': None,
                'mean': None,
                'std': None,
                'q1': None,
                'median': None,
                'q3': None,
                'iqr': None,
                'outlier_count': 0,
                'outlier_indices': []
            }
        
        # Calculate quartiles for outlier detection
        q1 = valid_data.quantile(0.25)
        q3 = valid_data.quantile(0.75)
        iqr = q3 - q1
        
        # Identify outliers
        lower_bound = q1 - self.OUTLIER_IQR_MULTIPLIER * iqr
        upper_bound = q3 + self.OUTLIER_IQR_MULTIPLIER * iqr
        outliers = (valid_data < lower_bound) | (valid_data > upper_bound)
        outlier_indices = valid_data[outliers].index.tolist()
        
        return {
            'name': name,
            'count': len(valid_data),
            'min': float(valid_data.min()),
            'max': float(valid_data.max()),
            'mean': float(valid_data.mean()),
            'std': float(valid_data.std()),
            'q1': float(q1),
            'median': float(valid_data.median()),
            'q3': float(q3),
            'iqr': float(iqr),
            'outlier_count': int(outliers.sum()),
            'outlier_indices': outlier_indices[:10],  # First 10
            'outlier_values': [float(v) for v in valid_data[outliers].head(10).values]
        }
    
    def detect_data_corruption(self, csv_path: Path) -> Dict:
        """
        Detect common data corruption patterns.
        
        Checks for:
        - Integer overflow values (±2^31, ±2^63)
        - Uninitialized variables (extreme outliers)
        - Suspicious value patterns
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🔎 CHECKING FOR DATA CORRUPTION: {csv_path.name}")
        logger.info(f"{'='*70}")
        
        try:
            # Try reading with error handling
            df = pd.read_csv(csv_path, header=None, on_bad_lines='skip', engine='python')
        except Exception as e:
            logger.error(f"Cannot read {csv_path}: {e}")
            return {
                'file': str(csv_path),
                'corruptions_found': [f"CSV parsing error: {str(e)}"],
                'severity': 'CRITICAL'
            }
        
        corruption_report = {
            'file': str(csv_path),
            'corruptions_found': [],
            'severity': 'OK'
        }
        
        # Check for integer overflow signatures
        INT32_MAX = 2147483647
        INT32_MIN = -2147483648
        INT64_MAX = 9223372036854775807
        
        all_values = df.values.flatten()
        all_values = all_values[~np.isnan(all_values)]  # Remove NaN
        
        # Integer overflow checks
        if np.any(np.abs(all_values) == INT32_MAX) or np.any(np.abs(all_values) == INT32_MIN):
            corruption = {
                'type': 'Integer Overflow (32-bit)',
                'severity': 'CRITICAL',
                'description': 'Data contains 32-bit integer overflow values',
                'count': int(np.sum((np.abs(all_values) == INT32_MAX) | (np.abs(all_values) == INT32_MIN)))
            }
            corruption_report['corruptions_found'].append(corruption)
            corruption_report['severity'] = 'CRITICAL'
            logger.error(f"❌ CRITICAL: Integer overflow detected!")
        
        # Extreme outliers (>1e9)
        extreme_outliers = np.abs(all_values) > 1e9
        if np.any(extreme_outliers):
            corruption = {
                'type': 'Extreme Outliers',
                'severity': 'CRITICAL',
                'description': 'Data contains extreme values (>1e9)',
                'count': int(np.sum(extreme_outliers)),
                'sample_values': [float(v) for v in all_values[extreme_outliers][:5]]
            }
            corruption_report['corruptions_found'].append(corruption)
            if corruption_report['severity'] != 'CRITICAL':
                corruption_report['severity'] = 'MAJOR'
            logger.error(f"❌ MAJOR: {corruption['count']} extreme outliers found")
        
        # Zero variance (all same value)
        for i, col in enumerate(df.columns):
            if df[col].std() == 0 and len(df[col].unique()) == 1:
                corruption = {
                    'type': 'Zero Variance Column',
                    'severity': 'MAJOR',
                    'description': f'Column {i} has constant value',
                    'value': float(df[col].iloc[0])
                }
                corruption_report['corruptions_found'].append(corruption)
                if corruption_report['severity'] != 'CRITICAL':
                    corruption_report['severity'] = 'MAJOR'
                logger.warning(f"⚠ Column {i} has zero variance (value={df[col].iloc[0]})")
        
        if not corruption_report['corruptions_found']:
            logger.info(f"✅ No data corruption detected")
        
        return corruption_report
    
    def save_validation_report(self, report: Dict, filename: str = "validation_report.json"):
        """Save validation report to JSON"""
        report_path = self.validation_log_dir / filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=_json_default)
        logger.info(f"📄 Report saved: {report_path}")
        return report_path
    
    def validate_and_report(self, csv_path: Path) -> Tuple[bool, Path]:
        """
        Run complete validation and save report.
        
        Returns:
            (is_valid, report_path)
        """
        is_valid, structure_report = self.validate_csv_structure(csv_path)
        corruption_report = self.detect_data_corruption(csv_path)
        
        # Combine reports
        full_report = {
            'timestamp': str(pd.Timestamp.now()),
            'file': str(csv_path),
            'structure_validation': structure_report,
            'corruption_check': corruption_report,
            'overall_valid': is_valid and corruption_report['severity'] == 'OK'
        }
        
        report_path = self.save_validation_report(full_report)
        
        logger.info(f"\n{'='*70}")
        if full_report['overall_valid']:
            logger.info(f"✅ VALIDATION PASSED - Data is reliable")
        else:
            logger.error(f"❌ VALIDATION FAILED - Data integrity issues detected")
        logger.info(f"{'='*70}\n")
        
        return full_report['overall_valid'], report_path


if __name__ == "__main__":
    logger.info("✅ Data validation module loaded")
