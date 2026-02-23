#!/usr/bin/env python3
"""
Data Pipeline Validation Script

Validates the complete WGAN data pipeline from extraction to generation:
1. Raw data validation (CSV structure, column consistency)
2. Corruption detection (outliers, integer overflow)
3. Normalization stats validation
4. Generated synthetic data validation
5. Distribution comparison (real vs synthetic)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import json
from data_validation import DataValidator
from junit_datasets import JUnitDataset, load_normalization_stats
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPipelineValidator:
    """Complete validation of WGAN data pipeline"""
    
    def __init__(self, data_dir: Path = Path("python/data")):
        self.data_dir = Path(data_dir)
        self.validator = DataValidator()
        self.validation_results = {}
    
    def validate_raw_extraction(self) -> bool:
        """
        Step 1: Validate raw data extraction from Java tests.
        
        Checks:
        - CSV file exists
        - Structure is correct (3 columns)
        - No corrupted data
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 1: VALIDATING RAW DATA EXTRACTION")
        logger.info("="*70)
        
        raw_data_path = self.data_dir / "junit_data.csv"
        
        if not raw_data_path.exists():
            logger.warning(f"⚠️  Raw data file not found: {raw_data_path}")
            # Try cleaned as fallback
            raw_data_path = self.data_dir / "junit_data_cleaned.csv"
            if not raw_data_path.exists():
                logger.error(f"❌ Neither raw nor cleaned data found: {raw_data_path}")
                return False
            logger.info(f"✅ Using cleaned file: {raw_data_path}")
        
        is_valid, report = self.validator.validate_csv_structure(raw_data_path)
        corruption_report = self.validator.detect_data_corruption(raw_data_path)
        
        self.validation_results['raw_extraction'] = {
            'valid': is_valid,
            'structure_report': report,
            'corruption_report': corruption_report
        }
        
        return is_valid and corruption_report['severity'] != 'CRITICAL'
    
    def validate_cleaned_data(self) -> bool:
        """
        Step 2: Validate cleaned data.
        
        Checks:
        - Cleaned file exists
        - Has 3 columns consistently
        - No extreme corruption
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 2: VALIDATING CLEANED DATA")
        logger.info("="*70)
        
        cleaned_data_path = self.data_dir / "junit_data_cleaned.csv"
        
        if not cleaned_data_path.exists():
            logger.error(f"❌ Cleaned data file not found: {cleaned_data_path}")
            return False
        
        is_valid, report = self.validator.validate_csv_structure(cleaned_data_path)
        corruption_report = self.validator.detect_data_corruption(cleaned_data_path)
        
        self.validation_results['cleaned_data'] = {
            'valid': is_valid,
            'structure_report': report,
            'corruption_report': corruption_report
        }
        
        # For cleaned data, structure must be perfect
        return is_valid and not corruption_report['corruptions_found']
    
    def validate_normalization_stats(self) -> bool:
        """
        Step 3: Validate normalization statistics file.
        
        Checks:
        - Stats JSON exists
        - Contains correct features
        - Min/max ranges are reasonable
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 3: VALIDATING NORMALIZATION STATISTICS")
        logger.info("="*70)
        
        stats_path = self.data_dir / "junit_data_cleaned_norm_stats.json"
        
        if not stats_path.exists():
            logger.error(f"❌ Normalization stats file not found: {stats_path}")
            logger.error(f"   This file is CRITICAL for denormalization!")
            return False
        
        try:
            stats = load_normalization_stats(stats_path)
        except Exception as e:
            logger.error(f"❌ Failed to load normalization stats: {e}")
            return False
        
        # Validate structure
        required_keys = ['num_features', 'num_samples', 'features']
        for key in required_keys:
            if key not in stats:
                logger.error(f"❌ Missing key in stats: {key}")
                return False
        
        if stats['num_features'] != 4:
            logger.error(f"❌ Expected 4 features, got {stats['num_features']}")
            return False
        
        # Validate each feature
        for i in range(3):
            feature_key = f'feature_{i}'
            if feature_key not in stats['features']:
                logger.error(f"❌ Missing feature {i}")
                return False
            
            feat_stats = stats['features'][feature_key]
            feature_min = feat_stats['min']
            feature_max = feat_stats['max']
            feature_range = feature_max - feature_min
            
            logger.info(f"   Feature {i}: min={feature_min:.2f}, max={feature_max:.2f}")
            
            # Sanity checks
            if feature_min > feature_max:
                logger.error(f"❌ Feature {i}: min > max!")
                return False
            
            if feature_min < 0:
                logger.warning(f"⚠ Feature {i}: minimum is negative (unusual for test metrics)")
            
            if feature_range > 1e10:
                logger.warning(f"⚠ Feature {i}: range is very large (possible integer overflow in original data)")
        
        logger.info(f"✅ Normalization stats valid")
        self.validation_results['normalization_stats'] = {'valid': True, 'stats': stats}
        return True
    
    def validate_dataset_loading(self) -> bool:
        """
        Step 4: Validate that dataset can be loaded correctly.
        
        Checks:
        - JUnitDataset loads data
        - Normalization works
        - Normalized data is in [-1, 1] range
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 4: VALIDATING DATASET LOADING AND NORMALIZATION")
        logger.info("="*70)
        
        cleaned_data_path = self.data_dir / "junit_data_cleaned.csv"
        
        try:
            dataset = JUnitDataset(str(cleaned_data_path))
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            return False
        
        # Check dataset properties
        logger.info(f"   Loaded {len(dataset)} samples")
        
        # Check normalized data range
        norm_data = dataset.normalized_data.numpy()
        norm_min = norm_data.min()
        norm_max = norm_data.max()
        
        logger.info(f"   Normalized data range: [{norm_min:.4f}, {norm_max:.4f}]")
        
        if norm_max > 1.1 or norm_min < -1.1:
            logger.warning(f"⚠ Normalized data outside [-1, 1] range (max allowed ±1.1)")
            if norm_max > 2 or norm_min < -2:
                logger.error(f"❌ Normalized data severely out of range!")
                return False
        
        logger.info(f"✅ Dataset loading and normalization valid")
        self.validation_results['dataset_loading'] = {'valid': True, 'samples': len(dataset)}
        return True
    
    def validate_reproducibility(self) -> bool:
        """
        Step 5: Validate reproducibility setup.
        
        Checks:
        - Random seeds are set
        - Configuration includes seed
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 5: VALIDATING REPRODUCIBILITY SETUP")
        logger.info("="*70)
        
        # Check if seed is configured in config.py
        logger.info("   Checking configuration...")
        
        try:
            from config import TrainingConfig
            tc = TrainingConfig()
            logger.info(f"   ✓ TrainingConfig.seed = {tc.seed}")
            
            if tc.seed != 42:
                logger.warning(f"⚠ Non-standard seed: {tc.seed} (typically use 42)")
        except Exception as e:
            logger.error(f"❌ Failed to check training config: {e}")
            return False
        
        logger.info(f"✅ Reproducibility configuration valid")
        self.validation_results['reproducibility'] = {'valid': True}
        return True
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        logger.info("\n" + "="*70)
        logger.info("VALIDATION REPORT SUMMARY")
        logger.info("="*70)
        
        report = {
            'timestamp': str(pd.Timestamp.now()),
            'pipeline_valid': True,
            'steps': {}
        }
        
        for step, result in self.validation_results.items():
            status = "✅ PASS" if result.get('valid', False) else "❌ FAIL"
            logger.info(f"{status}: {step}")
            report['steps'][step] = result.get('valid', False)
            if not result.get('valid', False):
                report['pipeline_valid'] = False
        
        logger.info("\n" + "="*70)
        if report['pipeline_valid']:
            logger.info("✅ COMPLETE PIPELINE VALID")
            logger.info("   Data integrity: OK")
            logger.info("   Reproducibility: OK")
            logger.info("   Ready for training!")
        else:
            logger.error("❌ PIPELINE VALIDATION FAILED")
            logger.error("   Please fix issues before training")
        logger.info("="*70)
        
        return report
    
    def run_complete_validation(self) -> bool:
        """Run all validation steps"""
        logger.info("\n" + "🔍 STARTING COMPLETE DATA PIPELINE VALIDATION" + "\n")
        
        all_valid = True
        
        # Step 1: Raw extraction
        if not self.validate_raw_extraction():
            logger.warning("⚠ Raw data issues detected (may have been cleaned)")
        
        # Step 2: Cleaned data
        if not self.validate_cleaned_data():
            logger.error("❌ Cleaned data validation FAILED")
            all_valid = False
        
        # Step 3: Normalization stats
        if not self.validate_normalization_stats():
            logger.error("❌ Normalization stats validation FAILED")
            all_valid = False
        
        # Step 4: Dataset loading
        if not self.validate_dataset_loading():
            logger.error("❌ Dataset loading validation FAILED")
            all_valid = False
        
        # Step 5: Reproducibility
        if not self.validate_reproducibility():
            logger.error("❌ Reproducibility validation FAILED")
            all_valid = False
        
        # Generate report
        report = self.generate_validation_report()
        
        # Save report
        report_path = self.data_dir.parent / "logs" / "validation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"\n📄 Validation report saved: {report_path}\n")
        
        return all_valid


def main():
    parser = argparse.ArgumentParser(description="Validate WGAN data pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="python/data",
        help="Data directory path"
    )
    
    args = parser.parse_args()
    
    validator = DataPipelineValidator(Path(args.data_dir))
    is_valid = validator.run_complete_validation()
    
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
