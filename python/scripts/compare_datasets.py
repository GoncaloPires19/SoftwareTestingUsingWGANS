"""
Compare real vs synthetic test data distributions
Analyze diversity, coverage, and quality metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComparativeAnalysis:
    """Compare real and synthetic test distributions"""
    
    def __init__(self, real_data_path: Path, synthetic_data_path: Path):
        self.real_data = pd.read_csv(real_data_path)
        self.synthetic_data = pd.read_csv(synthetic_data_path)
    
    def statistical_comparison(self) -> Dict:
        """
        Compare statistical properties
        """
        logger.info("\n📊 STATISTICAL COMPARISON")
        logger.info("=" * 70)
        
        comparison = {}
        
        for col in ['methods', 'branches', 'statements']:
            real = self.real_data[col]
            synth = self.synthetic_data[col]
            
            comparison[col] = {
                'real_mean': real.mean(),
                'synth_mean': synth.mean(),
                'real_std': real.std(),
                'synth_std': synth.std(),
                'real_min': real.min(),
                'synth_min': synth.min(),
                'real_max': real.max(),
                'synth_max': synth.max(),
                'mean_diff': abs(real.mean() - synth.mean()),
                'std_diff': abs(real.std() - synth.std()),
            }
            
            logger.info(f"\n{col.upper()}:")
            logger.info(f"  Real:      {comparison[col]['real_mean']:8.2f} ± {comparison[col]['real_std']:8.2f}")
            logger.info(f"  Synthetic: {comparison[col]['synth_mean']:8.2f} ± {comparison[col]['synth_std']:8.2f}")
            logger.info(f"  Difference: {comparison[col]['mean_diff']:8.2f} ({comparison[col]['mean_diff']/comparison[col]['real_mean']*100:.1f}%)")
        
        return comparison
    
    def diversity_metrics(self) -> Dict:
        """
        Calculate diversity metrics
        """
        logger.info("\n🎯 DIVERSITY METRICS")
        logger.info("=" * 70)
        
        metrics = {}
        
        # Entropy (measure of randomness)
        for name, data in [('Real', self.real_data), ('Synthetic', self.synthetic_data)]:
            entropies = []
            
            for col in ['methods', 'branches', 'statements']:
                # Normalize to [0,1]
                normalized = (data[col] - data[col].min()) / (data[col].max() - data[col].min() + 1e-8)
                
                # Histogram-based entropy
                hist, _ = np.histogram(normalized, bins=20, range=(0, 1))
                hist = hist / hist.sum()
                entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
                entropies.append(entropy)
            
            metrics[name] = {
                'mean_entropy': np.mean(entropies),
                'entropy_by_feature': {
                    'methods': entropies[0],
                    'branches': entropies[1],
                    'statements': entropies[2]
                }
            }
            
            logger.info(f"\n{name}:")
            logger.info(f"  Average entropy: {metrics[name]['mean_entropy']:.4f}")
            logger.info(f"  Methods entropy: {entropies[0]:.4f}")
            logger.info(f"  Branches entropy: {entropies[1]:.4f}")
            logger.info(f"  Statements entropy: {entropies[2]:.4f}")
        
        return metrics
    
    def coverage_analysis(self) -> Dict:
        """
        Analyze feature coverage and distribution
        """
        logger.info("\n📈 COVERAGE ANALYSIS")
        logger.info("=" * 70)
        
        coverage = {}
        
        for col in ['methods', 'branches', 'statements']:
            real_range = (self.real_data[col].min(), self.real_data[col].max())
            synth_range = (self.synthetic_data[col].min(), self.synthetic_data[col].max())
            
            # Coverage: what % of real range is covered by synthetic
            overlap_min = max(real_range[0], synth_range[0])
            overlap_max = min(real_range[1], synth_range[1])
            
            if overlap_max > overlap_min:
                overlap = overlap_max - overlap_min
                real_span = real_range[1] - real_range[0]
                coverage_pct = (overlap / real_span) * 100
            else:
                coverage_pct = 0
            
            coverage[col] = {
                'real_range': real_range,
                'synth_range': synth_range,
                'coverage_pct': coverage_pct
            }
            
            logger.info(f"\n{col.upper()}:")
            logger.info(f"  Real range:       [{real_range[0]:.0f}, {real_range[1]:.0f}]")
            logger.info(f"  Synthetic range:  [{synth_range[0]:.0f}, {synth_range[1]:.0f}]")
            logger.info(f"  Coverage:         {coverage_pct:.1f}%")
        
        return coverage
    
    def sample_quality(self) -> Dict:
        """
        Assess synthetic sample quality
        """
        logger.info("\n⭐ SAMPLE QUALITY METRICS")
        logger.info("=" * 70)
        
        quality = {}
        
        # Check for outliers and anomalies
        synth = self.synthetic_data
        
        for col in ['methods', 'branches', 'statements']:
            # Calculate Z-scores
            mean = synth[col].mean()
            std = synth[col].std()
            z_scores = np.abs((synth[col] - mean) / (std + 1e-8))
            
            # Outliers: |z| > 3
            outliers = (z_scores > 3).sum()
            outlier_pct = (outliers / len(synth)) * 100
            
            # Validity (within reasonable ranges)
            valid = ((synth[col] > 0) & (synth[col] < 10000)).sum()
            validity_pct = (valid / len(synth)) * 100
            
            quality[col] = {
                'outliers': outliers,
                'outlier_pct': outlier_pct,
                'valid_samples': valid,
                'validity_pct': validity_pct
            }
            
            logger.info(f"\n{col.upper()}:")
            logger.info(f"  Valid samples: {valid:,} ({validity_pct:.1f}%)")
            logger.info(f"  Outliers: {outliers:,} ({outlier_pct:.2f}%)")
        
        return quality
    
    def generate_report(self) -> str:
        """
        Generate comprehensive comparison report
        """
        report = []
        
        report.append("\n" + "=" * 70)
        report.append("SYNTHETIC TEST DATA - QUALITY REPORT")
        report.append("=" * 70)
        
        report.append(f"\nDataset Sizes:")
        report.append(f"  Real samples:      {len(self.real_data):,}")
        report.append(f"  Synthetic samples: {len(self.synthetic_data):,}")
        
        # Run all analyses
        stats = self.statistical_comparison()
        diversity = self.diversity_metrics()
        coverage = self.coverage_analysis()
        quality = self.sample_quality()
        
        # Summary scores
        report.append("\n" + "=" * 70)
        report.append("OVERALL ASSESSMENT")
        report.append("=" * 70)
        
        # Calculate mean difference across features
        mean_diffs = [stats[col]['mean_diff'] / max(stats[col]['real_mean'], 1) for col in ['methods', 'branches', 'statements']]
        avg_diff = np.mean(mean_diffs) * 100
        
        report.append(f"\nStatistical Similarity: {100 - min(avg_diff, 100):.1f}%")
        
        # Coverage score
        coverage_scores = [coverage[col]['coverage_pct'] for col in ['methods', 'branches', 'statements']]
        avg_coverage = np.mean(coverage_scores)
        
        report.append(f"Feature Coverage: {avg_coverage:.1f}%")
        
        # Quality score
        quality_scores = [quality[col]['validity_pct'] for col in ['methods', 'branches', 'statements']]
        avg_quality = np.mean(quality_scores)
        
        report.append(f"Sample Quality: {avg_quality:.1f}%")
        
        # Final score
        final_score = (100 - min(avg_diff, 100) + avg_coverage + avg_quality) / 3
        report.append(f"\nFINAL QUALITY SCORE: {final_score:.1f}/100")
        
        # Recommendation
        if final_score >= 80:
            report.append("✅ EXCELLENT - Ready for production use")
        elif final_score >= 70:
            report.append("✅ GOOD - Suitable for testing")
        elif final_score >= 60:
            report.append("⚠️ ACCEPTABLE - Can be used with caution")
        else:
            report.append("❌ POOR - Requires retraining")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def save_report(self, output_path: Path = Path("outputs/comparison_report.txt")):
        """Save report to file"""
        report = self.generate_report()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"\n💾 Report saved to {output_path}")
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare real vs synthetic test data")
    parser.add_argument("--real", type=Path, default=Path("python/data/junit_data_cleaned.csv"),
                       help="Path to real test data")
    parser.add_argument("--synthetic", type=Path, default=Path("python/outputs/synthetic_10k.csv"),
                       help="Path to synthetic data")
    parser.add_argument("--output", type=Path, default=Path("python/outputs/comparison_report.txt"),
                       help="Output report path")
    
    args = parser.parse_args()
    
    analysis = ComparativeAnalysis(args.real, args.synthetic)
    
    report = analysis.generate_report()
    print(report)
    
    analysis.save_report(args.output)


if __name__ == "__main__":
    main()
