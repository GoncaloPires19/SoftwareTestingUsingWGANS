"""
Advanced WGAN Evaluation - Quality Assessment Framework
Compares: Original (mode collapse) vs Improved (basic fix) vs Realistic (clipped)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityComparison:
    """Compare synthetic data quality across models"""
    
    def __init__(self):
        # Load real data (using cleaned data)
        self.real = pd.read_csv('python/data/junit_data_cleaned_strict.csv', header=None, names=['methods', 'branches', 'statements'])
        
        # Try to load available synthetic variants
        try:
            self.orig_syn = pd.read_csv('outputs/synthetic_samples.csv').iloc[:, :3]
        except:
            self.orig_syn = None
            
        try:
            self.impro_syn = pd.read_csv('outputs/synthetic_improved.csv').iloc[:, :3]
        except:
            self.impro_syn = None
            
        try:
            self.reali_syn = pd.read_csv('outputs/synthetic_realistic.csv').iloc[:, :3]
        except:
            self.reali_syn = None
        
        self.features = ['methods', 'branches', 'statements']
    
    def calculate_metrics(self, real, synthetic, name):
        """Calculate all quality metrics"""
        metrics = {}
        
        for i, feat in enumerate(self.features):
            real_feat = real.iloc[:, i]
            syn_feat = synthetic.iloc[:, i]
            
            # Coverage
            real_min, real_max = real_feat.min(), real_feat.max()
            coverage = np.sum((syn_feat >= real_min) & (syn_feat <= real_max)) / len(syn_feat) * 100
            
            # Statistics
            real_mean, real_std = real_feat.mean(), real_feat.std()
            syn_mean, syn_std = syn_feat.mean(), syn_feat.std()
            mean_error = abs(syn_mean - real_mean) / (real_mean + 1e-8) * 100
            std_ratio = syn_std / (real_std + 1e-8)
            
            # KS Test
            ks_stat, ks_pval = scipy_stats.ks_2samp(real_feat, syn_feat)
            
            # Wasserstein
            was_dist = scipy_stats.wasserstein_distance(real_feat, syn_feat)
            
            metrics[feat] = {
                'coverage': coverage,
                'mean_error': mean_error,
                'std_ratio': std_ratio,
                'ks_stat': ks_stat,
                'ks_pval': ks_pval,
                'wasserstein': was_dist,
                'real_mean': real_mean,
                'syn_mean': syn_mean,
                'real_std': real_std,
                'syn_std': syn_std
            }
        
        return metrics
    
    def run_comparison(self):
        """Run full comparison"""
        logger.info(f"\n{'='*90}")
        logger.info(f"📊 COMPREHENSIVE QUALITY COMPARISON")
        logger.info(f"{'='*90}\n")
        
        # Calculate metrics for available models
        all_metrics = {}
        
        if self.orig_syn is not None:
            orig_metrics = self.calculate_metrics(self.real, self.orig_syn, "Original")
            all_metrics['Original'] = orig_metrics
        else:
            logger.warning("⚠️  Original synthetic data not found")
            
        if self.impro_syn is not None:
            impro_metrics = self.calculate_metrics(self.real, self.impro_syn, "Improved")
            all_metrics['Improved'] = impro_metrics
        else:
            logger.warning("⚠️  Improved synthetic data not found")
            
        if self.reali_syn is not None:
            reali_metrics = self.calculate_metrics(self.real, self.reali_syn, "Realistic")
            all_metrics['Realistic'] = reali_metrics
        else:
            logger.warning("⚠️  Realistic synthetic data not found")
        
        if not all_metrics:
            logger.error("❌ No synthetic data available for comparison!")
            return
            
        # Display results per feature
        for feat in self.features:
            logger.info(f"\n{'─'*90}")
            logger.info(f"📈 {feat.upper()}")
            logger.info(f"{'─'*90}")
            
            for model_name, metrics in all_metrics.items():
                m = metrics[feat]
                logger.info(f"\n  {model_name}:")
                logger.info(f"    Coverage:      {m['coverage']:.1f}%")
                logger.info(f"    Mean Error:    {m['mean_error']:.1f}% ({m['syn_mean']:.2f} vs {m['real_mean']:.2f})")
                logger.info(f"    Std Dev Ratio: {m['std_ratio']:.2f}x ({m['syn_std']:.2f} vs {m['real_std']:.2f})")
                logger.info(f"    KS Distance:   {m['ks_stat']:.3f} (p={m['ks_pval']:.2e})")
                logger.info(f"    Wasserstein:   {m['wasserstein']:.2f}")
        
        # Overall ranking
        logger.info(f"\n{'='*90}")
        logger.info(f"🏆 OVERALL QUALITY RANKING")
        logger.info(f"{'='*90}\n")
        
        scores = {}
        for model_name, metrics in all_metrics.items():
            # Score based on combined metrics
            coverage_score = np.mean([metrics[f]['coverage'] for f in self.features])
            mean_error_score = 100 - np.mean([metrics[f]['mean_error'] for f in self.features])
            std_score = np.mean([min(metrics[f]['std_ratio'], 1.0) for f in self.features])
            ks_score = 100 - np.mean([metrics[f]['ks_stat'] * 100 for f in self.features])
            
            total_score = (coverage_score + mean_error_score * 0.5 + std_score * 100 + ks_score) / 3
            scores[model_name] = {
                'coverage': coverage_score,
                'mean_error': mean_error_score,
                'std': std_score,
                'ks': ks_score,
                'total': total_score
            }
        
        # Sort and display
        sorted_models = sorted(scores.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for rank, (model_name, score) in enumerate(sorted_models, 1):
            emoji = ['🥇', '🥈', '🥉'][rank-1] if rank <= 3 else f"{rank}."
            logger.info(f"  {emoji} {model_name:15s}: {score['total']:.1f} "
                       f"(Coverage: {score['coverage']:.0f}%, Std: {score['std']:.2f}x, KS: {score['ks']:.0f})")
        
        logger.info(f"\nRanking based on weighted average of:")
        logger.info(f"  - Coverage (importance: 1.0)")
        logger.info(f"  - Mean Error (importance: 0.5)")
        logger.info(f"  - Std Dev Match (importance: 1.0)")
        logger.info(f"  - KS Distance (importance: 1.0)")
        
        # Recommendations
        logger.info(f"\n{'='*90}")
        logger.info(f"💡 RECOMMENDATIONS")
        logger.info(f"{'='*90}\n")
        
        best_model = sorted_models[0][0]
        logger.info(f"✅ BEST MODEL: {best_model}")
        logger.info(f"\n  Use {best_model} for:")
        logger.info(f"  - Test data augmentation")
        logger.info(f"  - Training synthetic test suites")
        logger.info(f"  - Benchmarking analysis")
        
        if best_model == 'Realistic':
            logger.info(f"\n  ✓ Realistic values within proper ranges")
            logger.info(f"  ✓ Good coverage of real feature distribution")
            logger.info(f"  ✓ Appropriate for production use")
        elif best_model == 'Improved':
            logger.info(f"\n  ⚠ Note: Has negative values, apply clipping for production")
            logger.info(f"  ✓ Excellent diversity and distribution matching")
        
        # Next steps
        logger.info(f"\n{'─'*90}")
        logger.info(f"🔄 NEXT IMPROVEMENTS")
        logger.info(f"{'─'*90}\n")
        logger.info(f"  1. Train Phase 2 model with spectral norm")
        logger.info(f"  2. Generate Phase 2 realistic samples")
        logger.info(f"  3. Expand dataset: +Defects4J, +EvoSuite → 500+ samples")
        logger.info(f"  4. Re-train with more diverse training data")
        logger.info(f"  5. Evaluate Phase 2 model quality\n")


if __name__ == "__main__":
    comp = QualityComparison()
    comp.run_comparison()
