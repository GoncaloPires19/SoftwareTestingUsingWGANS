"""
Experimental Analysis: Training Validation & Overfitting Detection
Analyzes training history and compares real vs synthetic test distributions
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExperimentalAnalysis:
    """Comprehensive training analysis and overfitting detection"""
    
    def __init__(self, training_history_path, real_data_path, synthetic_data_path, output_dir='outputs/experimental'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        try:
            self.training_history = pd.read_csv(training_history_path)
            logger.info(f"✅ Loaded training history: {len(self.training_history)} epochs")
        except FileNotFoundError:
            logger.warning("⚠️  Training history not found")
            self.training_history = None
        
        try:
            self.real_data = pd.read_csv(real_data_path, header=None, names=['methods', 'branches', 'statements'])
            logger.info(f"✅ Loaded real test data: {len(self.real_data)} samples")
        except FileNotFoundError:
            logger.error("❌ Real test data not found")
            self.real_data = None
            
        try:
            raw_synthetic = pd.read_csv(synthetic_data_path, header=None)
            self.synthetic_data = raw_synthetic.iloc[:, :3].apply(pd.to_numeric, errors='coerce').dropna()
            self.synthetic_data.columns = ['methods', 'branches', 'statements']
            logger.info(f"✅ Loaded synthetic data: {len(self.synthetic_data)} samples")
        except FileNotFoundError:
            logger.warning("⚠️  Synthetic data not found")
            self.synthetic_data = None
        
        self.features = ['methods', 'branches', 'statements']
    
    def analyze_training_history(self):
        """Analyze training convergence and stability"""
        if self.training_history is None:
            logger.warning("⚠️  Cannot analyze training history - file not found")
            return {}
        
        logger.info("\n" + "="*90)
        logger.info("📚 TRAINING HISTORY ANALYSIS")
        logger.info("="*90)
        
        analysis = {
            'epochs': len(self.training_history),
            'metrics': {}
        }
        
        # Analyze loss columns
        loss_cols = [col for col in self.training_history.columns if 'loss' in col.lower()]
        
        for col in loss_cols:
            data = self.training_history[col].dropna()
            if len(data) > 0:
                analysis['metrics'][col] = {
                    'initial': float(data.iloc[0]),
                    'final': float(data.iloc[-1]),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'std': float(data.std())
                }
                
                # Check convergence
                last_10_pct = data.iloc[int(-len(data)*0.1):]
                convergence = abs(last_10_pct.mean() - last_10_pct.iloc[-1]) / (abs(last_10_pct.mean()) + 1e-8)
                analysis['metrics'][col]['convergence_score'] = float(convergence)
                
                # Trend
                early = data.iloc[:len(data)//2].mean()
                late = data.iloc[len(data)//2:].mean()
                trend = "declining ↓" if late < early else "rising ↑" if late > early else "stable →"
                
                logger.info(f"\n  {col}:")
                logger.info(f"    Initial: {analysis['metrics'][col]['initial']:.4f}")
                logger.info(f"    Final:   {analysis['metrics'][col]['final']:.4f}")
                logger.info(f"    Range:   [{analysis['metrics'][col]['min']:.4f}, {analysis['metrics'][col]['max']:.4f}]")
                logger.info(f"    Mean:    {analysis['metrics'][col]['mean']:.4f} (±{analysis['metrics'][col]['std']:.4f})")
                logger.info(f"    Trend:   {trend}")
                logger.info(f"    Convergence score: {convergence:.3f} {'✅ Good' if convergence < 0.1 else '⚠️  Fair' if convergence < 0.3 else '❌ Poor'}")
        
        # Check for divergence
        for col in loss_cols:
            data = self.training_history[col].dropna()
            if data.max() > data.mean() * 10:
                logger.warning(f"⚠️  Potential divergence in {col} (max {data.max():.4f} >> mean {data.mean():.4f})")
        
        return analysis
    
    def detect_overfitting(self):
        """Detect overfitting patterns"""
        if self.training_history is None:
            logger.warning("⚠️  Cannot detect overfitting - history not found")
            return {}
        
        logger.info("\n" + "="*90)
        logger.info("🔍 OVERFITTING DETECTION")
        logger.info("="*90)
        
        overfitting_analysis = {
            'detected_overfitting': False,
            'indicators': []
        }
        
        # Look for train/test loss columns
        train_cols = [col for col in self.training_history.columns if 'train' in col.lower()]
        test_cols = [col for col in self.training_history.columns if 'test' in col.lower() or 'val' in col.lower()]
        
        if train_cols and test_cols:
            logger.info("\n  Comparing train vs validation...")
            for train_col, test_col in zip(train_cols, test_cols):
                if train_col in self.training_history.columns and test_col in self.training_history.columns:
                    train_loss = self.training_history[train_col].dropna()
                    test_loss = self.training_history[test_col].dropna()
                    
                    if len(train_loss) > 0 and len(test_loss) > 0:
                        train_final = train_loss.iloc[-1]
                        test_final = test_loss.iloc[-1]
                        gap = abs(test_final - train_final) / (abs(train_final) + 1e-8)
                        
                        logger.info(f"\n    {train_col} vs {test_col}:")
                        logger.info(f"      Train (final): {train_final:.4f}")
                        logger.info(f"      Test  (final): {test_final:.4f}")
                        logger.info(f"      Gap:   {gap:.3f} {'⚠️  High' if gap > 0.2 else '✅ Low'}")
                        
                        if gap > 0.2:
                            overfitting_analysis['detected_overfitting'] = True
                            overfitting_analysis['indicators'].append(f"Large train-test gap in {train_col}: {gap:.3f}")
        
        if not overfitting_analysis['detected_overfitting']:
            logger.info("\n  ✅ No significant overfitting detected")
        
        return overfitting_analysis
    
    def compare_distributions(self):
        """Compare real vs synthetic distributions"""
        if self.real_data is None or self.synthetic_data is None:
            logger.warning("⚠️  Cannot compare distributions - data missing")
            return {}
        
        logger.info("\n" + "="*90)
        logger.info("📊 DISTRIBUTION COMPARISON: Real vs Synthetic")
        logger.info("="*90)
        
        comparison = {
            'features': {}
        }
        
        for feat in self.features:
            real_feat = self.real_data[feat].dropna()
            syn_feat = self.synthetic_data[feat].dropna()
            
            # Statistical tests
            ks_stat, ks_pval = scipy_stats.ks_2samp(real_feat, syn_feat)
            wasserstein = scipy_stats.wasserstein_distance(real_feat, syn_feat)
            
            # Moments
            real_mean, real_std = real_feat.mean(), real_feat.std()
            syn_mean, syn_std = syn_feat.mean(), syn_feat.std()
            
            comparison['features'][feat] = {
                'real_mean': float(real_mean),
                'real_std': float(real_std),
                'syn_mean': float(syn_mean),
                'syn_std': float(syn_std),
                'mean_error_pct': float(abs(syn_mean - real_mean) / (abs(real_mean) + 1e-8) * 100),
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pval),
                'wasserstein_distance': float(wasserstein),
                'coverage_pct': float(np.sum((syn_feat >= real_feat.min()) & (syn_feat <= real_feat.max())) / len(syn_feat) * 100)
            }
            
            logger.info(f"\n  {feat}:")
            logger.info(f"    Real:  mean={real_mean:.2f}, std={real_std:.2f}")
            logger.info(f"    Synth: mean={syn_mean:.2f}, std={syn_std:.2f}")
            logger.info(f"    Mean error: {comparison['features'][feat]['mean_error_pct']:.1f}%")
            logger.info(f"    Coverage: {comparison['features'][feat]['coverage_pct']:.1f}% (within real range)")
            logger.info(f"    KS test: stat={ks_stat:.3f}, p={ks_pval:.2e} {'✅' if ks_pval > 0.05 else '❌'}")
            logger.info(f"    Wasserstein: {wasserstein:.2f}")
        
        return comparison
    
    def quality_score(self):
        """Calculate overall quality score"""
        if self.real_data is None or self.synthetic_data is None:
            return None
        
        logger.info("\n" + "="*90)
        logger.info("⭐ OVERALL QUALITY SCORE")
        logger.info("="*90)
        
        scores = []
        
        for feat in self.features:
            real_feat = self.real_data[feat].dropna()
            syn_feat = self.synthetic_data[feat].dropna()
            
            # Coverage score (0-100)
            coverage = np.sum((syn_feat >= real_feat.min()) & (syn_feat <= real_feat.max())) / len(syn_feat) * 100
            coverage_score = coverage
            
            # Distribution match score (0-100, based on KS test)
            ks_stat, _ = scipy_stats.ks_2samp(real_feat, syn_feat)
            distribution_score = max(0, 100 - ks_stat * 100)
            
            # Mean/Std match score
            real_mean, real_std = real_feat.mean(), real_feat.std()
            syn_mean, syn_std = syn_feat.mean(), syn_feat.std()
            mean_error = abs(syn_mean - real_mean) / (abs(real_mean) + 1e-8) * 100
            mean_score = max(0, 100 - mean_error)
            
            # Combine
            feature_score = (coverage_score * 0.3 + distribution_score * 0.4 + mean_score * 0.3)
            scores.append(feature_score)
            
            logger.info(f"\n  {feat}:")
            logger.info(f"    Coverage:      {coverage_score:.1f}%")
            logger.info(f"    Distribution:  {distribution_score:.1f}%")
            logger.info(f"    Mean Match:    {mean_score:.1f}%")
            logger.info(f"    Feature Score: {feature_score:.1f}%")
        
        overall_score = np.mean(scores)
        logger.info(f"\n  ━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"  Overall Quality Score: {overall_score:.1f}%")
        logger.info(f"  ━━━━━━━━━━━━━━━━━━━━━━")
        
        # Interpretation
        if overall_score >= 80:
            logger.info(f"  ✅ Excellent - Model ready for production")
        elif overall_score >= 60:
            logger.info(f"  🟡 Good - Model suitable for research/prototyping")
        elif overall_score >= 40:
            logger.info(f"  🟠 Fair - Model needs improvement")
        else:
            logger.info(f"  ❌ Poor - Model requires major refinement")
        
        return overall_score
    
    def generate_visualizations(self):
        """Generate comparison plots"""
        if self.training_history is None or self.real_data is None or self.synthetic_data is None:
            logger.warning("⚠️  Cannot generate visualizations - data missing")
            return
        
        logger.info("\n📈 Generating visualizations...")
        
        # Training history plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('WGAN Training Analysis', fontsize=16, fontweight='bold')
        
        # Loss curves
        loss_cols = [col for col in self.training_history.columns if 'loss' in col.lower()]
        for i, col in enumerate(loss_cols[:2]):
            ax = axes[0, i]
            if col in self.training_history.columns:
                ax.plot(self.training_history[col], linewidth=1.5, label=col)
                ax.set_title(f'{col} over epochs')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        # Distribution comparison
        ax = axes[1, 0]
        for feat in self.features[:2]:
            ax.hist(self.real_data[feat], alpha=0.5, bins=20, label=f'Real {feat}')
            ax.hist(self.synthetic_data[feat], alpha=0.5, bins=20, label=f'Synthetic {feat}')
        ax.set_title('Feature Distributions')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = "SUMMARY STATISTICS\n" + "="*30 + "\n"
        summary_text += f"Real samples: {len(self.real_data)}\n"
        summary_text += f"Synthetic samples: {len(self.synthetic_data)}\n"
        summary_text += f"Training epochs: {len(self.training_history)}\n"
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'training_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Saved training analysis plot: {plot_path}")
        plt.close()
    
    def save_report(self, training_analysis, overfitting_analysis, distribution_comparison, quality_score, coverage_analysis=None):
        """Save comprehensive report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_analysis': training_analysis,
            'overfitting_detection': overfitting_analysis,
            'distribution_comparison': distribution_comparison,
            'coverage_metrics': coverage_analysis if coverage_analysis else {},
            'quality_score': quality_score,
            'recommendations': self.generate_recommendations(quality_score, overfitting_analysis)
        }
        
        report_path = self.output_dir / 'experimental_analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✅ Saved report: {report_path}")
    
    def generate_recommendations(self, quality_score, overfitting_analysis):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if quality_score is not None:
            if quality_score < 60:
                recommendations.append("1. Increase training epochs (currently may be undertrained)")
                recommendations.append("2. Verify data quality and check for outliers")
                recommendations.append("3. Adjust GAN hyperparameters (learning rate, batch size)")
            
            if quality_score >= 80:
                recommendations.append("✅ Model quality is excellent - ready for production")
        
        if overfitting_analysis.get('detected_overfitting'):
            recommendations.append("⚠️  Overfitting detected - consider:")
            recommendations.append("   - Add regularization (batch normalization, dropout)")
            recommendations.append("   - Increase training data diversity")
            recommendations.append("   - Use gradient penalty (WGAN-GP already does this)")
        
        return recommendations
    
    def analyze_coverage_metrics(self):
        """Analyze coverage metrics: methods, branches, statements"""
        if self.real_data is None or self.synthetic_data is None:
            logger.warning("⚠️  Cannot analyze coverage - data missing")
            return {}
        
        logger.info("\n" + "="*90)
        logger.info("📈 COVERAGE METRICS ANALYSIS")
        logger.info("="*90)
        
        coverage_analysis = {
            'real_stats': {},
            'synthetic_stats': {},
            'comparison': {},
            'correlations': {},
            'coverage_tiers': {},
            'imbalance_patterns': {}
        }
        
        feature_names = ['methods', 'branches', 'statements']
        
        # 1. Feature-level statistics
        logger.info("\n  📊 FEATURE STATISTICS:\n")
        
        for feat in feature_names:
            real_feat = self.real_data[feat].dropna()
            syn_feat = self.synthetic_data[feat].dropna()
            
            # Calculate percentages (0-100% normalized by max)
            real_max = real_feat.max()
            real_pct = (real_feat / real_max * 100) if real_max > 0 else real_feat * 100
            syn_pct = (syn_feat / real_max * 100) if real_max > 0 else syn_feat * 100
            
            coverage_analysis['real_stats'][feat] = {
                'count': len(real_feat),
                'min': float(real_feat.min()),
                'max': float(real_feat.max()),
                'mean': float(real_feat.mean()),
                'median': float(real_feat.median()),
                'std': float(real_feat.std()),
                'pct_mean': float(real_pct.mean()),
                'pct_std': float(real_pct.std())
            }
            
            coverage_analysis['synthetic_stats'][feat] = {
                'count': len(syn_feat),
                'min': float(syn_feat.min()),
                'max': float(syn_feat.max()),
                'mean': float(syn_feat.mean()),
                'median': float(syn_feat.median()),
                'std': float(syn_feat.std()),
                'pct_mean': float(syn_pct.mean()),
                'pct_std': float(syn_pct.std())
            }
            
            logger.info(f"  {feat.upper()}:")
            logger.info(f"    Real data:   min={real_feat.min():.0f}, max={real_feat.max():.0f}, mean={real_feat.mean():.1f}±{real_feat.std():.1f}")
            logger.info(f"    Synthetic:   min={syn_feat.min():.0f}, max={syn_feat.max():.0f}, mean={syn_feat.mean():.1f}±{syn_feat.std():.1f}")
            logger.info(f"    Percentage:  Real {real_pct.mean():.1f}±{real_pct.std():.1f}%, Synthetic {syn_pct.mean():.1f}±{syn_pct.std():.1f}%")
            
            # Error percentage
            mean_error = abs(syn_feat.mean() - real_feat.mean()) / (real_feat.mean() + 1e-8) * 100
            logger.info(f"    Mean error:  {mean_error:.1f}%")
        
        # 2. Coverage tier categorization
        logger.info("\n  📊 COVERAGE TIER DISTRIBUTION:\n")
        logger.info("     Tier Definitions: Very Low (0-20%), Low (20-40%), Medium (40-60%), High (60-80%), Very High (80-100%)")
        
        for data_type, data in [('Real', self.real_data), ('Synthetic', self.synthetic_data)]:
            logger.info(f"\n    {data_type} Data:")
            tier_counts = {'Very Low': 0, 'Low': 0, 'Medium': 0, 'High': 0, 'Very High': 0}
            
            for feat in feature_names:
                feat_data = data[feat].dropna()
                feat_max = self.real_data[feat].max()
                feat_pct = (feat_data / feat_max * 100) if feat_max > 0 else feat_data * 100
                
                # Categorize
                low = np.sum(feat_pct < 20)
                low_med = np.sum((feat_pct >= 20) & (feat_pct < 40))
                med = np.sum((feat_pct >= 40) & (feat_pct < 60))
                high = np.sum((feat_pct >= 60) & (feat_pct < 80))
                very_high = np.sum(feat_pct >= 80)
                
                logger.info(f"      {feat:12s}: Very Low={low:3d} ({low/len(feat_pct)*100:5.1f}%) | " +
                           f"Low={low_med:3d} ({low_med/len(feat_pct)*100:5.1f}%) | " +
                           f"Medium={med:3d} ({med/len(feat_pct)*100:5.1f}%) | " +
                           f"High={high:3d} ({high/len(feat_pct)*100:5.1f}%) | " +
                           f"Very High={very_high:3d} ({very_high/len(feat_pct)*100:5.1f}%)")
        
        # 3. Correlation analysis
        logger.info("\n  🔗 CORRELATION BETWEEN COVERAGE TYPES:\n")
        
        # Real data correlations
        real_corr = self.real_data[feature_names].corr()
        coverage_analysis['correlations']['real'] = real_corr.to_dict()
        
        logger.info("    Real Data Correlations:")
        for i, feat1 in enumerate(feature_names):
            for feat2 in feature_names[i+1:]:
                corr_val = real_corr.loc[feat1, feat2]
                logger.info(f"      {feat1} ↔ {feat2}: {corr_val:+.3f} {'(Strong +)' if corr_val > 0.7 else '(Strong -)' if corr_val < -0.7 else '(Moderate)' if abs(corr_val) > 0.3 else '(Weak)'}")
        
        # Synthetic data correlations
        syn_corr = self.synthetic_data[feature_names].corr()
        coverage_analysis['correlations']['synthetic'] = syn_corr.to_dict()
        
        logger.info("\n    Synthetic Data Correlations:")
        for i, feat1 in enumerate(feature_names):
            for feat2 in feature_names[i+1:]:
                corr_val = syn_corr.loc[feat1, feat2]
                logger.info(f"      {feat1} ↔ {feat2}: {corr_val:+.3f} {'(Strong +)' if corr_val > 0.7 else '(Strong -)' if corr_val < -0.7 else '(Moderate)' if abs(corr_val) > 0.3 else '(Weak)'}")
        
        # 4. Imbalance detection
        logger.info("\n  ⚖️  COVERAGE IMBALANCE DETECTION:\n")
        logger.info("     Imbalance = High coverage in one metric but low in another\n")
        
        for data_type, data in [('Real', self.real_data), ('Synthetic', self.synthetic_data)]:
            imbalance_count = 0
            imbalance_patterns = []
            
            feat_max = self.real_data[feature_names].max()
            feat_pct = data[feature_names].div(feat_max) * 100
            
            # Find cases where one is high (>70%) and another is low (<30%)
            for idx, row in feat_pct.iterrows():
                highs = [f for f in feature_names if row[f] > 70]
                lows = [f for f in feature_names if row[f] < 30]
                
                if highs and lows:
                    imbalance_count += 1
                    imbalance_patterns.append({
                        'high': highs,
                        'low': lows,
                        'values': row.to_dict()
                    })
            
            imbalance_pct = imbalance_count / len(data) * 100
            logger.info(f"    {data_type}: {imbalance_count} samples ({imbalance_pct:.1f}%) with coverage imbalance")
            
            coverage_analysis['imbalance_patterns'][data_type] = {
                'count': imbalance_count,
                'percentage': imbalance_pct,
                'examples': imbalance_patterns[:5]  # Show first 5 examples
            }
            
            # Show patterns
            if imbalance_patterns:
                logger.info(f"      Top patterns in {data_type} data:")
                pattern_types = {}
                for pattern in imbalance_patterns:
                    key = f"High {','.join(pattern['high'])} + Low {','.join(pattern['low'])}"
                    pattern_types[key] = pattern_types.get(key, 0) + 1
                
                for pattern_type, count in sorted(pattern_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                    logger.info(f"        {pattern_type}: {count} samples ({count/len(imbalance_patterns)*100:.1f}%)")
        
        logger.info("\n" + "="*90)
        return coverage_analysis
    
    def run(self):
        """Run full experimental analysis"""
        logger.info("\n" + "█"*90)
        logger.info("█" + " "*88 + "█")
        logger.info("█" + " "*20 + "EXPERIMENTAL ANALYSIS: Training Validation" + " "*26 + "█")
        logger.info("█" + " "*88 + "█")
        logger.info("█"*90)
        
        try:
            training_analysis = self.analyze_training_history()
            overfitting_analysis = self.detect_overfitting()
            distribution_comparison = self.compare_distributions()
            coverage_analysis = self.analyze_coverage_metrics()
            quality_score = self.quality_score()
            
            self.generate_visualizations()
            self.save_report(training_analysis, overfitting_analysis, distribution_comparison, quality_score, coverage_analysis)
            
            logger.info("\n" + "="*90)
            logger.info("✅ EXPERIMENTAL ANALYSIS COMPLETE")
            logger.info("="*90)
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}", exc_info=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experimental analysis for any dataset pipeline run')
    parser.add_argument('--training-history', type=str, required=True,
                        help='Path to training_history.csv')
    parser.add_argument('--real-data', type=str, required=True,
                        help='Path to real dataset CSV (no header, 3 columns)')
    parser.add_argument('--synthetic-data', type=str, required=True,
                        help='Path to generated synthetic CSV')
    parser.add_argument('--output-dir', type=str, default='outputs/experimental',
                        help='Directory to write analysis artifacts')
    args = parser.parse_args()

    analyzer = ExperimentalAnalysis(
        training_history_path=args.training_history,
        real_data_path=args.real_data,
        synthetic_data_path=args.synthetic_data,
        output_dir=args.output_dir,
    )
    analyzer.run()
