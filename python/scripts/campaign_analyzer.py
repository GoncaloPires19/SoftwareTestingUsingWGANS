"""
Campaign Results Analyzer: Compare training runs and generate insights
Tracks improvements and documents best configurations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CampaignAnalyzer:
    """Analyze and compare training campaign results"""
    
    def __init__(self):
        self.campaign_dir = Path('outputs/campaign')
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_df = None
        self.analysis = {}
        self.annotations = []
    
    def load_campaign_results(self) -> bool:
        """Load campaign results from CSV"""
        results_path = self.campaign_dir / 'campaign_results.csv'
        
        if not results_path.exists():
            logger.warning(f"⚠️  Campaign results not found: {results_path}")
            return False
        
        try:
            self.results_df = pd.read_csv(results_path)
            logger.info(f"✅ Loaded {len(self.results_df)} training runs")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading results: {e}")
            return False
    
    def analyze_convergence(self):
        """Analyze training convergence patterns"""
        logger.info("\n" + "="*90)
        logger.info("🔍 CONVERGENCE ANALYSIS")
        logger.info("="*90 + "\n")
        
        completed = self.results_df[self.results_df['status'] == 'completed']
        
        if len(completed) == 0:
            logger.warning("⚠️  No completed runs to analyze")
            return
        
        # Check for unstable training
        unstable_runs = completed[completed['max_d_loss'] - completed['min_d_loss'] > 2.0]
        
        for idx, run in completed.iterrows():
            config_name = run['config_name']
            d_volatility = run['max_d_loss'] - run['min_d_loss']
            convergence_score = 100 - (abs(run['final_d_loss'] - run['avg_d_loss']) / (abs(run['avg_d_loss']) + 1e-8) * 100)
            convergence_score = max(0, min(100, convergence_score))
            
            stability = "✅ Stable" if d_volatility < 1.0 else "⚠️  Volatile" if d_volatility < 2.0 else "❌ Unstable"
            
            logger.info(f"  {config_name:20s} | Convergence: {convergence_score:5.1f}% | {stability} (volatility: {d_volatility:.2f})")
            
            self.annotations.append({
                'config': config_name,
                'metric': 'convergence_score',
                'value': convergence_score,
                'note': f"Volatility: {d_volatility:.2f}"
            })
    
    def analyze_loss_dynamics(self):
        """Analyze critic vs generator loss dynamics"""
        logger.info("\n" + "─"*90)
        logger.info("📊 LOSS DYNAMICS ANALYSIS")
        logger.info("─"*90 + "\n")
        
        completed = self.results_df[self.results_df['status'] == 'completed']
        
        for idx, run in completed.iterrows():
            config_name = run['config_name']
            d_loss = run['final_d_loss']
            g_loss = run['final_g_loss']
            loss_ratio = abs(g_loss) / (abs(d_loss) + 1e-8)
            
            # Analyze loss relationship
            if loss_ratio > 5:
                dynamics = "❌ G dominates (may indicate vanishing G loss)"
            elif loss_ratio > 2:
                dynamics = "⚠️  Imbalanced (G stronger than D)"
            elif loss_ratio < 0.2:
                dynamics = "⚠️  D dominates (may overpower G)"
            else:
                dynamics = "✅ Balanced"
            
            logger.info(f"  {config_name:20s} | D: {d_loss:8.4f} | G: {g_loss:8.4f} | Ratio: {loss_ratio:.2f}x | {dynamics}")
            
            self.annotations.append({
                'config': config_name,
                'metric': 'loss_dynamics',
                'value': loss_ratio,
                'note': dynamics
            })
    
    def compare_configurations(self):
        """Compare different configuration choices"""
        logger.info("\n" + "─"*90)
        logger.info("⚙️  CONFIGURATION IMPACT ANALYSIS")
        logger.info("─"*90 + "\n")
        
        completed = self.results_df[self.results_df['status'] == 'completed']
        
        # Learning rate impact
        logger.info("  Learning Rate Impact:")
        lr_groups = completed.groupby('learning_rate_d').agg({
            'final_d_loss': ['mean', 'std'],
            'final_g_loss': ['mean', 'std'],
            'config_name': 'count'
        }).round(4)
        
        for lr, row in lr_groups.iterrows():
            count = row[('config_name', 'count')]
            d_loss = row[('final_d_loss', 'mean')]
            logger.info(f"    LR {lr:.0e}: {d_loss:.4f} (n={int(count)})")
        
        # Batch size impact
        logger.info("\n  Batch Size Impact:")
        bs_groups = completed.groupby('batch_size').agg({
            'final_d_loss': ['mean', 'std'],
            'training_time_seconds': ['mean']
        }).round(2)
        
        for bs, row in bs_groups.iterrows():
            d_loss = row[('final_d_loss', 'mean')]
            time_taken = row[('training_time_seconds', 'mean')]
            logger.info(f"    BS {int(bs):2d}: D Loss {d_loss:.4f}, Time {time_taken:.1f}s")
        
        # Gradient penalty impact
        logger.info("\n  Gradient Penalty Impact:")
        gp_groups = completed.groupby('lambda_gp').agg({
            'final_d_loss': ['mean'],
            'final_g_loss': ['mean']
        }).round(4)
        
        for gp, row in gp_groups.iterrows():
            d_loss = row[('final_d_loss', 'mean')]
            logger.info(f"    λ_gp {gp:5.1f}: D Loss {d_loss:.4f}")
    
    def rank_configurations(self):
        """Rank configurations by multiple criteria"""
        logger.info("\n" + "="*90)
        logger.info("🏆 CONFIGURATION RANKINGS")
        logger.info("="*90 + "\n")
        
        completed = self.results_df[self.results_df['status'] == 'completed']
        
        # Calculate comprehensive score
        scores = []
        
        for idx, run in completed.iterrows():
            # Normalize metrics (lower is better)
            d_loss_score = (run['final_d_loss'] - completed['final_d_loss'].min()) / (completed['final_d_loss'].max() - completed['final_d_loss'].min() + 1e-8)
            time_score = (run['training_time_seconds'] - completed['training_time_seconds'].min()) / (completed['training_time_seconds'].max() - completed['training_time_seconds'].min() + 1e-8)
            stability_score = (run['max_d_loss'] - run['min_d_loss']) / 2.0
            
            # Combined score (lower is better)
            combined = d_loss_score * 0.5 + time_score * 0.3 + stability_score * 0.2
            
            scores.append({
                'config': run['config_name'],
                'd_loss_score': d_loss_score,
                'time_score': time_score,
                'stability_score': stability_score,
                'combined': combined,
                'final_d_loss': run['final_d_loss']
            })
        
        scores_df = pd.DataFrame(scores).sort_values('combined')
        
        logger.info("  By Combined Score (lower is better):\n")
        for i, row in scores_df.head(5).iterrows():
            logger.info(f"  {i+1}. {row['config']:20s} | Score: {row['combined']:.3f} | D Loss: {row['final_d_loss']:.4f}")
    
    def generate_insights(self):
        """Generate actionable insights """
        logger.info("\n" + "="*90)
        logger.info("💡 KEY INSIGHTS & RECOMMENDATIONS")
        logger.info("="*90 + "\n")
        
        completed = self.results_df[self.results_df['status'] == 'completed']
        failed = self.results_df[self.results_df['status'] == 'failed']
        
        insights = []
        
        # Success rate
        success_rate = len(completed) / len(self.results_df) * 100
        logger.info(f"  ✓ Training Success Rate: {success_rate:.0f}% ({len(completed)}/{len(self.results_df)})")
        if failed.index is not None and len(failed) > 0:
            logger.info(f"    Failed runs: {list(failed['config_name'].values)}")
            insights.append(f"⚠️  {len(failed)} training runs failed - need debugging")
        
        # Best dataset
        dataset_performance = completed.groupby('dataset')['final_d_loss'].mean().sort_values()
        best_dataset = dataset_performance.index[0]
        logger.info(f"\n  🎯 Best Dataset: {best_dataset}")
        logger.info(f"     Avg D Loss: {dataset_performance.iloc[0]:.4f}")
        insights.append(f"✅ Dataset '{best_dataset}' shows best convergence")
        
        # Fastest setup
        fastest = completed.loc[completed['training_time_seconds'].idxmin()]
        logger.info(f"\n  ⚡ Fastest Training: {fastest['config_name']}")
        logger.info(f"     Time: {fastest['training_time_seconds']:.1f}s")
        logger.info(f"     Config: BS={int(fastest['batch_size'])}, Epochs={int(fastest['num_epochs'])}")
        
        # Most stable
        completed['stability'] = completed['max_d_loss'] - completed['min_d_loss']
        most_stable = completed.loc[completed['stability'].idxmin()]
        logger.info(f"\n  🛡️  Most Stable: {most_stable['config_name']}")
        logger.info(f"     Volatility: {most_stable['stability']:.4f}")
        insights.append(f"✅ Most stable config achieves low volatility with GP={most_stable['lambda_gp']}")
        
        # Hyperparameter trends
        logger.info(f"\n  ⚙️  Hyperparameter Trends:")
        
        # Learning rate correlation
        lr_corr = completed['learning_rate_d'].corr(completed['final_d_loss'])
        lr_direction = "higher is better ↑" if lr_corr < -0.3 else "lower is better ↓" if lr_corr > 0.3 else "neutral"
        logger.info(f"     Learning Rate: {lr_direction} (corr: {lr_corr:.2f})")
        
        if abs(lr_corr) > 0.3:
            insights.append(f"💡 Learning rate correlation: {lr_corr:.2f} suggests {lr_direction}")
        
        # Batch size analysis
        if len(completed['batch_size'].unique()) > 1:
            small_bs = completed[completed['batch_size'] < completed['batch_size'].median()]['final_d_loss'].mean()
            large_bs = completed[completed['batch_size'] >= completed['batch_size'].median()]['final_d_loss'].mean()
            if small_bs < large_bs:
                logger.info(f"     Batch Size: Smaller batches perform better")
                insights.append("💡 Smaller batch sizes lead to better convergence")
            else:
                logger.info(f"     Batch Size: Larger batches perform better")
                insights.append("💡 Larger batch sizes stabilize training")
        
        logger.info("\n" + "─"*90)
        logger.info("🎯 RECOMMENDATIONS:\n")
        for i, insight in enumerate(insights, 1):
            logger.info(f"  {i}. {insight}")
    
    def save_analysis_report(self):
        """Save comprehensive analysis report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_runs': len(self.results_df),
            'completed_runs': len(self.results_df[self.results_df['status'] == 'completed']),
            'failed_runs': len(self.results_df[self.results_df['status'] == 'failed']),
            'annotations': self.annotations
        }
        
        report_path = self.campaign_dir / 'analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✅ Analysis report saved: {report_path}")
    
    def generate_comparison_plots(self):
        """Generate comparison visualizations"""
        completed = self.results_df[self.results_df['status'] == 'completed']
        
        if len(completed) == 0:
            logger.warning("⚠️  No completed runs to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Campaign Comparison', fontsize=16, fontweight='bold')
        
        # D Loss comparison
        ax = axes[0, 0]
        completed.sort_values('final_d_loss')[['config_name', 'final_d_loss']].set_index('config_name').plot(
            kind='barh', ax=ax, legend=False, color='steelblue'
        )
        ax.set_title('Final Discriminator Loss')
        ax.set_xlabel('Loss')
        
        # Training time
        ax = axes[0, 1]
        completed.sort_values('training_time_seconds')[['config_name', 'training_time_seconds']].set_index('config_name').plot(
            kind='barh', ax=ax, legend=False, color='coral'
        )
        ax.set_title('Training Time')
        ax.set_xlabel('Seconds')
        
        # Dataset comparison
        ax = axes[1, 0]
        dataset_stats = completed.groupby('dataset')['final_d_loss'].agg(['mean', 'std'])
        dataset_stats['mean'].plot(kind='bar', ax=ax, color='mediumseagreen', yerr=dataset_stats['std'])
        ax.set_title('Loss by Dataset')
        ax.set_ylabel('Final D Loss')
        ax.set_xlabel('Dataset')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Convergence stability
        ax = axes[1, 1]
        completed['stability'] = completed['max_d_loss'] - completed['min_d_loss']
        completed.sort_values('stability')[['config_name', 'stability']].set_index('config_name').plot(
            kind='barh', ax=ax, legend=False, color='mediumpurple'
        )
        ax.set_title('Training Stability (Lower is Better)')
        ax.set_xlabel('Volatility')
        
        plt.tight_layout()
        plot_path = self.campaign_dir / 'campaign_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Comparison plot saved: {plot_path}")
        plt.close()
    
    def run_analysis(self):
        """Run complete analysis"""
        logger.info("\n" + "█"*90)
        logger.info("█" + " "*88 + "█")
        logger.info("█" + " "*20 + "CAMPAIGN RESULTS ANALYSIS & REPORTING" + " "*32 + "█")
        logger.info("█" + " "*88 + "█")
        logger.info("█"*90)
        
        if not self.load_campaign_results():
            logger.error("❌ Cannot proceed without campaign results")
            return
        
        self.analyze_convergence()
        self.analyze_loss_dynamics()
        self.compare_configurations()
        self.rank_configurations()
        self.generate_insights()
        
        self.save_analysis_report()
        self.generate_comparison_plots()
        
        logger.info("\n" + "="*90)
        logger.info("✅ ANALYSIS COMPLETE")
        logger.info("="*90 + "\n")

if __name__ == '__main__':
    analyzer = CampaignAnalyzer()
    analyzer.run_analysis()
