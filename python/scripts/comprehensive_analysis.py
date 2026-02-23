"""
Comprehensive comparative analysis of all training runs.
Compares hyperparameters, training dynamics, and final results.
"""
import pandas as pd
import json
from pathlib import Path
import numpy as np
from datetime import datetime

def load_training_data(output_dir):
    """Load training data from output directory."""
    output_path = Path(output_dir)
    
    try:
        # Load config
        with open(output_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Try different file formats
        if (output_path / 'training_history.csv').exists():
            history = pd.read_csv(output_path / 'training_history.csv')
            format_type = 'basic'
        elif (output_path / 'training_metrics.csv').exists():
            history = pd.read_csv(output_path / 'training_metrics.csv')
            format_type = 'advanced'
        else:
            return None
        
        return {
            'config': config,
            'history': history,
            'format': format_type,
            'path': str(output_path)
        }
    except Exception as e:
        return None

def extract_hyperparameters(data):
    """Extract hyperparameters from config."""
    config = data['config']
    
    if 'profile_config' in config:
        # Basic profiles format
        pc = config['profile_config']
        return {
            'profile': config['profile'],
            'epochs': pc['epochs'],
            'batch_size': pc['batch_size'],
            'learning_rate': pc['learning_rate'],
            'lambda_gp': pc['lambda_gp'],
            'n_critic': pc['n_critic'],
            'seed': config.get('seed', 'N/A')
        }
    else:
        # Advanced profiles format
        return {
            'profile': config['profile'],
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'lambda_gp': config.get('lambda_gp', 'N/A'),
            'n_critic': config.get('n_critic', 'N/A'),
            'seed': config.get('seed', 'N/A')
        }

def extract_metrics_basic(data):
    """Extract metrics from basic profile format."""
    history = data['history']
    
    # Find best epochs
    best_coverage_idx = history['coverage'].idxmax()
    best_coverage_row = history.iloc[best_coverage_idx]
    
    best_methods_idx = history['max_methods'].idxmax()
    best_branches_idx = history['max_branches'].idxmax()
    best_statements_idx = history['max_statements'].idxmax()
    
    final_row = history.iloc[-1]
    
    # Calculate stability (std dev of last 20% of epochs)
    last_20pct = int(len(history) * 0.2)
    if last_20pct > 0:
        recent_coverage = history['coverage'].iloc[-last_20pct:]
        coverage_stability = recent_coverage.std()
    else:
        coverage_stability = 0
    
    return {
        'format': 'basic',
        'total_epochs': len(history),
        'best_coverage_epoch': int(best_coverage_row['epoch']),
        'best_coverage': best_coverage_row['coverage'],
        'best_coverage_max': [
            best_coverage_row['max_methods'],
            best_coverage_row['max_branches'],
            best_coverage_row['max_statements']
        ],
        'peak_methods': history['max_methods'].max(),
        'peak_methods_epoch': int(history.iloc[best_methods_idx]['epoch']),
        'peak_branches': history['max_branches'].max(),
        'peak_branches_epoch': int(history.iloc[best_branches_idx]['epoch']),
        'peak_statements': history['max_statements'].max(),
        'peak_statements_epoch': int(history.iloc[best_statements_idx]['epoch']),
        'final_coverage': final_row['coverage'],
        'final_diversity': final_row['diversity'],
        'final_max': [
            final_row['max_methods'],
            final_row['max_branches'],
            final_row['max_statements']
        ],
        'avg_gen_loss': history['gen_loss'].mean(),
        'avg_critic_loss': history['critic_loss'].mean(),
        'coverage_stability': coverage_stability,
        'coverage_growth': final_row['coverage'] - history.iloc[0]['coverage']
    }

def extract_metrics_advanced(data):
    """Extract metrics from advanced profile format."""
    history = data['history']
    
    # Advanced profiles track different metrics
    final_row = history.iloc[-1]
    initial_row = history.iloc[0]
    
    # Calculate stability
    last_20pct = int(len(history) * 0.2)
    if last_20pct > 0:
        recent_diversity = history['diversity'].iloc[-last_20pct:]
        diversity_stability = recent_diversity.std()
    else:
        diversity_stability = 0
    
    return {
        'format': 'advanced',
        'total_epochs': len(history),
        'initial_line_cov': initial_row.get('line_cov', 'N/A'),
        'initial_branch_cov': initial_row.get('branch_cov', 'N/A'),
        'final_line_cov': final_row.get('line_cov', 'N/A'),
        'final_branch_cov': final_row.get('branch_cov', 'N/A'),
        'final_diversity': final_row['diversity'],
        'avg_gen_loss': history['gen_loss'].mean(),
        'avg_critic_loss': history['critic_loss'].mean(),
        'diversity_stability': diversity_stability,
        'coverage_growth': final_row.get('line_cov', 0) - initial_row.get('line_cov', 0)
    }

def calculate_quality_score(metrics, real_max=[777, 1000, 1000]):
    """Calculate overall quality score (0-100)."""
    if metrics['format'] == 'basic':
        # Score based on:
        # 1. Peak coverage relative to max
        coverage_score = min(metrics['best_coverage'], 100)
        
        # 2. Peak values relative to real max
        methods_pct = min((metrics['peak_methods'] / real_max[0]) * 100, 100)
        branches_pct = min((metrics['peak_branches'] / real_max[1]) * 100, 100)
        statements_pct = min((metrics['peak_statements'] / real_max[2]) * 100, 100)
        peak_score = (methods_pct + branches_pct + statements_pct) / 3
        
        # 3. Stability (inverse of std dev)
        stability_score = max(0, 100 - metrics['coverage_stability'] * 2)
        
        # 4. Diversity
        diversity_score = metrics['final_diversity'] * 100
        
        # Weighted average
        quality = (
            coverage_score * 0.3 +
            peak_score * 0.4 +
            stability_score * 0.2 +
            diversity_score * 0.1
        )
        return quality
    else:
        # For advanced profiles, use different scoring
        return metrics['final_diversity'] * 100

def main():
    """Main analysis function."""
    
    outputs_dir = Path('outputs')
    
    # Find all training directories
    patterns = [
        'aggressive_*',
        'experimental_*',
        'conservative_*',
        'coverage-seeking_*',
        'hybrid-realistic_*',
        'continuation_*'
    ]
    
    all_dirs = []
    for pattern in patterns:
        all_dirs.extend(outputs_dir.glob(pattern))
    
    if not all_dirs:
        print("No training outputs found.")
        return
    
    print(f"\n{'='*120}")
    print(f"COMPREHENSIVE TRAINING ANALYSIS")
    print(f"{'='*120}\n")
    print(f"Found {len(all_dirs)} training directories\n")
    
    # Load all data
    all_data = []
    for dir_path in sorted(all_dirs):
        data = load_training_data(dir_path)
        if data:
            all_data.append(data)
    
    print(f"Successfully loaded {len(all_data)} training runs\n")
    
    # Separate by format
    basic_runs = [d for d in all_data if d['format'] == 'basic']
    advanced_runs = [d for d in all_data if d['format'] == 'advanced']
    
    print(f"{'='*120}")
    print(f"BASIC PROFILES ANALYSIS ({len(basic_runs)} runs)")
    print(f"{'='*120}\n")
    
    if basic_runs:
        # Extract all info
        results = []
        for data in basic_runs:
            hyperparams = extract_hyperparameters(data)
            metrics = extract_metrics_basic(data)
            quality = calculate_quality_score(metrics)
            
            results.append({
                **hyperparams,
                **metrics,
                'quality_score': quality,
                'path': Path(data['path']).name
            })
        
        # Print hyperparameters table
        print("HYPERPARAMETERS")
        print(f"{'-'*120}")
        print(f"{'Profile':<20} {'Epochs':<8} {'Batch':<8} {'LR':<12} {'λ_GP':<8} {'n_critic':<10} {'Seed':<8}")
        print(f"{'-'*120}")
        
        for r in results:
            print(f"{r['profile']:<20} {r['epochs']:<8} {r['batch_size']:<8} "
                  f"{r['learning_rate']:<12.2e} {r['lambda_gp']:<8} "
                  f"{r['n_critic']:<10} {r['seed']:<8}")
        
        # Print coverage metrics
        print(f"\n{'='*120}")
        print("COVERAGE METRICS")
        print(f"{'-'*120}")
        print(f"{'Profile':<20} {'Best Epoch':<12} {'Best Cov%':<12} {'Final Cov%':<12} {'Cov Growth':<12} {'Quality':<10}")
        print(f"{'-'*120}")
        
        for r in results:
            print(f"{r['profile']:<20} {r['best_coverage_epoch']:<12} "
                  f"{r['best_coverage']:>10.2f}% {r['final_coverage']:>10.2f}% "
                  f"{r['coverage_growth']:>10.2f}% {r['quality_score']:>8.1f}")
        
        # Print peak values
        print(f"\n{'='*120}")
        print("PEAK VALUES (Real max: Methods=777, Branches=1000, Statements=1000)")
        print(f"{'-'*120}")
        print(f"{'Profile':<20} {'Peak Methods':<20} {'Peak Branches':<20} {'Peak Statements':<20}")
        print(f"{'-'*120}")
        
        real_max = [777, 1000, 1000]
        for r in results:
            methods_str = f"{r['peak_methods']:>5.0f} ({r['peak_methods']/real_max[0]*100:>5.1f}%)"
            branches_str = f"{r['peak_branches']:>5.0f} ({r['peak_branches']/real_max[1]*100:>5.1f}%)"
            statements_str = f"{r['peak_statements']:>5.0f} ({r['peak_statements']/real_max[2]*100:>5.1f}%)"
            
            print(f"{r['profile']:<20} {methods_str:<20} {branches_str:<20} {statements_str:<20}")
        
        # Print training dynamics
        print(f"\n{'='*120}")
        print("TRAINING DYNAMICS")
        print(f"{'-'*120}")
        print(f"{'Profile':<20} {'Avg Gen Loss':<15} {'Avg Critic Loss':<18} {'Coverage Stability':<20} {'Final Diversity':<15}")
        print(f"{'-'*120}")
        
        for r in results:
            print(f"{r['profile']:<20} {r['avg_gen_loss']:>13.4f} {r['avg_critic_loss']:>16.4f} "
                  f"{r['coverage_stability']:>18.2f} {r['final_diversity']:>13.3f}")
        
        # Best epochs summary
        print(f"\n{'='*120}")
        print("BEST EPOCHS PER PROFILE")
        print(f"{'-'*120}")
        print(f"{'Profile':<20} {'Best Coverage':<20} {'Peak Methods':<20} {'Peak Branches':<20} {'Peak Statements':<20}")
        print(f"{'-'*120}")
        
        for r in results:
            cov_str = f"Epoch {r['best_coverage_epoch']}"
            methods_str = f"Epoch {r['peak_methods_epoch']}"
            branches_str = f"Epoch {r['peak_branches_epoch']}"
            statements_str = f"Epoch {r['peak_statements_epoch']}"
            
            print(f"{r['profile']:<20} {cov_str:<20} {methods_str:<20} {branches_str:<20} {statements_str:<20}")
        
        # Recommendations
        print(f"\n{'='*120}")
        print("RECOMMENDATIONS")
        print(f"{'='*120}\n")
        
        # Best overall quality
        best_quality = max(results, key=lambda x: x['quality_score'])
        print(f"🏆 BEST OVERALL: {best_quality['profile']}")
        print(f"   Quality Score: {best_quality['quality_score']:.1f}/100")
        print(f"   Best Coverage: {best_quality['best_coverage']:.2f}% at epoch {best_quality['best_coverage_epoch']}")
        print(f"   Peak Values: Methods={best_quality['peak_methods']:.0f}, "
              f"Branches={best_quality['peak_branches']:.0f}, "
              f"Statements={best_quality['peak_statements']:.0f}\n")
        
        # Best branches (most important)
        best_branches = max(results, key=lambda x: x['peak_branches'])
        if best_branches['profile'] != best_quality['profile']:
            print(f"🎯 BEST BRANCHES COVERAGE: {best_branches['profile']}")
            print(f"   Peak Branches: {best_branches['peak_branches']:.0f} ({best_branches['peak_branches']/1000*100:.1f}%) "
                  f"at epoch {best_branches['peak_branches_epoch']}")
            print(f"   ⚠️  Recommended checkpoint: epoch_{best_branches['peak_branches_epoch']}.pth\n")
        
        # Most stable
        most_stable = min(results, key=lambda x: x['coverage_stability'])
        print(f"📊 MOST STABLE: {most_stable['profile']}")
        print(f"   Coverage Stability: {most_stable['coverage_stability']:.2f}")
        print(f"   Final Coverage: {most_stable['final_coverage']:.2f}%\n")
        
        # Save summary
        summary_df = pd.DataFrame(results)
        summary_path = outputs_dir / 'training_comparison_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"💾 Summary saved to: {summary_path}")
    
    # Advanced profiles
    if advanced_runs:
        print(f"\n{'='*120}")
        print(f"ADVANCED PROFILES ANALYSIS ({len(advanced_runs)} runs)")
        print(f"{'='*120}\n")
        
        adv_results = []
        for data in advanced_runs:
            hyperparams = extract_hyperparameters(data)
            metrics = extract_metrics_advanced(data)
            
            adv_results.append({
                **hyperparams,
                **metrics,
                'path': Path(data['path']).name
            })
        
        # Print table
        print(f"{'Profile':<25} {'Epochs':<8} {'Initial Cov':<15} {'Final Cov':<15} {'Growth':<12} {'Diversity':<12}")
        print(f"{'-'*120}")
        
        for r in adv_results:
            init_cov = f"{r['initial_line_cov']:.2f}" if isinstance(r['initial_line_cov'], (int, float)) else 'N/A'
            final_cov = f"{r['final_line_cov']:.2f}" if isinstance(r['final_line_cov'], (int, float)) else 'N/A'
            growth = f"{r['coverage_growth']:.2f}" if isinstance(r['coverage_growth'], (int, float)) else 'N/A'
            
            print(f"{r['profile']:<25} {r['epochs']:<8} {init_cov:<15} {final_cov:<15} {growth:<12} {r['final_diversity']:<12.3f}")
        
        print(f"\n⚠️  Note: Advanced profiles use different metrics (line/branch coverage from instrumented code)")
        print(f"   These are NOT directly comparable to basic profiles (feature space coverage)")
    
    print(f"\n{'='*120}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*120}\n")

if __name__ == '__main__':
    main()
