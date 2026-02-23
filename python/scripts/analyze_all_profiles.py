"""
Analyze and compare all training profiles.
"""
import pandas as pd
import json
from pathlib import Path

def analyze_profile(output_dir):
    """Analyze a single training profile."""
    output_path = Path(output_dir)
    
    # Load config
    with open(output_path / 'config.json', 'r') as f:
        config = json.load(f)
    
    # Load history
    history = pd.read_csv(output_path / 'training_history.csv')
    
    # Find best epochs
    best_coverage_idx = history['coverage'].idxmax()
    best_coverage = history.iloc[best_coverage_idx]
    
    best_methods_idx = history['max_methods'].idxmax()
    best_methods = history.iloc[best_methods_idx]
    
    best_branches_idx = history['max_branches'].idxmax()
    best_branches = history.iloc[best_branches_idx]
    
    best_statements_idx = history['max_statements'].idxmax()
    best_statements = history.iloc[best_statements_idx]
    
    final = history.iloc[-1]
    
    return {
        'profile': config['profile'],
        'epochs': config['epochs'],
        'batch_size': config['profile_config']['batch_size'],
        'learning_rate': config['profile_config']['learning_rate'],
        'lambda_gp': config['profile_config']['lambda_gp'],
        'n_critic': config['profile_config']['n_critic'],
        'best_coverage_epoch': int(best_coverage['epoch']),
        'best_coverage': best_coverage['coverage'],
        'best_coverage_max': [best_coverage['max_methods'], best_coverage['max_branches'], best_coverage['max_statements']],
        'best_methods_epoch': int(best_methods['epoch']),
        'best_methods': best_methods['max_methods'],
        'best_branches_epoch': int(best_branches['epoch']),
        'best_branches': best_branches['max_branches'],
        'best_statements_epoch': int(best_statements['epoch']),
        'best_statements': best_statements['max_statements'],
        'final_epoch': int(final['epoch']),
        'final_coverage': final['coverage'],
        'final_max': [final['max_methods'], final['max_branches'], final['max_statements']],
        'final_diversity': final['diversity']
    }

def main():
    """Analyze all available profiles."""
    
    outputs_dir = Path('outputs')
    
    # Find all profile directories
    profile_dirs = []
    for pattern in ['aggressive_*', 'experimental_*', 'conservative_*', 'coverage-seeking_*', 'hybrid-realistic_*']:
        profile_dirs.extend(outputs_dir.glob(pattern))
    
    if not profile_dirs:
        print("No profile outputs found.")
        return
    
    print(f"\n{'='*100}")
    print(f"TRAINING PROFILES COMPARISON")
    print(f"{'='*100}\n")
    
    results = []
    for profile_dir in sorted(profile_dirs):
        try:
            result = analyze_profile(profile_dir)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {profile_dir.name}: {e}")
    
    # Print summary table
    print(f"{'Profile':<20} {'Best Epoch':<12} {'Best Cov%':<12} {'Peak Methods':<14} {'Peak Branches':<15} {'Peak Statements':<16}")
    print(f"{'-'*100}")
    
    for r in results:
        print(f"{r['profile']:<20} "
              f"{r['best_coverage_epoch']:<12} "
              f"{r['best_coverage']:>10.2f}% "
              f"{r['best_methods']:>13.0f} "
              f"{r['best_branches']:>14.0f} "
              f"{r['best_statements']:>15.0f}")
    
    print(f"\n{'='*100}")
    print(f"DETAILED ANALYSIS")
    print(f"{'='*100}\n")
    
    real_max = [777, 1000, 1000]  # Real data max values
    
    for r in results:
        print(f"\n{r['profile'].upper()}")
        print(f"  Configuration:")
        print(f"    - Epochs: {r['epochs']}, Batch: {r['batch_size']}, LR: {r['learning_rate']}, λ_GP: {r['lambda_gp']}, n_critic: {r['n_critic']}")
        print(f"  Best Coverage: {r['best_coverage']:.2f}% at epoch {r['best_coverage_epoch']}")
        print(f"    - Max values: [{r['best_coverage_max'][0]:.0f}, {r['best_coverage_max'][1]:.0f}, {r['best_coverage_max'][2]:.0f}]")
        print(f"  Peak Values:")
        print(f"    - Methods:    {r['best_methods']:>6.0f} at epoch {r['best_methods_epoch']:<3} ({r['best_methods']/real_max[0]*100:>5.1f}% of max)")
        print(f"    - Branches:   {r['best_branches']:>6.0f} at epoch {r['best_branches_epoch']:<3} ({r['best_branches']/real_max[1]*100:>5.1f}% of max)")
        print(f"    - Statements: {r['best_statements']:>6.0f} at epoch {r['best_statements_epoch']:<3} ({r['best_statements']/real_max[2]*100:>5.1f}% of max)")
        print(f"  Final State (epoch {r['final_epoch']}):")
        print(f"    - Coverage: {r['final_coverage']:.2f}%, Diversity: {r['final_diversity']:.3f}")
        print(f"    - Max values: [{r['final_max'][0]:.0f}, {r['final_max'][1]:.0f}, {r['final_max'][2]:.0f}]")
    
    print(f"\n{'='*100}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*100}\n")
    
    # Find best by coverage
    best_by_coverage = max(results, key=lambda x: x['best_coverage'])
    print(f"✅ Best Coverage: {best_by_coverage['profile']} with {best_by_coverage['best_coverage']:.2f}% at epoch {best_by_coverage['best_coverage_epoch']}")
    
    # Find best by peak branches (most important metric)
    best_by_branches = max(results, key=lambda x: x['best_branches'])
    print(f"✅ Best Branches: {best_by_branches['profile']} with {best_by_branches['best_branches']:.0f} ({best_by_branches['best_branches']/1000*100:.1f}%) at epoch {best_by_branches['best_branches_epoch']}")
    
    # Find best by diversity
    best_by_diversity = max(results, key=lambda x: x['final_diversity'])
    print(f"✅ Best Diversity: {best_by_diversity['profile']} with {best_by_diversity['final_diversity']:.3f}")
    
    print(f"\n⚠️  Note: Best checkpoint may not be the final epoch - use checkpoints from peak performance epochs")

if __name__ == '__main__':
    main()
