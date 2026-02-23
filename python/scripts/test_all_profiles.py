"""
Test all available training profiles and compare results.
"""
import subprocess
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Profile configurations to test
PROFILES_TO_TEST = [
    {
        'name': 'aggressive',
        'epochs': 50,
        'description': '⚡ AGGRESSIVE - Fast training, may be unstable'
    },
    {
        'name': 'experimental',
        'epochs': 80,
        'description': '🧪 EXPERIMENTAL - Improved configuration'
    },
    {
        'name': 'conservative',
        'epochs': 150,
        'description': '🛡️ CONSERVATIVE - Maximum stability'
    },
    {
        'name': 'coverage-seeking',
        'epochs': 150,
        'description': '🎯 COVERAGE-SEEKING - Focus on max coverage'
    }
]

def run_profile_training(profile_name, epochs):
    """Run training with specific profile."""
    print(f"\n{'='*70}")
    print(f"TRAINING: {profile_name.upper()}")
    print(f"{'='*70}\n")
    
    cmd = [
        'python',
        'python/scripts/train_profile_aware.py',
        '--profile', profile_name,
        '--epochs', str(epochs),
        '--seed', '42'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2h timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {profile_name.upper()} training completed successfully")
            return True, result.stdout
        else:
            print(f"❌ {profile_name.upper()} training failed")
            print(f"Error: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  {profile_name.upper()} training timed out")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ Error running {profile_name.upper()}: {e}")
        return False, str(e)

def analyze_training_results(profile_name):
    """Analyze results from a completed training run."""
    # Find most recent output directory for this profile
    output_base = Path('outputs')
    
    # Search for directories matching the profile name
    matching_dirs = sorted(
        output_base.glob(f'{profile_name}_*'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if not matching_dirs:
        print(f"⚠️  No output directory found for {profile_name}")
        return None
    
    result_dir = matching_dirs[0]
    
    # Read training history
    history_file = result_dir / 'training_metrics.csv'
    if not history_file.exists():
        history_file = result_dir / 'training_history.csv'
    
    if not history_file.exists():
        print(f"⚠️  No history file found in {result_dir}")
        return None
    
    df = pd.read_csv(history_file)
    
    # Extract key metrics
    final_row = df.iloc[-1]
    best_coverage_row = df.loc[df['coverage'].idxmax()] if 'coverage' in df.columns else final_row
    
    metrics = {
        'profile': profile_name,
        'total_epochs': len(df),
        'final_coverage': final_row.get('coverage', 0),
        'best_coverage': best_coverage_row.get('coverage', 0),
        'best_coverage_epoch': best_coverage_row.name + 1 if 'coverage' in df.columns else 0,
        'final_diversity': final_row.get('diversity', 0),
        'final_gen_loss': final_row.get('gen_loss', 0),
        'output_dir': str(result_dir)
    }
    
    print(f"\n📊 {profile_name.upper()} Results:")
    print(f"   Final Coverage:  {metrics['final_coverage']:.2f}%")
    print(f"   Best Coverage:   {metrics['best_coverage']:.2f}% (epoch {metrics['best_coverage_epoch']})")
    print(f"   Final Diversity: {metrics['final_diversity']:.3f}")
    print(f"   Output:          {metrics['output_dir']}")
    
    return metrics

def main():
    """Run all profile tests and compare results."""
    print("="*70)
    print("TRAINING PROFILE COMPARISON")
    print("="*70)
    print(f"\nTesting {len(PROFILES_TO_TEST)} profiles:\n")
    
    for i, profile_info in enumerate(PROFILES_TO_TEST, 1):
        print(f"{i}. {profile_info['name']:20s} - {profile_info['description']}")
    
    print("\n" + "="*70)
    input("\nPress Enter to start training (this will take several hours)... ")
    
    results = []
    
    for profile_info in PROFILES_TO_TEST:
        profile_name = profile_info['name']
        epochs = profile_info['epochs']
        
        # Run training
        success, output = run_profile_training(profile_name, epochs)
        
        if success:
            # Analyze results
            metrics = analyze_training_results(profile_name)
            if metrics:
                results.append(metrics)
        
        print(f"\n{'='*70}\n")
    
    # Generate comparison report
    if results:
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70 + "\n")
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('best_coverage', ascending=False)
        
        print("Ranking by Best Coverage:")
        print("-" * 70)
        for idx, row in df_results.iterrows():
            print(f"{row['profile']:20s} | Coverage: {row['best_coverage']:6.2f}% "
                  f"| Diversity: {row['final_diversity']:.3f} "
                  f"| Epoch: {row['best_coverage_epoch']:3.0f}")
        
        # Save results
        output_file = Path('outputs') / f'profile_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_results.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
    else:
        print("\n⚠️  No successful training runs to compare")

if __name__ == '__main__':
    main()
