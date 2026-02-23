"""
Analyze coverage evolution during training by generating samples at each checkpoint.
This helps identify if extended training improves max value coverage.
"""

import torch
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model import Generator
from config import ModelConfig

def denormalize(normalized_data, norm_stats):
    """Denormalize from [-1, 1] to original scale."""
    min_vals = np.array(norm_stats['min'])
    max_vals = np.array(norm_stats['max'])
    
    # From [-1, 1] to [0, 1]
    data_01 = (normalized_data + 1) / 2
    # From [0, 1] to original scale
    return data_01 * (max_vals - min_vals) + min_vals

def generate_from_checkpoint(model_path, norm_stats, num_samples=500, device='cpu'):
    """Generate synthetic samples from a checkpoint."""
    config = ModelConfig()
    
    # Load generator
    generator = Generator(
        z_dim=config.z_dim,
        output_dim=config.output_dim
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if 'generator' in checkpoint:
            # Full training checkpoint
            generator.load_state_dict(checkpoint['generator'])
        elif 'model_state_dict' in checkpoint:
            # Single model checkpoint
            generator.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume it's just the state dict
            generator.load_state_dict(checkpoint)
    else:
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    
    # Generate samples
    with torch.no_grad():
        z = torch.randn(num_samples, config.z_dim).to(device)
        synthetic_normalized = generator(z).cpu().numpy()
    
    # Denormalize
    synthetic_denorm = denormalize(synthetic_normalized, norm_stats)
    
    return synthetic_denorm

def analyze_checkpoint_coverage(checkpoint_path, norm_stats, real_max_values):
    """Analyze max values and coverage for a checkpoint."""
    synthetic = generate_from_checkpoint(checkpoint_path, norm_stats)
    
    max_values = synthetic.max(axis=0)
    coverage = (max_values / real_max_values) * 100
    
    return {
        'methods_max': float(max_values[0]),
        'branches_max': float(max_values[1]),
        'statements_max': float(max_values[2]),
        'methods_coverage': float(coverage[0]),
        'branches_coverage': float(coverage[1]),
        'statements_coverage': float(coverage[2]),
        'mean_coverage': float(coverage.mean())
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints-dir', type=str, required=True,
                        help='Directory containing checkpoint_epoch_*.pth files')
    parser.add_argument('--norm-stats', type=str, 
                        default='python/data/junit_data_cleaned_norm_stats.json',
                        help='Path to normalization stats JSON')
    parser.add_argument('--output', type=str, default='outputs/coverage_evolution.csv',
                        help='Output CSV file')
    args = parser.parse_args()
    
    # Load normalization stats
    with open(args.norm_stats, 'r') as f:
        norm_stats = json.load(f)
    
    real_max_values = np.array(norm_stats['max'])
    
    # Find all checkpoints
    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoint_files = sorted(checkpoints_dir.glob('checkpoint_epoch_*.pth'))
    
    if len(checkpoint_files) == 0:
        print(f"No checkpoints found in {checkpoints_dir}")
        # Also check for generator_final.pth
        final_path = checkpoints_dir / 'generator_final.pth'
        if final_path.exists():
            checkpoint_files = [final_path]
        else:
            return
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    results = []
    for ckpt_path in checkpoint_files:
        # Extract epoch number from filename
        if 'epoch_' in ckpt_path.name:
            epoch = int(ckpt_path.name.split('epoch_')[1].split('.')[0])
        elif 'final' in ckpt_path.name:
            epoch = -1  # Use -1 for final model
        else:
            continue
        
        print(f"Analyzing epoch {epoch}...")
        
        try:
            stats = analyze_checkpoint_coverage(ckpt_path, norm_stats, real_max_values)
            stats['epoch'] = epoch
            stats['checkpoint'] = ckpt_path.name
            results.append(stats)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Save results
    df = pd.DataFrame(results)
    df = df.sort_values('epoch')
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to {output_path}")
    print("\nCoverage Evolution Summary:")
    print(df[['epoch', 'methods_coverage', 'branches_coverage', 'statements_coverage', 'mean_coverage']].to_string(index=False))
    
    # Show trend
    if len(df) > 1:
        first = df.iloc[0]
        last = df.iloc[-1]
        print("\n--- TREND ANALYSIS ---")
        print(f"Methods coverage:    {first['methods_coverage']:.1f}% → {last['methods_coverage']:.1f}% "
              f"({last['methods_coverage'] - first['methods_coverage']:+.1f}%)")
        print(f"Branches coverage:   {first['branches_coverage']:.1f}% → {last['branches_coverage']:.1f}% "
              f"({last['branches_coverage'] - first['branches_coverage']:+.1f}%)")
        print(f"Statements coverage: {first['statements_coverage']:.1f}% → {last['statements_coverage']:.1f}% "
              f"({last['statements_coverage'] - first['statements_coverage']:+.1f}%)")
        print(f"Mean coverage:       {first['mean_coverage']:.1f}% → {last['mean_coverage']:.1f}% "
              f"({last['mean_coverage'] - first['mean_coverage']:+.1f}%)")
        
        if last['mean_coverage'] < 60:
            print("\n⚠️  Mean coverage still low (<60%). Consider:")
            print("   - Training more epochs")
            print("   - Using coverage-seeking profile")
            print("   - Adjusting lambda_gp or learning rates")

if __name__ == '__main__':
    main()
