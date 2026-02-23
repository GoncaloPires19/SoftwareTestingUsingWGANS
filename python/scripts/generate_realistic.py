"""
Generate synthetic data with realistic value clipping
Ensures all values are within realistic ranges [0, max]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from model import Generator
from config import ModelConfig
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_realistic(n_samples=1000, model_path='python/models/generator_improved.pth'):
    """Generate synthetic test data with realistic clipping"""
    
    model_config = ModelConfig(z_dim=100, output_dim=3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load generator
    generator = Generator(z_dim=model_config.z_dim, output_dim=model_config.output_dim)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.to(device)
    generator.eval()
    
    # Load normalization stats
    with open('python/data/junit_data_all_norm_stats.json', 'r') as f:
        stats = json.load(f)
    
    # Generate
    with torch.no_grad():
        z = torch.randn(n_samples, model_config.z_dim).to(device)
        synthetic = generator(z).cpu().numpy()
    
    logger.info(f"Generated {n_samples} samples")
    
    # Denormalize
    feature_names = ['methods', 'branches', 'statements', 'method_coverage']
    synthetic_denorm = np.zeros_like(synthetic)
    
    for i in range(4):
        feat_key = f'feature_{i}'
        feat_min = stats['features'][feat_key]['min']
        feat_max = stats['features'][feat_key]['max']
        feat_range = feat_max - feat_min
        
        # Inverse norm: x = (x_norm + 1) / 2 * range + min
        synthetic_denorm[:, i] = (synthetic[:, i] + 1) / 2 * feat_range + feat_min
    
    logger.info(f"\nBefore clipping:")
    for i, name in enumerate(feature_names):
        logger.info(f"  {name}: [{synthetic_denorm[:, i].min():.2f}, {synthetic_denorm[:, i].max():.2f}]")
    
    # **CLIP to realistic ranges** ✅
    synthetic_clipped = synthetic_denorm.copy()
    for i in range(4):
        feat_key = f'feature_{i}'
        feat_min = stats['features'][feat_key]['min']
        feat_max = stats['features'][feat_key]['max']
        
        # Clip to valid range
        synthetic_clipped[:, i] = np.clip(synthetic_clipped[:, i], feat_min, feat_max)
    
    logger.info(f"\nAfter clipping (realistic ranges):")
    for i, name in enumerate(feature_names):
        real_min = stats['features'][f'feature_{i}']['min']
        real_max = stats['features'][f'feature_{i}']['max']
        syn_min = synthetic_clipped[:, i].min()
        syn_max = synthetic_clipped[:, i].max()
        coverage = np.sum((synthetic_clipped[:, i] >= real_min) & (synthetic_clipped[:, i] <= real_max)) / len(synthetic_clipped) * 100
        logger.info(f"  {name}: [{syn_min:.2f}, {syn_max:.2f}] | Real: [{real_min:.2f}, {real_max:.2f}] | Coverage: {coverage:.1f}%")
    
    # Save to CSV
    df = pd.DataFrame(synthetic_clipped, columns=feature_names)
    
    # Add metadata
    df['sample_id'] = [f'synthetic_{i:06d}' for i in range(len(df))]
    df['model'] = 'improved_wgan'
    df['clipping'] = 'yes'
    
    output_file = 'outputs/synthetic_realistic.csv'
    df.to_csv(output_file, index=False)
    
    logger.info(f"\n✓ Saved {len(df)} realistic samples to: {output_file}")
    
    # Summary stats
    logger.info(f"\n📊 Summary Statistics:")
    for i, name in enumerate(feature_names):
        real_mean = stats['features'][f'feature_{i}']['mean']
        syn_mean = synthetic_clipped[:, i].mean()
        error = abs(syn_mean - real_mean) / (real_mean + 1e-8) * 100
        logger.info(f"  {name}: Mean={syn_mean:.2f} (Real: {real_mean:.2f}, Error: {error:.1f}%)")
    
    return synthetic_clipped

if __name__ == "__main__":
    generate_realistic(1000)
