"""
Generate large synthetic test suite (10,000+ samples)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
import pandas as pd
import logging
from model import Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_large_suite(
    model_path: Path = Path("python/models/generator_final.pth"),
    num_samples: int = 10000,
    output_path: Path = Path("python/outputs/synthetic_10k.csv")
) -> pd.DataFrame:
    """
    Generate large synthetic test suite
    """
    
    logger.info("=" * 70)
    logger.info(f"SYNTHETIC TEST GENERATION - {num_samples:,} SAMPLES")
    logger.info("=" * 70)
    
    # Load model
    logger.info(f"\n📦 Loading generator from {model_path}...")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Initialize generator
    device = torch.device("cpu")
    generator = Generator(z_dim=100, output_dim=3).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    logger.info(f"✅ Generator loaded ({sum(p.numel() for p in generator.parameters()):,} params)")
    
    # Generate samples in batches
    logger.info(f"\n🧬 Generating {num_samples:,} samples...")
    
    batch_size = 1000
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    all_samples = []
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            
            # Generate noise
            noise = torch.randn(current_batch_size, 100, device=device)
            
            # Generate samples
            fake_data = generator(noise).cpu().numpy()
            
            all_samples.append(fake_data)
            
            if (batch_idx + 1) % 5 == 0 or batch_idx == num_batches - 1:
                logger.info(f"   Generated {(batch_idx + 1) * batch_size:,} / {num_samples:,} samples")
    
    # Combine all batches
    synthetic_data = np.vstack(all_samples)
    synthetic_data = synthetic_data[:num_samples]  # Trim to exact count
    
    logger.info(f"✅ Generated samples shape: {synthetic_data.shape}")
    
    # Denormalize
    # Assuming data was normalized to [-1, 1]
    # Real data ranges: methods [50-500], branches [100-2000], statements [10-5000]
    denorm_data = synthetic_data.copy()
    
    # Scale from [-1,1] to approximate real ranges
    denorm_data[:, 0] = (denorm_data[:, 0] + 1) / 2 * 450 + 50      # methods: 50-500
    denorm_data[:, 1] = (denorm_data[:, 1] + 1) / 2 * 1900 + 100    # branches: 100-2000
    denorm_data[:, 2] = (denorm_data[:, 2] + 1) / 2 * 4990 + 10     # statements: 10-5000
    
    # Clip to valid ranges
    denorm_data[:, 0] = np.clip(denorm_data[:, 0], 50, 500)
    denorm_data[:, 1] = np.clip(denorm_data[:, 1], 100, 2000)
    denorm_data[:, 2] = np.clip(denorm_data[:, 2], 10, 5000)
    
    # Create DataFrame
    df = pd.DataFrame(
        denorm_data,
        columns=['methods', 'branches', 'statements']
    )
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"\n✅ Saved {len(df):,} synthetic samples to {output_path}")
    
    # Statistics
    logger.info(f"\n📊 Synthetic Data Statistics:")
    logger.info(f"   Methods: {df['methods'].mean():.1f} ± {df['methods'].std():.1f}")
    logger.info(f"   Branches: {df['branches'].mean():.1f} ± {df['branches'].std():.1f}")
    logger.info(f"   Statements: {df['statements'].mean():.1f} ± {df['statements'].std():.1f}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate large synthetic test suite")
    parser.add_argument("--model", type=Path, default=Path("python/models/generator_final.pth"),
                       help="Path to generator model")
    parser.add_argument("--num-samples", type=int, default=10000,
                       help="Number of samples to generate")
    parser.add_argument("--output", type=Path, default=Path("python/outputs/synthetic_10k.csv"),
                       help="Output file path")
    
    args = parser.parse_args()
    
    df = generate_large_suite(
        model_path=args.model,
        num_samples=args.num_samples,
        output_path=args.output
    )
    
    logger.info("\n✅ Synthetic test generation complete!")
