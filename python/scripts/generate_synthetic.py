"""
Generate synthetic test cases using trained WGAN generator

CRITICAL: Uses actual min/max statistics from training data (NOT hardcoded ranges).
This ensures synthetic data is denormalized to the same scale as real test data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
import pandas as pd
import json
import logging

from model import Generator
from junit_datasets import load_normalization_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticTestGenerator:
    """Generate synthetic tests using trained generator"""
    
    def __init__(
        self,
        generator_path: Path,
        z_dim: int = 100,
        output_dim: int = 3,
        device: str = "cpu"
    ):
        """
        Args:
            generator_path: Path to saved generator weights
            z_dim: Latent space dimension
            output_dim: Output feature dimension (4: methods, branches, statements, method_coverage)
            device: "cpu" or "cuda"
        """
        self.device = torch.device(device)
        
        # Load generator
        self.generator = Generator(
            z_dim=z_dim,
            output_dim=output_dim,
        ).to(self.device)
        
        self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
        self.generator.eval()
        
        logger.info(f"✅ Loaded generator from {generator_path}")
    
    def generate_samples(self, num_samples: int = 1000) -> np.ndarray:
        """
        Generate synthetic test samples
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            (num_samples, 4) array of normalized features
        """
        logger.info(f"🧬 Generating {num_samples} synthetic samples...")
        
        with torch.no_grad():
            z = torch.randn(num_samples, 100, device=self.device)
            synthetic = self.generator(z)
            synthetic = synthetic.cpu().numpy()
        
        # Clip to [-1, 1]
        synthetic = np.clip(synthetic, -1, 1)
        
        logger.info(f"✅ Generated samples shape: {synthetic.shape}")
        logger.info(f"   Range: [{synthetic.min():.4f}, {synthetic.max():.4f}]")
        
        return synthetic
    
    def denormalize_features(self, features: np.ndarray, norm_stats: dict = None) -> np.ndarray:
        """
        Denormalize features back to original scale using actual training statistics.
        
        CRITICAL FIX: Uses actual min/max from training data, NOT hardcoded ranges!
        
        Args:
            features (np.ndarray): Normalized features in [-1, 1] range
            norm_stats (dict): Normalization statistics from training
                             If None, attempts to load from default location
        
        Returns:
            np.ndarray: Denormalized features in original scale
        """
        if norm_stats is None:
            # Try to load from default location
            default_stats_path = Path("python/data/junit_data_cleaned_norm_stats.json")
            if default_stats_path.exists():
                logger.info(f"📂 Loading normalization stats from {default_stats_path}")
                norm_stats = load_normalization_stats(default_stats_path)
            else:
                logger.error(f"❌ Normalization stats not found at {default_stats_path}")
                logger.error(f"   Cannot denormalize without actual min/max values!")
                raise FileNotFoundError(
                    f"Normalization statistics file not found. "
                    f"Expected at: {default_stats_path}\n"
                    f"Run training first to generate statistics."
                )
        
        denorm = features.copy()
        
        # Use actual statistics from training data
        # Get num_features from stats or infer from features dict
        if 'num_features' in norm_stats:
            num_features = norm_stats['num_features']
        elif 'features' in norm_stats:
            num_features = len(norm_stats['features'])
        else:
            # Old format fallback
            num_features = len(norm_stats.get('min', [0, 0, 0]))
        
        for col_idx in range(min(denorm.shape[1] if len(denorm.shape) > 1 else 1, num_features)):
            feature_stats = norm_stats['features'][f'feature_{col_idx}']
            feature_min = feature_stats['min']
            feature_max = feature_stats['max']
            feature_range = feature_max - feature_min
            
            if feature_range == 0:
                logger.warning(f"Column {col_idx} has zero range, using value {feature_min}")
                if len(denorm.shape) == 1:
                    denorm = np.full_like(denorm, feature_min)
                else:
                    denorm[:, col_idx] = feature_min
            else:
                # Inverse of normalization: x = (x_norm + 1) / 2 * range + min
                if len(denorm.shape) == 1:
                    denorm = (features + 1) / 2 * feature_range + feature_min
                else:
                    denorm[:, col_idx] = (features[:, col_idx] + 1) / 2 * feature_range + feature_min
            
            logger.info(f"   Feature {col_idx}: denormalized to range [{feature_min:.2f}, {feature_max:.2f}]")
        
        return denorm
    
    def save_to_csv(
        self,
        synthetic_features: np.ndarray,
        output_path: Path,
        norm_stats: dict = None,
        denormalize: bool = True,
    ) -> None:
        """
        Save synthetic samples to CSV
        
        Args:
            synthetic_features: (N, 3 or 4) array of features (methods, branches, statements, [method_coverage])
            output_path: Where to save CSV
            norm_stats: Normalization statistics for denormalization
            denormalize: Whether to denormalize features
        """
        
        features = synthetic_features.copy()
        
        if denormalize:
            features = self.denormalize_features(features, norm_stats)
        
        # Create DataFrame with 3 or 4 features depending on output
        if features.shape[1] == 4:
            columns = ["methods", "branches", "statements", "method_coverage"]
        else:
            columns = ["methods", "branches", "statements"]
        
        df = pd.DataFrame(features, columns=columns)
        
        # Add metadata
        df["test_id"] = [f"synthetic_{i:06d}" for i in range(len(df))]
        df["source"] = "wgan"
        df["epoch"] = 0  # Placeholder
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Saved {len(df)} samples to {output_path}")
        logger.info(f"   Features: methods={df['methods'].mean():.1f} ± {df['methods'].std():.1f}")
        logger.info(f"             branches={df['branches'].mean():.1f} ± {df['branches'].std():.1f}")
        logger.info(f"             statements={df['statements'].mean():.1f} ± {df['statements'].std():.1f}")
        if "method_coverage" in df.columns:
            logger.info(f"             method_coverage={df['method_coverage'].mean():.4f} ± {df['method_coverage'].std():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic tests with WGAN")
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/generator_final.pth",
        help="Path to trained generator"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/synthetic_samples.csv",
        help="Output CSV file"
    )
    
    parser.add_argument(
        "--denormalize",
        action="store_true",
        default=True,
        help="Denormalize features to original scale"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Compute device"
    )
    
    parser.add_argument(
        "--norm-stats",
        type=str,
        default="python/data/junit_data_cleaned_norm_stats.json",
        help="Path to normalization statistics JSON"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # Load normalization statistics
    norm_stats = None
    if args.denormalize:
        stats_path = Path(args.norm_stats)
        if stats_path.exists():
            logger.info(f"📂 Loading normalization stats from {stats_path}")
            norm_stats = load_normalization_stats(stats_path)
        else:
            logger.warning(f"⚠ Normalization stats not found at {stats_path}")
            logger.warning(f"  Synthetic data will NOT be denormalized correctly!")
    
    # Generate
    generator = SyntheticTestGenerator(
        model_path,
        device=args.device
    )
    
    synthetic = generator.generate_samples(args.num_samples)
    generator.save_to_csv(
        synthetic,
        Path(args.output),
        norm_stats=norm_stats,
        denormalize=args.denormalize
    )
    
    logger.info("\n✅ Generation complete!")


if __name__ == "__main__":
    main()
