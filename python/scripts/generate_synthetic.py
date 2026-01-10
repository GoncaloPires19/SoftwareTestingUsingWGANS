"""
Generate synthetic test cases using trained WGAN generator
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
from data_loader import UnifiedDataLoader


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
            output_dim: Output feature dimension
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
            (num_samples, 3) array of normalized features
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
    
    def denormalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Denormalize features back to original scale
        
        Assumes features were normalized to [-1, 1] during training
        Approximately reverses the normalization
        """
        # Inverse of [-1, 1] mapping
        # y = 2*x - 1, so x = (y + 1) / 2
        features_01 = (features + 1) / 2
        
        # Scale to approximate ranges
        # Methods: 1-500
        # Branches: 1-1000
        # Statements: 1-2000
        
        denorm = features_01.copy()
        denorm[:, 0] = features_01[:, 0] * 500 + 1
        denorm[:, 1] = features_01[:, 1] * 1000 + 1
        denorm[:, 2] = features_01[:, 2] * 2000 + 1
        
        return np.clip(denorm, 1, None).astype(int)
    
    def save_to_csv(
        self,
        synthetic_features: np.ndarray,
        output_path: Path,
        denormalize: bool = True,
    ) -> None:
        """
        Save synthetic samples to CSV
        
        Args:
            synthetic_features: (N, 3) array of features
            output_path: Where to save CSV
            denormalize: Whether to denormalize features
        """
        
        features = synthetic_features.copy()
        
        if denormalize:
            features = self.denormalize_features(features)
        
        # Create DataFrame
        df = pd.DataFrame(
            features,
            columns=["methods", "branches", "statements"]
        )
        
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
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # Generate
    generator = SyntheticTestGenerator(
        model_path,
        device=args.device
    )
    
    synthetic = generator.generate_samples(args.num_samples)
    generator.save_to_csv(synthetic, Path(args.output), denormalize=args.denormalize)
    
    logger.info("\n✅ Generation complete!")


if __name__ == "__main__":
    main()
