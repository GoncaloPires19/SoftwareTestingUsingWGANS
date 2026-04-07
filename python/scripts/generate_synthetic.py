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
from typing import Optional

from model import Generator
from junit_datasets import load_normalization_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_latest_trained_model(explicit_model: Optional[str]) -> Optional[Path]:
    """Resolve a generator checkpoint path, favoring the latest trained output."""
    if explicit_model:
        return Path(explicit_model)

    # Backward-compatible default first.
    legacy_default = Path("models/generator_final.pth")
    if legacy_default.exists():
        return legacy_default

    outputs_root = Path("outputs")
    if not outputs_root.exists():
        return None

    preferred_names = [
        "generator_best_by_robust_v2.pth",
        "generator_best_by_coverage.pth",
        "generator_best_by_robust.pth",
        "generator_final.pth",
    ]

    candidates = []
    for run_dir in outputs_root.iterdir():
        models_dir = run_dir / "models"
        if not models_dir.exists() or not models_dir.is_dir():
            continue
        for filename in preferred_names:
            model_path = models_dir / filename
            if model_path.exists():
                candidates.append(model_path)
                break

    if not candidates:
        return None

    # Most recent checkpoint wins.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_norm_stats_path(explicit_stats: Optional[str], model_path: Path) -> Optional[Path]:
    """Resolve normalization stats path from explicit argument, model run config, or defaults."""
    candidates = []
    if explicit_stats:
        candidates.append(Path(explicit_stats))
    candidates.append(Path("python/data/junit_data_cleaned_norm_stats.json"))

    candidates.extend(_extract_norm_stats_candidates_from_run_config(model_path))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _extract_norm_stats_candidates_from_run_config(model_path: Path) -> list[Path]:
    """Extract candidate stats paths from a run config near the selected model."""
    if model_path.parent.name != "models":
        return []

    config_path = model_path.parent.parent / "config.json"
    if not config_path.exists():
        return []

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            run_config = json.load(f)
    except Exception as exc:
        logger.warning(f"Could not parse run config for norm stats: {exc}")
        return []

    candidates = []
    for key in ("norm_stats_path", "normalization_stats_path"):
        if key in run_config:
            candidates.append(Path(run_config[key]))
    return candidates


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
                logger.error("   Cannot denormalize without actual min/max values!")
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
        include_header: bool = False,
        include_metadata: bool = False,
        keep_extra_features: bool = False,
    ) -> None:
        """
        Save synthetic samples to CSV
        
        Args:
            synthetic_features: (N, 3 or 4) array of features (methods, branches, statements, [method_coverage])
            output_path: Where to save CSV
            norm_stats: Normalization statistics for denormalization
            denormalize: Whether to denormalize features
            include_header: Whether to write CSV header row
            include_metadata: Whether to append metadata columns (test_id/source/epoch)
            keep_extra_features: Keep features beyond the 3 core schema columns
        """
        
        features = synthetic_features.copy()
        
        if denormalize:
            features = self.denormalize_features(features, norm_stats)
        
        # Canonical schema is 3 columns. Keep extras only when explicitly requested.
        if not keep_extra_features and features.shape[1] > 3:
            logger.info("Dropping extra generated feature columns to keep 3-column schema compatibility")
            features = features[:, :3]

        if features.shape[1] >= 4:
            columns = ["methods", "branches", "statements", "method_coverage"]
        else:
            columns = ["methods", "branches", "statements"]
        
        df = pd.DataFrame(features, columns=columns)
        
        if include_metadata:
            df["test_id"] = [f"synthetic_{i:06d}" for i in range(len(df))]
            df["source"] = "wgan"
            df["epoch"] = 0  # Placeholder
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, header=include_header)
        
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
        default=None,
        help="Path to trained generator (auto-resolved from latest run when omitted)"
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
        default=None,
        help="Path to normalization statistics JSON (auto-resolved when omitted)"
    )

    parser.add_argument(
        "--with-header",
        action="store_true",
        help="Write CSV header row (default is headerless for schema compatibility)"
    )

    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Append metadata columns (test_id/source/epoch)"
    )

    parser.add_argument(
        "--keep-extra-features",
        action="store_true",
        help="Keep generated features beyond the canonical 3-column schema"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = _resolve_latest_trained_model(args.model)
    if model_path is None or not model_path.exists():
        logger.error("Model not found. Provide --model or run training first.")
        logger.error("Expected either models/generator_final.pth or outputs/<run>/models/generator_*.pth")
        return
    
    # Load normalization statistics
    norm_stats = None
    if args.denormalize:
        stats_path = _resolve_norm_stats_path(args.norm_stats, model_path)
        if stats_path and stats_path.exists():
            logger.info(f"📂 Loading normalization stats from {stats_path}")
            norm_stats = load_normalization_stats(stats_path)
        else:
            logger.warning("⚠ Normalization stats not found")
            logger.warning("  Synthetic data will NOT be denormalized correctly!")
    
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
        denormalize=args.denormalize,
        include_header=args.with_header,
        include_metadata=args.include_metadata,
        keep_extra_features=args.keep_extra_features,
    )
    
    logger.info("\n✅ Generation complete!")


if __name__ == "__main__":
    main()
