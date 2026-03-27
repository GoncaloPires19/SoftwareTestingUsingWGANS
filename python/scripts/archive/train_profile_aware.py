"""
Profile-Aware WGAN Trainer - Main orchestrator for profile-based training.

Integrates:
- Training profiles (training_profiles_advanced.py)
- Dataset expansion (dataset_expander.py)
- Coverage tracking (instrumentation.py)
- Coverage analysis (coverage_analysis.py)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pandas.errors import ParserError

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training_profiles_advanced import (
    ProfileType, 
    get_profile_config,
    PROFILE_DESCRIPTIONS
)
from dataset_expander import DatasetExpander, ProgramMetrics
from instrumentation import TrainingInstrumentation, CoverageReporter
from coverage_analysis import CoverageAnalyzer, EpochCoverageTracker
from model import Generator, Critic
from utils import compute_gradient_penalty


# Use strictly cleaned dataset (outliers >1000 and negatives removed)
DEFAULT_CLEANED_DATA_PATH = 'python/data/junit_data_cleaned_strict.csv'
DEFAULT_RAW_DATA_PATH = 'python/data/junit_data.csv'


class ProfileAwareTrainer:
    """
    Main training orchestrator with profile support.
    
    Integrates all components:
    - Load profile configuration
    - Prepare expanded dataset
    - Track coverage metrics
    - Run WGAN training
    - Generate reports
    """
    
    def __init__(self, 
                 profile_name: str,
                 output_dir: Path = None,
                 device: str = None,
                 seed: int = 42):
        """
        Initialize trainer.
        
        Args:
            profile_name: Profile to use (str or ProfileType)
            output_dir: Output directory for models and reports
            device: 'cuda' or 'cpu'
            seed: Random seed for reproducibility
        """
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.seed = seed
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load profile
        if isinstance(profile_name, str):
            self.profile_name = profile_name
            self.profile = get_profile_config(profile_name)
        else:
            self.profile_name = profile_name.name
            self.profile = profile_name
        
        print(f"\n✓ Loaded {self.profile_name} profile")
        print(f"  - Epochs: {self.profile.epochs}")
        print(f"  - Batch Size: {self.profile.batch_size}")
        print(f"  - Learning Rate: {self.profile.learning_rate}")
        print(f"  - Lambda GP: {self.profile.lambda_gp}")
        print(f"  - N Critic: {self.profile.n_critic}")
        
        # Setup output directory
        self.output_dir = Path(output_dir) if output_dir else Path('./outputs')
        self.run_dir = self.output_dir / f"{self.profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup instrumentation
        self.instrumentation = TrainingInstrumentation(self.run_dir)
        self.reporter = CoverageReporter(self.instrumentation)
        
        # Models (will be initialized in setup)
        self.generator = None
        self.critic = None
        self.gen_optimizer = None
        self.critic_optimizer = None

    def _load_metrics_csv(self, csv_path: Path) -> np.ndarray:
        """
        Load metrics CSV expecting at least 3 numeric columns.

        Tries a strict pandas read first. If parser fails (inconsistent column counts),
        falls back to line-by-line parsing and keeps rows with >=3 numeric values.
        """
        try:
            df = pd.read_csv(csv_path, header=None)
            if df.shape[1] < 3:
                raise ValueError(f"CSV has only {df.shape[1]} columns; expected at least 3")

            data = df.iloc[:, :3].apply(pd.to_numeric, errors='coerce').dropna().values
            if len(data) == 0:
                raise ValueError("No valid numeric rows with at least 3 columns found")

            return data.astype(np.float32)

        except ParserError:
            recovered_rows = []
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as file_handle:
                for raw_line in file_handle:
                    line = raw_line.strip()
                    if not line:
                        continue

                    parts = [item.strip() for item in line.split(',') if item.strip() != '']
                    if len(parts) < 3:
                        continue

                    try:
                        values = [float(parts[0]), float(parts[1]), float(parts[2])]
                    except ValueError:
                        continue

                    if all(np.isfinite(values)):
                        recovered_rows.append(values)

            if not recovered_rows:
                raise ValueError(f"Could not recover valid 3-column rows from {csv_path}")

            return np.array(recovered_rows, dtype=np.float32)
        
    def setup_dataset(self,
                      use_expanded: bool = True,
                      data_path: str = DEFAULT_CLEANED_DATA_PATH,
                      allow_synthetic_fallback: bool = False) -> DataLoader:
        """
        Prepare dataset for training.
        
        Args:
            use_expanded: If True, try to use expanded dataset; else use synthetic
            
        Returns:
            DataLoader with dataset
        """
        print("\n✓ Setting up dataset...")
        
        # Try to load real data from candidate paths
        data = None
        errors = []
        candidate_paths = [Path(data_path)]

        if use_expanded:
            for candidate in [Path(DEFAULT_CLEANED_DATA_PATH), Path(DEFAULT_RAW_DATA_PATH)]:
                if candidate not in candidate_paths:
                    candidate_paths.append(candidate)

        for candidate_path in candidate_paths:
            try:
                if not candidate_path.exists():
                    errors.append(f"{candidate_path}: file not found")
                    continue

                print(f"  - Loading from {candidate_path}...")
                data = self._load_metrics_csv(candidate_path)
                print(f"  - Loaded {len(data)} test samples from CSV")
                break
            except Exception as e:
                errors.append(f"{candidate_path}: {type(e).__name__}: {e}")

        if data is None and not allow_synthetic_fallback:
            error_message = "\n    ".join(errors) if errors else "No candidate paths were attempted"
            raise RuntimeError(
                "Failed to load real training data.\n"
                f"    {error_message}\n"
                "Use --allow-synthetic-fallback only for smoke tests."
            )
        
        if data is None:
            print("  - Generating synthetic data...")
            # Generate synthetic data with reasonable dimensions
            rng = np.random.default_rng(self.seed)
            num_methods = rng.integers(5, 50, size=300)
            num_branches = rng.integers(10, 100, size=300)
            num_statements = rng.integers(20, 200, size=300)
            data = np.column_stack([num_methods, num_branches, num_statements]).astype(np.float32)
            print(f"  - Generated {len(data)} synthetic test samples")
        
        data = np.array(data, dtype=np.float32)
        
        # Normalize data to [-1, 1]
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        # Avoid division by zero
        data_range = np.where(data_max - data_min > 0, data_max - data_min, 1.0)
        data_norm = 2 * (data - data_min) / data_range - 1
        
        # Store normalization params
        self.norm_params = {'min': data_min, 'max': data_max}
        
        # Update instrumentation with actual max values from dataset
        self.instrumentation.max_methods = float(data_max[0])
        self.instrumentation.max_branches = float(data_max[1])
        self.instrumentation.max_statements = float(data_max[2])
        
        # Create dataset
        dataset = TensorDataset(torch.FloatTensor(data_norm))
        dataloader = DataLoader(
            dataset,
            batch_size=self.profile.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        print(f"  - DataLoader created: {len(dataloader)} batches of {self.profile.batch_size}")
        return dataloader
    
    def setup_models(self):
        """Initialize Generator and Critic models."""
        print(f"\n✓ Setting up models...")
        
        self.generator = Generator(z_dim=100, output_dim=3).to(self.device)
        self.critic = Critic(input_dim=3).to(self.device)
        
        # Optimizers
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.profile.learning_rate,
            betas=(0.5, 0.999)
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.profile.learning_rate * 0.5,
            betas=(0.5, 0.999)
        )
        
        print(f"  - Generator: {sum(p.numel() for p in self.generator.parameters())} params")
        print(f"  - Critic: {sum(p.numel() for p in self.critic.parameters())} params")
    
    def train(self, dataloader: DataLoader):
        """
        Main training loop.
        
        Args:
            dataloader: DataLoader with training data
        """
        print(f"\n{'='*60}")
        print(f"Starting training: {self.profile_name}")
        print(f"{'='*60}\n")
        
        self.setup_models()
        
        # Learning rate scheduler
        def get_lr(epoch):
            """Decay learning rate after warmup."""
            warmup = 10
            if epoch < warmup:
                return 1.0
            decay_rate = 0.98
            return decay_rate ** (epoch - warmup)
        
        # Training metrics
        training_metrics = {
            'epoch': [],
            'gen_loss': [],
            'critic_loss': [],
            'line_cov': [],
            'branch_cov': [],
            'diversity': []
        }
        
        # Main training loop
        for epoch in range(self.profile.epochs):
            # Update learning rates
            lr_scale = get_lr(epoch)
            for param_group in self.gen_optimizer.param_groups:
                param_group['lr'] = self.profile.learning_rate * lr_scale
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = self.profile.learning_rate * 0.5 * lr_scale
            
            epoch_gen_loss = 0.0
            epoch_critic_loss = 0.0
            
            # Train critic multiple times per epoch
            n_critic = self.profile.n_critic
            
            for batch_idx, (real_data,) in enumerate(dataloader):
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # Train Critic
                for _ in range(n_critic):
                    self.critic_optimizer.zero_grad()
                    
                    # Generate fake data
                    z = torch.randn(batch_size, 100, device=self.device)
                    fake_data = self.generator(z)
                    
                    # Critic scores
                    real_score = self.critic(real_data)
                    fake_score = self.critic(fake_data.detach())
                    
                    # Wasserstein loss
                    critic_loss = fake_score.mean() - real_score.mean()
                    
                    # Gradient penalty
                    gp = compute_gradient_penalty(
                        self.critic, real_data, fake_data,
                        device=self.device,
                        batch_size=batch_size,
                        lambda_gp=self.profile.lambda_gp
                    )
                    critic_loss += gp
                    
                    critic_loss.backward()
                    self.critic_optimizer.step()
                    
                    epoch_critic_loss += critic_loss.item()
                
                # Train Generator
                self.gen_optimizer.zero_grad()
                
                z = torch.randn(batch_size, 100, device=self.device)
                fake_data = self.generator(z)
                fake_score = self.critic(fake_data)
                
                gen_loss = -fake_score.mean()
                gen_loss.backward()
                self.gen_optimizer.step()
                
                epoch_gen_loss += gen_loss.item()
            
            # Average losses
            num_batches = len(dataloader)
            avg_gen_loss = epoch_gen_loss / num_batches  # Generator trains once per batch
            avg_critic_loss = epoch_critic_loss / (num_batches * n_critic)  # Critic trains n_critic times per batch
            
            # Track coverage
            with torch.no_grad():
                z = torch.randn(256, 100, device=self.device)
                synthetic_tests_norm = self.generator(z)
                
                # Denormalize synthetic tests to original scale for coverage calculation
                data_min = torch.tensor(self.norm_params['min'], device=self.device)
                data_max = torch.tensor(self.norm_params['max'], device=self.device)
                data_range = data_max - data_min
                # synthetic_tests_norm is in [-1, 1], convert back to original scale
                synthetic_tests = (synthetic_tests_norm + 1) / 2 * data_range + data_min
                
                # Clip negative values to 0 (test metrics should be non-negative)
                synthetic_tests = torch.clamp(synthetic_tests, min=0.0)
            
            coverage_metrics = self.instrumentation.track_epoch_coverage(
                epoch,
                synthetic_tests.cpu(),
                gen_loss=avg_gen_loss,
                critic_loss=avg_critic_loss
            )
            
            # Record metrics
            training_metrics['epoch'].append(epoch)
            training_metrics['gen_loss'].append(avg_gen_loss)
            training_metrics['critic_loss'].append(avg_critic_loss)
            training_metrics['line_cov'].append(coverage_metrics['line_coverage'])
            training_metrics['branch_cov'].append(coverage_metrics['branch_coverage'])
            training_metrics['diversity'].append(coverage_metrics['path_diversity'])
            
            # Print progress
            if (epoch + 1) % max(1, self.profile.epochs // 10) == 0:
                print(f"Epoch {epoch+1:3d}/{self.profile.epochs}: "
                      f"Gen Loss={avg_gen_loss:7.4f} | "
                      f"Critic Loss={avg_critic_loss:7.4f} | "
                      f"Coverage={coverage_metrics['line_coverage']:6.2f}% | "
                      f"Diversity={coverage_metrics['path_diversity']:.3f}")
        
        # Save results
        self._save_results(training_metrics)
    
    def _save_results(self, metrics: Dict):
        """Save training results and models."""
        print(f"\n{'='*60}")
        print(f"Training Complete - Saving Results")
        print(f"{'='*60}\n")
        
        # Save models
        model_dir = self.run_dir / 'models'
        model_dir.mkdir(exist_ok=True)
        
        torch.save(self.generator.state_dict(),
                  model_dir / 'generator_final.pth')
        torch.save(self.critic.state_dict(),
                  model_dir / 'critic_final.pth')
        print(f"✓ Models saved to {model_dir}")
        
        # Save training metrics
        metrics_df = pd.DataFrame(metrics)
        metrics_file = self.run_dir / 'training_metrics.csv'
        metrics_df.to_csv(metrics_file, index=False)
        print(f"✓ Training metrics saved to {metrics_file}")
        
        # Save coverage trajectory
        self.instrumentation.export_trajectory_csv('coverage_trajectory.csv')
        self.instrumentation.export_json('coverage_history.json')
        
        # Generate report
        report = self.reporter.generate_report(self.profile_name, self.run_dir)
        
        # Save config
        config_file = self.run_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump({
                'profile': self.profile_name,
                'seed': self.seed,
                'epochs': self.profile.epochs,
                'batch_size': self.profile.batch_size,
                'learning_rate': self.profile.learning_rate,
                'description': PROFILE_DESCRIPTIONS.get(self.profile_name, "")
            }, f, indent=2)
        print(f"✓ Config saved to {config_file}")
        
        print(f"\n✓ All results saved to: {self.run_dir}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Training Summary: {self.profile_name}")
        print(f"{'='*60}")
        print(f"Run Directory: {self.run_dir}")
        print(f"Final Coverage: {metrics['line_cov'][-1]:.2f}%")
        print(f"Final Diversity: {metrics['diversity'][-1]:.3f}")
        print(f"Final Gen Loss: {metrics['gen_loss'][-1]:.4f}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Train WGAN with profile support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python python/scripts/train_profile_aware.py --profile coverage-seeking
  python python/scripts/train_profile_aware.py --profile vulnerability-focused --gpu
  python python/scripts/train_profile_aware.py --profile hybrid-realistic --epochs 10 --test
        """
    )
    
    parser.add_argument('--profile',
        choices=['coverage-seeking', 'vulnerability-focused', 
                 'generalization-stress', 'hybrid-realistic'],
        required=True,
        help='Training profile to use'
    )
    
    parser.add_argument('--output', type=str, default='./outputs',
        help='Output directory for models and reports'
    )
    
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument('--gpu', action='store_true',
        help='Use GPU if available'
    )
    
    parser.add_argument('--epochs', type=int, default=None,
        help='Override profile epochs'
    )
    
    parser.add_argument('--test', action='store_true',
        help='Run short test (2 epochs) instead of full training'
    )

    parser.add_argument('--data-path', type=str, default=DEFAULT_CLEANED_DATA_PATH,
        help='Path to CSV training data (expects at least 3 numeric columns)'
    )

    parser.add_argument('--allow-synthetic-fallback', action='store_true',
        help='Allow synthetic data generation if CSV loading fails (smoke-test only)'
    )
    
    parser.add_argument('--no-expand', action='store_true',
        help='Use base dataset instead of expanded'
    )
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    # Create trainer
    trainer = ProfileAwareTrainer(
        profile_name=args.profile,
        output_dir=args.output,
        device=device,
        seed=args.seed
    )
    
    # Override epochs if specified
    if args.epochs:
        trainer.profile.epochs = args.epochs
    
    if args.test:
        trainer.profile.epochs = 2
        print("\n⚠️  Test mode: Running 2 epochs only")
    
    # Setup dataset
    dataloader = trainer.setup_dataset(
        use_expanded=not args.no_expand,
        data_path=args.data_path,
        allow_synthetic_fallback=args.allow_synthetic_fallback
    )
    
    # Run training
    trainer.train(dataloader)


if __name__ == '__main__':
    main()
