#!/usr/bin/env python3
"""
WGAN-GP Training with Optimized Hyperparameters

Fixes identified issues from performance analysis:
- Increased lambda_gp (10.0) for stricter Lipschitz constraint (prevent mode collapse)
- Reduced learning_rate (0.00002) for smoother, more stable learning
- Increased n_critic (10) for better feature balance training
- Increased epochs (100) for convergence
- Reduced batch_size (8) for more frequent updates

Performance Target: 70%+ quality score
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from config import TrainingConfig, ModelConfig, set_random_seed
from model import Generator, Critic
from utils import compute_gradient_penalty

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WGANGPTrainer:
    """WGAN-GP trainer with improved hyperparameters"""
    
    def __init__(self, config: TrainingConfig, model_config: ModelConfig, device: str = "cpu"):
        self.config = config
        self.model_config = model_config
        self.device = torch.device(device)
        
        # Initialize models
        self.generator = Generator(
            z_dim=model_config.z_dim,
            output_dim=model_config.output_dim
        ).to(self.device)
        
        self.critic = Critic(
            input_dim=model_config.output_dim,
            output_dim=1
        ).to(self.device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        self.c_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        # Training history
        self.history = {
            'gen_loss': [],
            'critic_loss': [],
            'gp_loss': [],
            'epoch': []
        }
        
        logger.info("✅ WGAN-GP trainer initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Generator: {model_config.z_dim} → {model_config.output_dim}")
        logger.info(f"   λ_gp: {config.lambda_gp} (Lipschitz penalty)")
        logger.info(f"   n_critic: {config.n_critic} (critic iterations)")
        logger.info(f"   lr: {config.learning_rate}")
        logger.info(f"   batch_size: {config.batch_size}")
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        gen_losses = []
        critic_losses = []
        gp_losses = []
        
        for batch in train_loader:
            # Handle TensorDataset tuple format
            if isinstance(batch, (list, tuple)):
                real_data = batch[0]
            else:
                real_data = batch
            
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # ===== Train Critic (n_critic iterations) =====
            for _ in range(self.config.n_critic):
                self.c_optimizer.zero_grad()
                
                # Real data score
                real_score = self.critic(real_data)
                
                # Synthetic data
                z = torch.randn(batch_size, self.model_config.z_dim, device=self.device)
                synthetic_data = self.generator(z)
                
                # Synthetic data score
                synthetic_score = self.critic(synthetic_data)
                
                # Critic loss (Wasserstein distance)
                critic_loss = synthetic_score.mean() - real_score.mean()
                
                # Gradient penalty
                gp = compute_gradient_penalty(
                    self.critic,
                    real_data,
                    synthetic_data,
                    self.device,
                    batch_size,
                    lambda_gp=self.config.lambda_gp
                )
                
                # Total critic loss
                total_critic_loss = critic_loss + self.config.lambda_gp * gp
                
                total_critic_loss.backward()
                self.c_optimizer.step()
                
                critic_losses.append(critic_loss.item())
                gp_losses.append(gp.item())
            
            # ===== Train Generator =====
            self.g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, self.model_config.z_dim, device=self.device)
            synthetic_data = self.generator(z)
            synthetic_score = self.critic(synthetic_data)
            
            gen_loss = -synthetic_score.mean()
            
            gen_loss.backward()
            self.g_optimizer.step()
            
            gen_losses.append(gen_loss.item())
        
        return np.mean(gen_losses), np.mean(critic_losses), np.mean(gp_losses)
    
    def train(self, train_loader, output_dir: Path):
        """Full training loop"""
        logger.info("\n" + "="*80)
        logger.info("🚀 STARTING OPTIMIZED WGAN-GP TRAINING")
        logger.info("="*80)
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"λ_gp: {self.config.lambda_gp}")
        logger.info(f"n_critic: {self.config.n_critic}")
        logger.info(f"Learning rate: {self.config.learning_rate}\n")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config.epochs):
            gen_loss, critic_loss, gp_loss = self.train_epoch(train_loader)
            
            self.history['epoch'].append(epoch)
            self.history['gen_loss'].append(gen_loss)
            self.history['critic_loss'].append(critic_loss)
            self.history['gp_loss'].append(gp_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"G_loss: {gen_loss:.4f} | "
                    f"C_loss: {critic_loss:.4f} | "
                    f"GP: {gp_loss:.4f}"
                )
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(output_dir / f"checkpoint_epoch_{epoch+1}.pth")
        
        logger.info("\n✅ Training complete!")
        
        # Save final models
        self.save_models(output_dir)
        
        # Save history
        self.save_history(output_dir)
        
        return self.history
    
    def save_models(self, output_dir: Path):
        """Save trained models"""
        gen_path = output_dir / "generator_optimized_final.pth"
        crit_path = output_dir / "critic_optimized_final.pth"
        
        torch.save(self.generator.state_dict(), gen_path)
        torch.save(self.critic.state_dict(), crit_path)
        
        logger.info(f"✅ Models saved:")
        logger.info(f"   Generator: {gen_path}")
        logger.info(f"   Critic: {crit_path}")
    
    def save_checkpoint(self, path: Path):
        """Save training checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'generator': self.generator.state_dict(),
            'critic': self.critic.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'c_optimizer': self.c_optimizer.state_dict(),
        }, path)
    
    def save_history(self, output_dir: Path):
        """Save training history"""
        hist_df = pd.DataFrame(self.history)
        hist_path = output_dir / "training_history_optimized.csv"
        hist_df.to_csv(hist_path, index=False)
        
        logger.info(f"✅ Training history saved: {hist_path}")
        
        # Plot training curves
        self.plot_training_curves(output_dir)
    
    def plot_training_curves(self, output_dir: Path):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Generator loss
        axes[0, 0].plot(self.history['epoch'], self.history['gen_loss'], 'b-')
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Critic loss
        axes[0, 1].plot(self.history['epoch'], self.history['critic_loss'], 'r-')
        axes[0, 1].set_title('Critic Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient penalty
        axes[1, 0].plot(self.history['epoch'], self.history['gp_loss'], 'g-')
        axes[1, 0].set_title('Gradient Penalty Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('GP Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Both losses on same plot
        axes[1, 1].plot(self.history['epoch'], self.history['gen_loss'], 'b-', label='Gen Loss')
        axes[1, 1].plot(self.history['epoch'], self.history['critic_loss'], 'r-', label='Critic Loss')
        axes[1, 1].set_title('Generator vs Critic Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / "training_curves_optimized.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        logger.info(f"✅ Training curves saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Train WGAN-GP with optimized hyperparameters")
    
    parser.add_argument(
        "--data",
        type=str,
        default="python/data/junit_data_cleaned_v2.csv",
        help="Path to training data (real JUnit tests)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/optimized_YYYYMMDD_HHMMSS/)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override (default: uses TrainingConfig.seed)"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info("📂 Loading training data...")
    csv_path = Path(args.data)
    if not csv_path.exists():
        logger.error(f"❌ Data file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path, header=None)
    data = torch.FloatTensor(df.values[:, :3].astype(np.float32))  # Use first 3 features only
    logger.info(f"✅ Loaded {len(data)} samples")
    
    # Initialize trainer configuration first so DataLoader stays consistent with config.
    config = TrainingConfig()
    if args.seed is not None:
        config.seed = args.seed
    set_random_seed(config.seed)

    # Create data loader
    dataset = torch.utils.data.TensorDataset(data)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )

    # Initialize trainer
    model_config = ModelConfig()
    
    trainer = WGANGPTrainer(config, model_config, device=args.device)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"optimized_balanced_{timestamp}"
    
    # Train
    history = trainer.train(train_loader, output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("✅ OPTIMIZED TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Generator: generator_optimized_final.pth")
    logger.info(f"Critic: critic_optimized_final.pth")
    logger.info(f"History: training_history_optimized.csv")
    logger.info("\nNext step: Generate synthetic data and analyze performance")


if __name__ == "__main__":
    main()
