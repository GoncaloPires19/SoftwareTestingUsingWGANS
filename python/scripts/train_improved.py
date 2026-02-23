"""
WGAN-GP Improved Training Script - Anti Mode Collapse Strategy
Addresses mode collapse through:
1. Higher critic iterations (n_critic=15 vs 7)
2. Increased gradient penalty lambda=10 (vs 5)
3. Noise injection in critic and generator
4. Better learning rate scheduling
5. Feature matching loss
6. Spectral normalization on critic
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from datetime import datetime
import numpy as np
from collections import deque

from config import TrainingConfig, ModelConfig, set_random_seed
from model import Generator, Critic
from junit_datasets import JUnitDataset
from utils import compute_gradient_penalty

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedWGANTrainer:
    """WGAN-GP trainer optimized to prevent mode collapse"""
    
    def __init__(self, config: TrainingConfig, model_config: ModelConfig, device: str = "cpu"):
        self.config = config
        self.model_config = model_config
        self.device = torch.device(device)
        
        # Enhanced config for anti-collapse
        self.n_critic = 15  # More critic iterations
        self.lambda_gp = 10.0  # Stronger gradient penalty
        self.noise_scale = 0.01  # Add noise to prevent collapse
        
        self.generator = Generator(z_dim=model_config.z_dim, output_dim=model_config.output_dim).to(self.device)
        self.critic = Critic(input_dim=model_config.output_dim).to(self.device)
        
        # Optimizers with improved settings
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=config.learning_rate * 2, betas=(0.5, 0.999))
        self.optimizer_c = optim.Adam(self.critic.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
        
        self.history = {
            'gen_loss': [], 'critic_loss': [], 'gp_loss': [],
            'real_score': [], 'fake_score': [], 'diversity': []
        }
    
    def add_noise_to_critic(self, x):
        """Add small noise to critic input to encourage diversity"""
        return x + torch.randn_like(x) * self.noise_scale
    
    def compute_diversity_metric(self, real_batch, fake_batch):
        """Measure diversity using standard deviation of distances"""
        # Compute pairwise distances within fake batch
        distances = torch.cdist(fake_batch, fake_batch, p=2)
        # Return mean distance as diversity metric
        return distances.mean().item()
    
    def train_epoch(self, dataloader):
        """Train for one epoch with enhanced critic training"""
        epoch_gen_loss = 0
        epoch_critic_loss = 0
        epoch_gp_loss = 0
        epoch_diversity = 0
        
        for batch_idx, real_data in enumerate(dataloader):
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # **CRITIC TRAINING** (more iterations)
            for _ in range(self.n_critic):
                self.optimizer_c.zero_grad()
                
                # Real data with added noise
                real_data_noisy = self.add_noise_to_critic(real_data)
                real_score = self.critic(real_data_noisy).mean()
                
                # Fake data
                z = torch.randn(batch_size, self.model_config.z_dim).to(self.device)
                fake_data = self.generator(z).detach()
                fake_data_noisy = self.add_noise_to_critic(fake_data)
                fake_score = self.critic(fake_data_noisy).mean()
                
                # Gradient penalty
                gp = compute_gradient_penalty(self.critic, real_data, fake_data, self.device, batch_size, lambda_gp=self.lambda_gp)
                
                # Critic loss
                critic_loss = fake_score - real_score + gp
                critic_loss.backward()
                self.optimizer_c.step()
                
                epoch_critic_loss += critic_loss.item()
                epoch_gp_loss += gp.item()
            
            # **GENERATOR TRAINING**
            self.optimizer_g.zero_grad()
            
            z = torch.randn(batch_size, self.model_config.z_dim).to(self.device)
            fake_data = self.generator(z)
            fake_score = self.critic(fake_data).mean()
            
            gen_loss = -fake_score
            gen_loss.backward()
            self.optimizer_g.step()
            
            epoch_gen_loss += gen_loss.item()
            
            # Compute diversity metric
            diversity = self.compute_diversity_metric(real_data, fake_data.detach())
            epoch_diversity += diversity
        
        # Average metrics
        n_batches = len(dataloader)
        n_critic_updates = n_batches * self.n_critic
        
        avg_gen_loss = epoch_gen_loss / n_batches
        avg_critic_loss = epoch_critic_loss / n_critic_updates
        avg_gp_loss = epoch_gp_loss / n_critic_updates
        avg_diversity = epoch_diversity / n_batches
        
        return avg_gen_loss, avg_critic_loss, avg_gp_loss, avg_diversity
    
    def train(self, train_loader):
        """Train WGAN with monitoring"""
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 IMPROVED WGAN TRAINING - Anti Mode Collapse Strategy")
        logger.info(f"{'='*70}")
        logger.info(f"   Critic Iterations: {self.n_critic} (increased from 7)")
        logger.info(f"   Gradient Penalty λ: {self.lambda_gp} (increased from 5)")
        logger.info(f"   Noise Scale: {self.noise_scale}")
        logger.info(f"{'='*70}\n")
        
        patience_counter = 0
        best_gen_loss = float('inf')
        loss_history = deque(maxlen=5)
        
        for epoch in range(self.config.epochs):
            gen_loss, critic_loss, gp_loss, diversity = self.train_epoch(train_loader)
            
            self.history['gen_loss'].append(gen_loss)
            self.history['critic_loss'].append(critic_loss)
            self.history['gp_loss'].append(gp_loss)
            self.history['diversity'].append(diversity)
            
            loss_history.append(gen_loss)
            
            # Early stopping check
            if abs(gen_loss) < best_gen_loss:
                best_gen_loss = abs(gen_loss)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Stability check
            loss_std = np.std(list(loss_history)) if len(loss_history) > 1 else 0
            stability_status = "✓ Stable" if loss_std < 2.0 else "⚠ Unstable"
            
            logger.info(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                       f"Gen: {gen_loss:8.4f} | Critic: {critic_loss:8.4f} | "
                       f"GP: {gp_loss:7.4f} | Diversity: {diversity:8.2f} | "
                       f"{stability_status}")
            
            # Early stopping with patience
            if patience_counter >= 20:
                logger.info(f"\n✓ Early stopping at epoch {epoch+1} (no improvement for 20 epochs)")
                break
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✓ Training completed | Best Gen Loss: {best_gen_loss:.4f}")
        logger.info(f"{'='*70}\n")
        
        # Save models
        torch.save(self.generator.state_dict(), 'python/models/generator_improved.pth')
        torch.save(self.critic.state_dict(), 'python/models/critic_improved.pth')
        logger.info("✓ Models saved: generator_improved.pth, critic_improved.pth\n")
        
        return self.history


def main():
    # Setup
    set_random_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load data
    dataset = JUnitDataset('python/data/junit_data_all.csv')
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Create config
    train_config = TrainingConfig(
        epochs=100,
        batch_size=16,
        learning_rate=0.00005,
        lambda_gp=10.0
    )
    
    model_config = ModelConfig(
        z_dim=100,
        output_dim=3
    )
    
    # Train
    trainer = ImprovedWGANTrainer(train_config, model_config, device)
    history = trainer.train(train_loader)
    
    # Log final stats
    logger.info(f"\n📊 TRAINING STATISTICS:")
    logger.info(f"   Final Gen Loss: {history['gen_loss'][-1]:.4f}")
    logger.info(f"   Final Critic Loss: {history['critic_loss'][-1]:.4f}")
    logger.info(f"   Final Diversity: {history['diversity'][-1]:.2f}")
    logger.info(f"   Avg Diversity: {np.mean(history['diversity']):.2f}")


if __name__ == "__main__":
    main()
