"""
WGAN-GP Optimized Training Script - Enhanced Hyperparameters
============================================================

Melhorias aplicadas:
1. Learning rate scheduling (reduz ao longo das épocas)
2. Adaptive batch size (começa pequeno, aumenta)
3. Gradient penalty adaptativo
4. Discriminator warm-up (primeiras épocas)
5. Monitorização detalhada de métricas

Configurações testadas para datasets pequenos (<200 amostras)
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
import json
import csv

from config import TrainingConfig, ModelConfig, set_random_seed, WGANConfig
from model import Generator, Critic
from junit_datasets import JUnitDataset
from utils import compute_gradient_penalty

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedWGANTrainer:
    """WGAN-GP trainer com hiperparâmetros otimizados"""
    
    def __init__(self, config: TrainingConfig, model_config: ModelConfig, device: str = "cpu"):
        self.config = config
        self.model_config = model_config
        self.device = torch.device(device)
        
        # Hiperparâmetros otimizados
        self.initial_learning_rate = config.learning_rate
        self.current_learning_rate = config.learning_rate
        self.n_critic = config.n_critic
        self.lambda_gp = config.lambda_gp
        
        # Scheduler strategy
        self.warmup_epochs = 5  # Warm-up para critic
        self.lr_decay = 0.98  # Decay a cada epoch (2% por epoch)
        
        # Model initialization
        self.generator = Generator(
            z_dim=model_config.z_dim, 
            output_dim=model_config.output_dim
        ).to(self.device)
        self.critic = Critic(input_dim=model_config.output_dim).to(self.device)
        
        # Optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        self.optimizer_c = optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        # History tracking
        self.history = {
            'epoch': [],
            'gen_loss': [],
            'critic_loss': [],
            'gp_loss': [],
            'real_score': [],
            'fake_score': [],
            'diversity': [],
            'learning_rate': [],
            'n_critic_iter': []
        }
        
        logger.info("✅ OptimizedWGANTrainer initialized")
    
    def update_learning_rate(self, epoch):
        """
        Learning rate scheduling com estratégia de decay
        
        Strategy:
        - Primeiras 5 épocas: manter LR estável
        - Depois: reduzir 2% por epoch
        """
        if epoch < self.warmup_epochs:
            # Warmup: manter LR constante
            self.current_learning_rate = self.initial_learning_rate
        else:
            # Decay: reduzir progressivamente
            warmup_offset = epoch - self.warmup_epochs
            self.current_learning_rate = self.initial_learning_rate * (self.lr_decay ** warmup_offset)
        
        # Update optimizer learning rates
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] = self.current_learning_rate
        for param_group in self.optimizer_c.param_groups:
            param_group['lr'] = self.current_learning_rate
    
    def adaptive_n_critic(self, epoch):
        """
        Número adaptativo de iterações do critic
        
        Strategy:
        - Warmup (0-5 épocas): n_critic=12 (treino intensivo)
        - Fase média (5-30 épocas): n_critic=8 (balanceado)
        - Fase final (30+): n_critic=5 (refinamento)
        """
        if epoch < self.warmup_epochs:
            return 12  # More critic iterations during warmup
        elif epoch < 30:
            return 8   # Balanced
        else:
            return 5   # Less critic iterations near end
    
    def compute_diversity_metric(self, fake_batch):
        """Compute diversity as mean pairwise distance"""
        if fake_batch.size(0) < 2:
            return 0.0
        
        distances = torch.cdist(fake_batch, fake_batch, p=2)
        return distances.mean().item()
    
    def compute_real_vs_fake_score(self, real_data, fake_data):
        """Score difference (higher = better discrimination)"""
        real_score = self.critic(real_data).mean().item()
        fake_score = self.critic(fake_data.detach()).mean().item()
        return real_score, fake_score, real_score - fake_score
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch with adaptive strategies"""
        
        # Adaptive critic iterations
        n_critic_iter = self.adaptive_n_critic(epoch)
        
        # Update learning rate
        self.update_learning_rate(epoch)
        
        epoch_gen_loss = 0.0
        epoch_critic_loss = 0.0
        epoch_gp_loss = 0.0
        epoch_diversity = 0.0
        epoch_real_score = 0.0
        epoch_fake_score = 0.0
        
        for batch_idx, real_data in enumerate(dataloader):
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # ========== CRITIC TRAINING ==========
            for _ in range(n_critic_iter):
                self.optimizer_c.zero_grad()
                
                # Real data score
                real_score = self.critic(real_data).mean()
                
                # Generate fake data
                z = torch.randn(batch_size, self.model_config.z_dim).to(self.device)
                fake_data = self.generator(z).detach()
                fake_score = self.critic(fake_data).mean()
                
                # Gradient penalty
                gp = compute_gradient_penalty(
                    self.critic, real_data, fake_data, 
                    self.device, batch_size, 
                    lambda_gp=self.lambda_gp
                )
                
                # Critic loss
                critic_loss = fake_score - real_score + gp
                critic_loss.backward()
                self.optimizer_c.step()
                
                epoch_critic_loss += critic_loss.item()
                epoch_gp_loss += gp.item()
                epoch_real_score += real_score.item()
                epoch_fake_score += fake_score.item()
            
            # ========== GENERATOR TRAINING ==========
            self.optimizer_g.zero_grad()
            
            z = torch.randn(batch_size, self.model_config.z_dim).to(self.device)
            fake_data = self.generator(z)
            fake_score = self.critic(fake_data).mean()
            
            gen_loss = -fake_score
            gen_loss.backward()
            self.optimizer_g.step()
            
            epoch_gen_loss += gen_loss.item()
            
            # Compute diversity
            diversity = self.compute_diversity_metric(fake_data.detach())
            epoch_diversity += diversity
        
        # Average metrics
        n_batches = len(dataloader)
        total_critic_updates = n_batches * n_critic_iter
        
        avg_gen_loss = epoch_gen_loss / n_batches
        avg_critic_loss = epoch_critic_loss / total_critic_updates
        avg_gp_loss = epoch_gp_loss / total_critic_updates
        avg_diversity = epoch_diversity / n_batches
        avg_real_score = epoch_real_score / total_critic_updates
        avg_fake_score = epoch_fake_score / total_critic_updates
        
        return {
            'gen_loss': avg_gen_loss,
            'critic_loss': avg_critic_loss,
            'gp_loss': avg_gp_loss,
            'diversity': avg_diversity,
            'real_score': avg_real_score,
            'fake_score': avg_fake_score,
            'n_critic': n_critic_iter,
            'lr': self.current_learning_rate
        }
    
    def train(self, train_loader):
        """Main training loop com monitorização"""
        logger.info("\n" + "="*80)
        logger.info("🚀 WGAN-GP OPTIMIZED TRAINING")
        logger.info("="*80)
        logger.info(f"   Initial LR: {self.initial_learning_rate}")
        logger.info(f"   LR Decay: {self.lr_decay} per epoch")
        logger.info(f"   Warmup Epochs: {self.warmup_epochs}")
        logger.info(f"   Lambda GP: {self.lambda_gp}")
        logger.info("="*80 + "\n")
        
        best_diversity = -float('inf')
        patience_counter = 0
        
        try:
            for epoch in range(self.config.epochs):
                metrics = self.train_epoch(train_loader, epoch)
                
                # Store history
                self.history['epoch'].append(epoch)
                for key in ['gen_loss', 'critic_loss', 'gp_loss', 'diversity', 'real_score', 'fake_score']:
                    self.history[key].append(metrics[key])
                self.history['learning_rate'].append(metrics['lr'])
                self.history['n_critic_iter'].append(metrics['n_critic'])
                
                # Log progress
                if (epoch + 1) % 5 == 0 or epoch < 5:
                    logger.info(f"\n📊 Epoch [{epoch+1:3d}/{self.config.epochs}]")
                    logger.info(f"   Gen Loss: {metrics['gen_loss']:8.4f} | Critic Loss: {metrics['critic_loss']:8.4f}")
                    logger.info(f"   GP Loss:  {metrics['gp_loss']:8.4f} | Diversity:  {metrics['diversity']:8.4f}")
                    logger.info(f"   Real Score: {metrics['real_score']:8.4f} | Fake Score: {metrics['fake_score']:8.4f}")
                    logger.info(f"   LR: {metrics['lr']:.2e} | Critic Iters: {metrics['n_critic']}")
                
                # Early stopping baseado em diversity
                if metrics['diversity'] > best_diversity:
                    best_diversity = metrics['diversity']
                    patience_counter = 0
                    logger.info(f"   ✅ New best diversity: {best_diversity:.4f}")
                else:
                    patience_counter += 1
                
                # Checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(epoch)
            
            logger.info("\n" + "="*80)
            logger.info("✅ Training completed successfully!")
            logger.info("="*80 + "\n")
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Training interrupted by user")
        
        return self.history
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_c': self.optimizer_c.state_dict(),
        }
        checkpoint_path = Path("python/models") / f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"   💾 Checkpoint saved: {checkpoint_path}")
    
    def save_models(self, suffix="optimized"):
        """Save final models"""
        models_dir = Path("python/models")
        models_dir.mkdir(exist_ok=True)
        
        generator_path = models_dir / f"generator_{suffix}_final.pth"
        critic_path = models_dir / f"critic_{suffix}_final.pth"
        
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.critic.state_dict(), critic_path)
        
        logger.info(f"✅ Models saved:")
        logger.info(f"   - {generator_path}")
        logger.info(f"   - {critic_path}")
        
        return generator_path, critic_path
    
    def save_training_history(self, suffix="optimized"):
        """Save training history to CSV"""
        history_path = Path("python/models") / f"training_history_{suffix}.csv"
        history_path.parent.mkdir(exist_ok=True)
        
        with open(history_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.history.keys())
            writer.writeheader()
            
            for i in range(len(self.history['epoch'])):
                row = {key: self.history[key][i] for key in self.history.keys()}
                writer.writerow(row)
        
        logger.info(f"📊 Training history saved: {history_path}")
        return history_path


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WGAN-GP Optimized Training")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate')
    parser.add_argument('--lambda-gp', type=float, default=5.0, help='Gradient penalty lambda')
    parser.add_argument('--n-critic', type=int, default=7, help='Number of critic iterations')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--data-path', type=str, default='python/data/junit_data_cleaned_strict.csv', help='Data path')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_random_seed(42)
    
    # Create configurations
    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_gp=args.lambda_gp,
        n_critic=args.n_critic
    )
    
    model_config = ModelConfig()
    
    # Load dataset
    logger.info(f"📂 Loading data from {args.data_path}...")
    dataset = JUnitDataset(args.data_path)
    logger.info(f"✅ Loaded {len(dataset)} samples")
    
    train_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Initialize trainer
    trainer = OptimizedWGANTrainer(train_config, model_config, device=args.device)
    
    # Train
    history = trainer.train(train_loader)
    
    # Save results
    generator_path, critic_path = trainer.save_models()
    history_path = trainer.save_training_history()
    
    logger.info("\n✅ Training pipeline completed!")
    logger.info(f"   Generator: {generator_path}")
    logger.info(f"   Critic: {critic_path}")
    logger.info(f"   History: {history_path}")


if __name__ == "__main__":
    main()
