"""
Cascade training: use combined real + synthetic data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

from config import WGANConfig, DatasetMode
from model import Generator, Critic
from utils import compute_gradient_penalty
from metrics import MetricsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CascadeTrainer:
    """Train on combined real + synthetic data"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cpu")
        
        # Initialize models
        self.generator = Generator(z_dim=100, output_dim=3).to(self.device)
        self.critic = Critic(input_dim=3).to(self.device)
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2)
        )
        self.c_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2)
        )
        
        # Metrics
        self.metrics_tracker = MetricsTracker()
    
    def load_combined_data(self, combined_csv_path: Path, batch_size: int = 32):
        """Load combined real + synthetic data"""
        
        logger.info(f"Loading combined data from {combined_csv_path}")
        
        df = pd.read_csv(combined_csv_path)
        
        # Use only numeric columns
        data = df[['methods', 'branches', 'statements']].values.astype(np.float32)
        
        # Normalize to [-1, 1]
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data_norm = 2 * (data - data_min) / (data_max - data_min + 1e-8) - 1
        
        logger.info(f"Data shape: {data_norm.shape}")
        logger.info(f"Data range: [{data_norm.min():.4f}, {data_norm.max():.4f}]")
        
        # Create dataset
        tensor_data = torch.from_numpy(data_norm).float()
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader, data_norm
    
    def train(self, train_loader, epochs: int = 100):
        """Train WGAN-GP"""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"CASCADE TRAINING - {epochs} EPOCHS")
        logger.info(f"{'='*70}\n")
        
        history = []
        
        for epoch in range(1, epochs + 1):
            epoch_metrics = {
                'epoch': epoch,
                'train_d_loss': 0.0,
                'train_g_loss': 0.0,
            }
            
            # Train discriminator
            for batch_idx, (batch,) in enumerate(train_loader):
                batch = batch.to(self.device)
                batch_size = batch.size(0)
                
                # Train critic for n_critic iterations
                for _ in range(self.config.training.n_critic):
                    self.c_optimizer.zero_grad()
                    
                    # Real data
                    real_output = self.critic(batch)
                    
                    # Fake data
                    noise = torch.randn(batch_size, 100, device=self.device)
                    fake_data = self.generator(noise)
                    fake_output = self.critic(fake_data.detach())
                    
                    # Wasserstein loss
                    wasserstein_loss = -(real_output.mean() - fake_output.mean())
                    
                    # Gradient penalty
                    gp = compute_gradient_penalty(
                        self.critic, batch, fake_data.detach(), self.device, batch_size
                    )
                    
                    # Total loss
                    c_loss = wasserstein_loss + self.config.training.lambda_gp * gp
                    c_loss.backward()
                    self.c_optimizer.step()
                
                epoch_metrics['train_d_loss'] = c_loss.item()
            
            # Train generator
            self.g_optimizer.zero_grad()
            noise = torch.randn(batch_size, 100, device=self.device)
            fake_data = self.generator(noise)
            fake_output = self.critic(fake_data)
            g_loss = -fake_output.mean()
            g_loss.backward()
            self.g_optimizer.step()
            
            epoch_metrics['train_g_loss'] = g_loss.item()
            history.append(epoch_metrics)
            
            # Log
            if epoch % self.config.output.log_interval == 0:
                logger.info(f"[{epoch:3d}] D: {epoch_metrics['train_d_loss']:8.4f} G: {epoch_metrics['train_g_loss']:8.4f}")
        
        # Save models
        torch.save(self.generator.state_dict(), "python/models/generator_cascade.pth")
        torch.save(self.critic.state_dict(), "python/models/critic_cascade.pth")
        
        logger.info(f"\n✅ Training complete!")
        logger.info(f"   Generator saved: python/models/generator_cascade.pth")
        logger.info(f"   Critic saved: python/models/critic_cascade.pth")
        
        return history


def main():
    parser = argparse.ArgumentParser(description="Cascade training on combined real + synthetic data")
    parser.add_argument("--data", type=Path, default=Path("python/data/combined_cascade_5k.csv"),
                       help="Combined data CSV path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    # Configuration
    config = WGANConfig(
        experiment_name="cascade_training",
        training=__import__('config').TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_enabled=False
        )
    )
    
    # Trainer
    trainer = CascadeTrainer(config)
    
    # Load data
    train_loader, data = trainer.load_combined_data(args.data, args.batch_size)
    
    # Train
    history = trainer.train(train_loader, args.epochs)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Epochs: {len(history)}")
    logger.info(f"Final D Loss: {history[-1]['train_d_loss']:.4f}")
    logger.info(f"Final G Loss: {history[-1]['train_g_loss']:.4f}")


if __name__ == "__main__":
    main()
