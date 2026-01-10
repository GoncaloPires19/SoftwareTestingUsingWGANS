"""
Train WGAN with custom combined dataset for cascading synthesis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from datetime import datetime

from config import WGANConfig, TrainingConfig
from metrics import MetricsTracker
from model import Generator, Critic
from utils import compute_gradient_penalty

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WGANTrainer:
    """Train WGAN on custom dataset"""
    
    def __init__(self, config: WGANConfig):
        self.config = config
        self.device = torch.device(config.training.device)
        
        # Models
        self.generator = Generator(z_dim=100, output_dim=3).to(self.device)
        self.critic = Critic(input_dim=3).to(self.device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), 
                                     lr=config.training.learning_rate,
                                     betas=(config.training.beta1, config.training.beta2))
        self.c_optimizer = optim.Adam(self.critic.parameters(),
                                     lr=config.training.learning_rate,
                                     betas=(config.training.beta1, config.training.beta2))
        
        self.metrics = MetricsTracker()
    
    def train(self, train_loader, epochs=150, patience=25):
        """Train WGAN"""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"CASCADING SYNTHESIS TRAINING")
        logger.info(f"{'='*70}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {len(train_loader.dataset) // len(list(train_loader))}")
        logger.info(f"Total samples: {len(train_loader.dataset)}")
        
        history = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Train critic
            critic_loss = 0
            for _ in range(self.config.training.n_critic):
                batch_loss = 0
                for batch_idx, (real_data,) in enumerate(train_loader):
                    real_data = real_data.to(self.device)
                    batch_size = real_data.size(0)
                    
                    # Critic
                    self.c_optimizer.zero_grad()
                    
                    # Real
                    real_output = self.critic(real_data)
                    
                    # Fake
                    noise = torch.randn(batch_size, 100, device=self.device)
                    fake_data = self.generator(noise)
                    fake_output = self.critic(fake_data.detach())
                    
                    # Gradient penalty
                    gp = compute_gradient_penalty(self.critic, real_data, fake_data, self.device)
                    
                    # Loss
                    c_loss = -torch.mean(real_output) + torch.mean(fake_output) + \
                            self.config.training.lambda_gp * gp
                    
                    c_loss.backward()
                    self.c_optimizer.step()
                    
                    batch_loss += c_loss.item()
                
                critic_loss += batch_loss / (batch_idx + 1)
            
            # Train generator
            generator_loss = 0
            for batch_idx, (real_data,) in enumerate(train_loader):
                batch_size = real_data.size(0)
                
                self.g_optimizer.zero_grad()
                
                noise = torch.randn(batch_size, 100, device=self.device)
                fake_data = self.generator(noise)
                fake_output = self.critic(fake_data)
                
                g_loss = -torch.mean(fake_output)
                
                g_loss.backward()
                self.g_optimizer.step()
                
                generator_loss += g_loss.item()
            
            generator_loss /= (batch_idx + 1)
            critic_loss /= self.config.training.n_critic
            
            # Log
            if epoch % self.config.output.log_interval == 0:
                logger.info(f"[{epoch:3d}] D: {critic_loss:8.4f} G: {generator_loss:8.4f}")
            
            history.append({
                'epoch': epoch,
                'critic_loss': critic_loss,
                'generator_loss': generator_loss,
            })
            
            # Early stopping
            if generator_loss < best_loss:
                best_loss = generator_loss
                patience_counter = 0
                # Save models
                torch.save(self.generator.state_dict(), 
                          Path(self.config.output.models_dir) / 'generator_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"\n⏹️ Early stopping at epoch {epoch}")
                    break
        
        # Save final models
        torch.save(self.generator.state_dict(),
                  Path(self.config.output.models_dir) / 'generator_final.pth')
        torch.save(self.critic.state_dict(),
                  Path(self.config.output.models_dir) / 'critic_final.pth')
        
        logger.info(f"\n✅ Training complete! {len(history)} epochs")
        
        return history


def train_cascading(data_path: Path, epochs: int = 150, name: str = "cascading"):
    """Train on cascading dataset"""
    
    # Load data
    logger.info(f"📂 Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Normalize to [-1, 1]
    data = df[['methods', 'branches', 'statements']].values.astype(np.float32)
    
    # Z-score normalize
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data_norm = (data - mean) / (std + 1e-8)
    
    logger.info(f"✅ Loaded {len(data)} samples")
    logger.info(f"   Range: [{data_norm.min():.2f}, {data_norm.max():.2f}]")
    
    # Create dataset
    tensor_data = torch.FloatTensor(data_norm)
    dataset = TensorDataset(tensor_data)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Config
    config = WGANConfig(experiment_name=name)
    
    # Train
    trainer = WGANTrainer(config)
    history = trainer.train(train_loader, epochs=epochs)
    
    # Save history
    df_history = pd.DataFrame(history)
    output_path = Path(config.output.output_dir) / f"training_history_{name}.csv"
    df_history.to_csv(output_path, index=False)
    logger.info(f"✅ History saved to {output_path}")
    
    return trainer, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WGAN on cascading dataset")
    parser.add_argument("--data", type=Path, required=True, help="Path to training data CSV")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--name", default="cascading", help="Experiment name")
    
    args = parser.parse_args()
    
    trainer, history = train_cascading(
        data_path=args.data,
        epochs=args.epochs,
        name=args.name
    )
