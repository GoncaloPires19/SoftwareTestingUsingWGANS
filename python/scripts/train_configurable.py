"""
WGAN-GP training with configurable datasets, early stopping, and advanced metrics.
Supports: Juliet, OWASP, Defects4J, EvoSuite, and combinations.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import logging

from config import WGANConfig, DatasetMode, MetricType, get_config
from data_loader import UnifiedDataLoader
from metrics import MetricsTracker, compute_all_metrics
from model import Generator, Critic
from utils import gradient_penalty


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStoppingTracker:
    """Early stopping logic with patience"""
    
    def __init__(self, patience: int = 10, metric_name: str = "diversity", higher_is_better: bool = True):
        self.patience = patience
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.best_value = -np.inf if higher_is_better else np.inf
        self.patience_counter = 0
        self.should_stop = False
    
    def check(self, current_value: float) -> bool:
        """Check if should stop training"""
        
        improved = (
            current_value > self.best_value if self.higher_is_better
            else current_value < self.best_value
        )
        
        if improved:
            self.best_value = current_value
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.should_stop = True
                return True
        
        return False


class WGANTrainer:
    """WGAN-GP trainer with multi-dataset support"""
    
    def __init__(self, config: WGANConfig):
        self.config = config
        self.device = torch.device(config.training.device)
        
        # Initialize models
        self.generator = Generator(
            z_dim=config.model.z_dim,
            output_dim=config.model.output_dim,
            hidden_dims=config.model.generator_hidden,
            batch_norm=config.model.batch_norm_g,
        ).to(self.device)
        
        self.critic = Critic(
            input_dim=config.model.output_dim,
            hidden_dims=config.model.critic_hidden,
            batch_norm=config.model.batch_norm_c,
        ).to(self.device)
        
        # Optimizers
        self.opt_g = optim.Adam(
            self.generator.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2),
        )
        self.opt_c = optim.Adam(
            self.critic.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2),
        )
        
        # Metrics
        self.metrics_tracker = MetricsTracker(
            metrics_list=[m.value for m in config.evaluation.metrics]
        )
        
        # Early stopping
        self.early_stopper = None
        if config.training.early_stopping_enabled:
            self.early_stopper = EarlyStoppingTracker(
                patience=config.training.early_stopping_patience,
                metric_name=config.training.early_stopping_metric.value,
                higher_is_better=True,
            )
        
        logger.info(f"\n✅ Models initialized:")
        logger.info(f"   Generator: {sum(p.numel() for p in self.generator.parameters()):,} params")
        logger.info(f"   Critic: {sum(p.numel() for p in self.critic.parameters()):,} params")
    
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train one epoch"""
        
        self.generator.train()
        self.critic.train()
        
        d_losses = []
        g_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            real_data = batch["features"].to(self.device)
            batch_size = real_data.size(0)
            
            # ===== Train Critic =====
            for _ in range(self.config.training.n_critic):
                self.opt_c.zero_grad()
                
                # Real samples
                real_score = self.critic(real_data)
                
                # Fake samples
                z = torch.randn(batch_size, self.config.model.z_dim, device=self.device)
                fake_data = self.generator(z)
                fake_score = self.critic(fake_data.detach())
                
                # Gradient penalty
                gp = gradient_penalty(
                    self.critic,
                    real_data,
                    fake_data.detach(),
                    device=self.device,
                    lambda_gp=self.config.training.lambda_gp,
                )
                
                # Critic loss: minimize (fake - real + gp)
                d_loss = -real_score.mean() + fake_score.mean() + gp
                d_loss.backward()
                self.opt_c.step()
            
            d_losses.append(d_loss.item())
            
            # ===== Train Generator =====
            self.opt_g.zero_grad()
            
            z = torch.randn(batch_size, self.config.model.z_dim, device=self.device)
            fake_data = self.generator(z)
            fake_score = self.critic(fake_data)
            
            # Generator loss: maximize critic score
            g_loss = -fake_score.mean()
            g_loss.backward()
            self.opt_g.step()
            
            g_losses.append(g_loss.item())
        
        return np.mean(d_losses), np.mean(g_losses)
    
    def evaluate(self, val_loader: DataLoader, real_data: np.ndarray) -> Dict:
        """Evaluate on validation set"""
        
        self.generator.eval()
        self.critic.eval()
        
        with torch.no_grad():
            val_losses = []
            
            for batch in val_loader:
                real = batch["features"].to(self.device)
                score = self.critic(real)
                val_losses.append(score.mean().item())
            
            val_loss = np.mean(val_losses)
            
            # Generate samples for metrics
            z = torch.randn(len(real_data), self.config.model.z_dim, device=self.device)
            fake_data = self.generator(z).cpu().numpy()
            
            # Compute metrics
            metrics = compute_all_metrics(
                real_data,
                fake_data,
                metrics_to_compute=[m.value for m in self.config.evaluation.metrics]
            )
        
        return val_loss, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, real_data: np.ndarray):
        """
        Full training loop
        """
        
        print("\n" + "="*70)
        print("🚀 WGAN-GP TRAINING")
        print("="*70)
        print(f"Experiment: {self.config.experiment_name}")
        print(f"Epochs: {self.config.training.epochs}")
        print(f"Dataset mode: {self.config.data.dataset_mode.value}")
        print(f"Early stopping: {self.config.training.early_stopping_enabled}")
        print("="*70 + "\n")
        
        history = {
            "epoch": [],
            "train_d_loss": [],
            "train_g_loss": [],
            "val_loss": [],
        }
        
        # Add metric columns
        for metric_name in self.config.evaluation.metrics:
            history[metric_name.value] = []
        
        start_time = datetime.now()
        
        for epoch in range(1, self.config.training.epochs + 1):
            epoch_start = datetime.now()
            
            # Train
            d_loss, g_loss = self.train_epoch(train_loader)
            
            # Evaluate
            val_loss, metrics = self.evaluate(val_loader, real_data)
            
            # Track
            history["epoch"].append(epoch)
            history["train_d_loss"].append(d_loss)
            history["train_g_loss"].append(g_loss)
            history["val_loss"].append(val_loss)
            
            for metric_name, value in metrics.items():
                if metric_name in history:
                    history[metric_name].append(value)
            
            self.metrics_tracker.update(metrics)
            
            # Log
            elapsed = (datetime.now() - epoch_start).total_seconds()
            
            if epoch % self.config.output.log_interval == 0:
                metric_str = " | ".join([
                    f"{m}: {metrics.get(m, 0):.4f}"
                    for m in [mt.value for mt in self.config.evaluation.metrics][:2]
                ])
                print(f"[{epoch:3d}] D:{d_loss:8.4f} G:{g_loss:8.4f} Val:{val_loss:8.4f} | {metric_str} ({elapsed:.2f}s)")
            
            # Early stopping check
            if self.config.training.early_stopping_enabled:
                metric_value = metrics.get(self.config.training.early_stopping_metric.value, 0)
                
                if self.early_stopper.check(metric_value):
                    logger.info(f"\n⏹️  Early stopping at epoch {epoch}")
                    logger.info(f"   Best {self.config.training.early_stopping_metric.value}: {self.early_stopper.best_value:.4f}")
                    break
            
            # Checkpoint
            if self.config.training.checkpoint_interval and epoch % self.config.training.checkpoint_interval == 0:
                self._save_checkpoint(epoch)
        
        # Save final models
        self._save_models()
        
        # Save history
        history_df = pd.DataFrame(history)
        history_path = self.config.output.output_dir / "training_history.csv"
        history_df.to_csv(history_path, index=False)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*70)
        print(f"✅ Training completed in {total_time:.1f}s ({total_time/60:.1f}min)")
        print("="*70)
        print(f"📁 Models: {self.config.output.models_dir}/")
        print(f"📊 History: {history_path}")
        
        self.metrics_tracker.print_summary()
        
        return history_df
    
    def _save_checkpoint(self, epoch: int):
        """Save checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "generator": self.generator.state_dict(),
            "critic": self.critic.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_c": self.opt_c.state_dict(),
        }
        
        path = self.config.output.models_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, path)
    
    def _save_models(self):
        """Save final models"""
        torch.save(
            self.generator.state_dict(),
            self.config.output.models_dir / "generator_final.pth"
        )
        torch.save(
            self.critic.state_dict(),
            self.config.output.models_dir / "critic_final.pth"
        )


def main():
    parser = argparse.ArgumentParser(description="Train WGAN with configurable datasets")
    
    # Dataset options
    parser.add_argument(
        "--dataset",
        choices=["juliet", "owasp", "defects4j", "evosuite", "combined"],
        default="combined",
        help="Dataset mode"
    )
    
    # Training options
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--lambda-gp", type=float, default=10, help="Gradient penalty lambda")
    
    # Early stopping
    parser.add_argument("--early-stopping", action="store_true", default=True, help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # Experiment
    parser.add_argument("--name", default="wgan_experiment", help="Experiment name")
    parser.add_argument("--preset", choices=["quick", "juliet_only", "combined", "evosuite_synthetic"],
                       help="Use preset configuration")
    
    args = parser.parse_args()
    
    # Load config
    if args.preset:
        config = WGANConfig.from_preset(args.preset)
    else:
        config = get_config(
            dataset_mode=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            experiment_name=args.name,
            early_stopping=args.early_stopping,
        )
        
        # Override with command line args
        config.training.epochs = args.epochs
        config.training.batch_size = args.batch_size
        config.training.learning_rate = args.lr
        config.training.lambda_gp = args.lambda_gp
        config.training.early_stopping_patience = args.patience
    
    # Create trainer
    trainer = WGANTrainer(config)
    
    # Load data
    logger.info("📂 Loading data...")
    data_loader = UnifiedDataLoader()
    
    train_data, val_data, test_data, metadata = data_loader.load_combined(
        include_juliet=config.data.dataset_mode in [DatasetMode.JULIET, DatasetMode.COMBINED],
        include_owasp=config.data.dataset_mode in [DatasetMode.OWASP, DatasetMode.COMBINED],
        include_defects4j=config.data.dataset_mode in [DatasetMode.DEFECTS4J, DatasetMode.COMBINED],
        include_evosuite=config.data.dataset_mode in [DatasetMode.EVOSUITE, DatasetMode.COMBINED],
        include_current_junit=True,  # Always include current data
    )
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        train_data, val_data, test_data,
        batch_size=config.training.batch_size,
    )
    
    # Train
    history = trainer.train(train_loader, val_loader, train_data)
    
    logger.info("\n✅ Training complete!")


if __name__ == "__main__":
    main()
