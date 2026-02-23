"""
Training Campaign: Execute multiple training runs with different configurations
Tracks, compares and annotates results for optimization research
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple
import time
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for a single training run"""
    name: str
    dataset: str
    num_epochs: int
    batch_size: int
    learning_rate_d: float
    learning_rate_g: float
    lambda_gp: float
    critic_iterations: int
    description: str = ""

class TrainingCampaign:
    """Orchestrate multiple training runs with different configurations"""
    
    def __init__(self):
        self.output_dir = Path('outputs/campaign')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.campaign_metadata = {
            'start_time': datetime.now().isoformat(),
            'configs': [],
            'results': []
        }
    
    def define_training_configs(self) -> List[TrainingConfig]:
        """Define different training configurations to test"""
        configs = [
            # Baseline: Current setup
            TrainingConfig(
                name="baseline_junit",
                dataset="junit_only",
                num_epochs=15,
                batch_size=16,
                learning_rate_d=5e-5,
                learning_rate_g=5e-5,
                lambda_gp=10.0,
                critic_iterations=5,
                description="Current baseline with JUnit data only"
            ),
            
            # Increased diversity
            TrainingConfig(
                name="expanded_all",
                dataset="all_combined",
                num_epochs=20,
                batch_size=16,
                learning_rate_d=5e-5,
                learning_rate_g=5e-5,
                lambda_gp=10.0,
                critic_iterations=5,
                description="All datasets combined (junit+defects4j+juliet+owasp)"
            ),
            
            # Higher learning rates for faster convergence
            TrainingConfig(
                name="high_lr",
                dataset="junit_only",
                num_epochs=20,
                batch_size=16,
                learning_rate_d=1e-4,
                learning_rate_g=1e-4,
                lambda_gp=10.0,
                critic_iterations=5,
                description="Higher learning rates for faster training"
            ),
            
            # Lower learning rates for stability
            TrainingConfig(
                name="low_lr",
                dataset="junit_only",
                num_epochs=25,
                batch_size=16,
                learning_rate_d=2e-5,
                learning_rate_g=2e-5,
                lambda_gp=10.0,
                critic_iterations=5,
                description="Lower learning rates for more stable training"
            ),
            
            # Increased gradient penalty
            TrainingConfig(
                name="high_gp",
                dataset="junit_only",
                num_epochs=20,
                batch_size=16,
                learning_rate_d=5e-5,
                learning_rate_g=5e-5,
                lambda_gp=50.0,
                critic_iterations=5,
                description="High gradient penalty (50x) for Lipschitz constraint"
            ),
            
            # More critic iterations
            TrainingConfig(
                name="critic_heavy",
                dataset="junit_only",
                num_epochs=15,
                batch_size=16,
                learning_rate_d=5e-5,
                learning_rate_g=5e-5,
                lambda_gp=10.0,
                critic_iterations=10,
                description="10 critic iterations per generator update"
            ),
            
            # Larger batch size
            TrainingConfig(
                name="large_batch",
                dataset="junit_only",
                num_epochs=15,
                batch_size=32,
                learning_rate_d=5e-5,
                learning_rate_g=5e-5,
                lambda_gp=10.0,
                critic_iterations=5,
                description="Doubled batch size (32) for better gradient estimates"
            ),
            
            # Combined improvements
            TrainingConfig(
                name="optimized",
                dataset="all_combined",
                num_epochs=25,
                batch_size=32,
                learning_rate_d=3e-5,
                learning_rate_g=3e-5,
                lambda_gp=20.0,
                critic_iterations=6,
                description="Optimized config: combined data, adjusted rates, balanced critic"
            ),
        ]
        
        return configs
    
    def get_or_create_dummy_data(self, dataset_name: str, num_samples: int = 200) -> torch.Tensor:
        """Load dataset or create dummy data for testing"""
        try:
            # Try to load actual dataset
            csv_path = Path(f'data/processed/{dataset_name}.csv')
            if csv_path.exists():
                df = pd.read_csv(csv_path, header=None, names=['methods', 'branches', 'statements'])
                data = df[['methods', 'branches', 'statements']].values.astype(np.float32)
                
                # Normalize to [-1, 1]
                min_vals = data.min(axis=0)
                max_vals = data.max(axis=0)
                data_norm = 2 * (data - min_vals) / (max_vals - min_vals + 1e-8) - 1
                
                logger.info(f"✅ Loaded {dataset_name}: {len(data)} samples")
                return torch.from_numpy(data_norm)
        except Exception as e:
            logger.warning(f"⚠️  Could not load {dataset_name}: {e}")
        
        # Fall back to synthetic data
        logger.info(f"📊 Generating dummy data for {dataset_name}")
        data = np.random.randn(num_samples, 3).astype(np.float32)
        return torch.from_numpy(data)
    
    def train_single_config(self, config: TrainingConfig, device: str = 'cpu') -> Dict:
        """Train a single configuration"""
        logger.info(f"\n{'='*90}")
        logger.info(f"🚀 TRAINING: {config.name}")
        logger.info(f"{'='*90}")
        logger.info(f"  Dataset: {config.dataset}")
        logger.info(f"  Epochs: {config.num_epochs}")
        logger.info(f"  LR (D/G): {config.learning_rate_d}/ {config.learning_rate_g}")
        logger.info(f"  Gradient Penalty: {config.lambda_gp}")
        logger.info(f"  Critic Iterations: {config.critic_iterations}")
        
        start_time = time.time()
        
        try:
            # Load data
            data_tensor = self.get_or_create_dummy_data(config.dataset)
            
            # Create simple dataset
            class SimpleDataset(Dataset):
                def __init__(self, data):
                    self.data = data
                def __len__(self):
                    return len(self.data)
                def __getitem__(self, idx):
                    return self.data[idx]
            
            dataset = SimpleDataset(data_tensor)
            dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
            
            # Simple models for testing
            generator = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 3),
                nn.Tanh()
            ).to(device)
            
            critic = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ).to(device)
            
            opt_g = optim.Adam(generator.parameters(), lr=config.learning_rate_g, betas=(0.5, 0.9))
            opt_d = optim.Adam(critic.parameters(), lr=config.learning_rate_d, betas=(0.5, 0.9))
            
            # Training loop
            history = {
                'train_d_loss': [],
                'train_g_loss': [],
                'epoch': []
            }
            
            for epoch in range(config.num_epochs):
                d_losses = []
                g_losses = []
                
                for batch_idx, real_data in enumerate(dataloader):
                    real_data = real_data.to(device)
                    
                    # Train critic
                    for _ in range(config.critic_iterations):
                        opt_d.zero_grad()
                        
                        # Real data
                        real_pred = critic(real_data)
                        
                        # Fake data
                        z = torch.randn(real_data.size(0), 3).to(device)
                        fake_data = generator(z)
                        fake_pred = critic(fake_data.detach())
                        
                        # Wasserstein loss
                        d_loss = -(real_pred.mean() - fake_pred.mean())
                        
                        # Gradient penalty
                        gp = self._gradient_penalty(critic, real_data, fake_data, device)
                        d_loss = d_loss + config.lambda_gp * gp
                        
                        d_loss.backward()
                        opt_d.step()
                        d_losses.append(d_loss.item())
                    
                    # Train generator
                    opt_g.zero_grad()
                    z = torch.randn(real_data.size(0), 3).to(device)
                    fake_data = generator(z)
                    fake_pred = critic(fake_data)
                    g_loss = -fake_pred.mean()
                    g_loss.backward()
                    opt_g.step()
                    g_losses.append(g_loss.item())
                
                avg_d_loss = np.mean(d_losses) if d_losses else 0
                avg_g_loss = np.mean(g_losses) if g_losses else 0
                
                history['train_d_loss'].append(avg_d_loss)
                history['train_g_loss'].append(avg_g_loss)
                history['epoch'].append(epoch)
                
                if epoch % max(1, config.num_epochs // 5) == 0:
                    logger.info(f"  Epoch {epoch:3d}/{config.num_epochs} | D Loss: {avg_d_loss:7.4f} | G Loss: {avg_g_loss:7.4f}")
            
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            result = {
                'config_name': config.name,
                'dataset': config.dataset,
                'num_epochs': config.num_epochs,
                'batch_size': config.batch_size,
                'learning_rate_d': config.learning_rate_d,
                'learning_rate_g': config.learning_rate_g,
                'lambda_gp': config.lambda_gp,
                'critic_iterations': config.critic_iterations,
                'final_d_loss': float(history['train_d_loss'][-1]),
                'final_g_loss': float(history['train_g_loss'][-1]),
                'avg_d_loss': float(np.mean(history['train_d_loss'])),
                'avg_g_loss': float(np.mean(history['train_g_loss'])),
                'min_d_loss': float(np.min(history['train_d_loss'])),
                'max_d_loss': float(np.max(history['train_d_loss'])),
                'training_time_seconds': elapsed_time,
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"\n✅ Training completed in {elapsed_time:.1f}s")
            logger.info(f"  Final D Loss: {result['final_d_loss']:.4f}")
            logger.info(f"  Final G Loss: {result['final_g_loss']:.4f}")
            
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            return {
                'config_name': config.name,
                'dataset': config.dataset,
                'status': 'failed',
                'error': str(e),
                'training_time_seconds': elapsed_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def _gradient_penalty(self, critic, real_data, fake_data, device):
        """Calculate gradient penalty for WGAN-GP"""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=device)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        
        d_interpolates = critic(interpolates)
        fake = torch.ones(batch_size, 1, device=device, requires_grad=True)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def run_campaign(self):
        """Execute full training campaign"""
        logger.info("\n" + "█"*90)
        logger.info("█" + " "*88 + "█")
        logger.info("█" + " "*15 + "WGAN TRAINING CAMPAIGN: Multiple Configurations" + " "*25 + "█")
        logger.info("█" + " "*88 + "█")
        logger.info("█"*90)
        
        configs = self.define_training_configs()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"\n🖥️  Using device: {device}")
        
        # Run all configurations
        for i, config in enumerate(configs, 1):
            logger.info(f"\n[{i}/{len(configs)}] Starting training...")
            result = self.train_single_config(config, device)
            self.results.append(result)
            self.campaign_metadata['configs'].append(asdict(config))
        
        # Save and analyze results
        self._save_results()
        self._analyze_results()
    
    def _save_results(self):
        """Save campaign results"""
        results_df = pd.DataFrame(self.results)
        results_path = self.output_dir / 'campaign_results.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"\n✅ Results saved: {results_path}")
        
        # Save JSON metadata
        metadata_path = self.output_dir / 'campaign_metadata.json'
        self.campaign_metadata['results'] = self.results
        self.campaign_metadata['end_time'] = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(self.campaign_metadata, f, indent=2)
        logger.info(f"✅ Metadata saved: {metadata_path}")
    
    def _analyze_results(self):
        """Analyze and compare results"""
        logger.info("\n" + "="*90)
        logger.info("📊 CAMPAIGN ANALYSIS & RECOMMENDATIONS")
        logger.info("="*90 + "\n")
        
        results_df = pd.DataFrame(self.results)
        
        # Filter only completed trainings
        completed = results_df[results_df['status'] == 'completed']
        
        if len(completed) == 0:
            logger.warning("⚠️  No completed trainings to analyze")
            return
        
        # Best by different metrics
        logger.info("🏆 BEST PERFORMERS:\n")
        
        best_d_loss = completed.loc[completed['final_d_loss'].idxmin()]
        logger.info(f"  Lowest D Loss: {best_d_loss['config_name']}")
        logger.info(f"    → Final D Loss: {best_d_loss['final_d_loss']:.4f}")
        
        best_g_loss = completed.loc[completed['final_g_loss'].idxmin()]
        logger.info(f"\n  Lowest G Loss: {best_g_loss['config_name']}")
        logger.info(f"    → Final G Loss: {best_g_loss['final_g_loss']:.4f}")
        
        fastest = completed.loc[completed['training_time_seconds'].idxmin()]
        logger.info(f"\n  Fastest Training: {fastest['config_name']}")
        logger.info(f"    → Time: {fastest['training_time_seconds']:.1f}s")
        
        # Dataset comparison
        logger.info(f"\n" + "─"*90)
        logger.info(f"📈 DATASET COMPARISON")
        logger.info(f"─"*90 + "\n")
        
        for dataset in completed['dataset'].unique():
            dataset_runs = completed[completed['dataset'] == dataset]
            logger.info(f"  {dataset}:")
            logger.info(f"    Runs: {len(dataset_runs)}")
            logger.info(f"    Avg D Loss: {dataset_runs['final_d_loss'].mean():.4f}")
            logger.info(f"    Avg G Loss: {dataset_runs['final_g_loss'].mean():.4f}")
            logger.info(f"    Avg Time: {dataset_runs['training_time_seconds'].mean():.1f}s")
        
        logger.info(f"\n" + "="*90)

if __name__ == '__main__':
    campaign = TrainingCampaign()
    campaign.run_campaign()
