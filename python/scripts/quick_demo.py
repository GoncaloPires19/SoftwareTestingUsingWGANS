"""
Quick Demo: Run a single training configuration to demonstrate the campaign workflow
Use this to test the system before running the full campaign
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
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickDemoTraining:
    """Demonstrate single training run"""
    
    def __init__(self):
        self.output_dir = Path('outputs/demo')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_demo(self):
        """Run a quick demo training"""
        
        print("\n" + "█"*90)
        print("█" + " "*88 + "█")
        print("█" + " "*15 + "CAMPAIGN DEMO: Single Training Configuration" + " "*28 + "█")
        print("█" + " "*88 + "█")
        print("█"*90 + "\n")
        
        # Load data
        logger.info("📂 Loading dataset: junit_only.csv")
        data_path = Path('data/processed/junit_only.csv')
        
        if not data_path.exists():
            logger.error(f"❌ Dataset not found. Run data_aggregator.py first!")
            return False
        
        try:
            df = pd.read_csv(data_path, header=None, names=['methods', 'branches', 'statements'])
            logger.info(f"✅ Loaded {len(df)} samples")
            
            # Normalize
            data = df.values.astype(np.float32)
            min_vals = data.min(axis=0)
            max_vals = data.max(axis=0)
            data_norm = 2 * (data - min_vals) / (max_vals - min_vals + 1e-8) - 1
            data_tensor = torch.from_numpy(data_norm)
            
            # Dataset
            class SimpleDataset(Dataset):
                def __init__(self, data):
                    self.data = data
                def __len__(self):
                    return len(self.data)
                def __getitem__(self, idx):
                    return self.data[idx]
            
            dataset = SimpleDataset(data_tensor)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
            
            # Models
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"🖥️  Device: {device}")
            
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
            
            opt_g = optim.Adam(generator.parameters(), lr=5e-5, betas=(0.5, 0.9))
            opt_d = optim.Adam(critic.parameters(), lr=5e-5, betas=(0.5, 0.9))
            
            # Training
            logger.info("\n🚀 Starting training...\n")
            print("═"*90)
            print(f"{'Epoch':<8} {'D Loss':<12} {'G Loss':<12} {'Status':<50} {'Time':<10}")
            print("═"*90)
            
            start_time = time.time()
            history = {'d_loss': [], 'g_loss': [], 'epochs': []}
            num_epochs = 5  # Short demo
            
            for epoch in range(num_epochs):
                epoch_start = time.time()
                d_losses = []
                g_losses = []
                
                for batch_idx, real_data in enumerate(dataloader):
                    real_data = real_data.to(device)
                    
                    # Train critic (5 iterations)
                    for _ in range(5):
                        opt_d.zero_grad()
                        
                        real_pred = critic(real_data)
                        z = torch.randn(real_data.size(0), 3).to(device)
                        fake_data = generator(z)
                        fake_pred = critic(fake_data.detach())
                        
                        d_loss = -(real_pred.mean() - fake_pred.mean())
                        
                        # Gradient penalty
                        batch_size = real_data.size(0)
                        alpha = torch.rand(batch_size, 1, device=device)
                        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
                        d_interpolates = critic(interpolates)
                        
                        gradients = torch.autograd.grad(
                            outputs=d_interpolates,
                            inputs=interpolates,
                            grad_outputs=torch.ones(batch_size, 1, device=device),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True
                        )[0]
                        
                        gp = ((gradients.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
                        d_loss = d_loss + 10.0 * gp
                        
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
                
                avg_d = np.mean(d_losses)
                avg_g = np.mean(g_losses)
                epoch_time = time.time() - epoch_start
                
                history['d_loss'].append(avg_d)
                history['g_loss'].append(avg_g)
                history['epochs'].append(epoch)
                
                status = "✅ Converging" if epoch > 0 and avg_d < history['d_loss'][epoch-1] else "📈 Training"
                print(f"{epoch:<8} {avg_d:<12.4f} {avg_g:<12.4f} {status:<50} {epoch_time:<10.2f}s")
            
            total_time = time.time() - start_time
            print("═"*90)
            
            # Summary
            logger.info(f"\n✅ Training completed in {total_time:.1f}s")
            logger.info(f"  Final D Loss: {history['d_loss'][-1]:.4f}")
            logger.info(f"  Final G Loss: {history['g_loss'][-1]:.4f}")
            
            # Save results
            results = {
                'config': 'demo_junit',
                'dataset': 'junit_only',
                'num_epochs': num_epochs,
                'final_d_loss': float(history['d_loss'][-1]),
                'final_g_loss': float(history['g_loss'][-1]),
                'avg_d_loss': float(np.mean(history['d_loss'])),
                'avg_g_loss': float(np.mean(history['g_loss'])),
                'training_time': total_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save model
            torch.save(generator.state_dict(), self.output_dir / 'generator_demo.pth')
            torch.save(critic.state_dict(), self.output_dir / 'critic_demo.pth')
            logger.info(f"💾 Models saved to {self.output_dir}")
            
            print("\n" + "="*90)
            print("📊 DEMO RESULTS SUMMARY")
            print("="*90)
            print(f"\n  Configuration: demo_junit")
            print(f"  Dataset: junit_only (196 samples)")
            print(f"  Epochs: {num_epochs}")
            print(f"  Batch Size: 16")
            print(f"")
            print(f"  Training Results:")
            print(f"    • Final D Loss: {results['final_d_loss']:.4f}")
            print(f"    • Final G Loss: {results['final_g_loss']:.4f}")
            print(f"    • Avg D Loss: {results['avg_d_loss']:.4f}")
            print(f"    • Training Time: {results['training_time']:.1f}s")
            print(f"")
            print(f"  Output Files:")
            print(f"    ✅ Generator model: outputs/demo/generator_demo.pth")
            print(f"    ✅ Critic model: outputs/demo/critic_demo.pth")
            print("\n" + "="*90)
            
            print("\n💡 NEXT STEPS:")
            print("\n1. Run full training campaign with 8 configurations:")
            print("   python python/scripts/run_full_campaign.py")
            print("\n2. Or run individual phases:")
            print("   python python/scripts/data_aggregator.py")
            print("   python python/scripts/training_campaign.py")
            print("   python python/scripts/campaign_analyzer.py")
            print("\n3. See results in:")
            print("   outputs/campaign/campaign_results.csv")
            print("   outputs/campaign/analysis_report.json")
            print("   outputs/campaign/campaign_comparison.png")
            print("\n" + "="*90 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            return False

if __name__ == '__main__':
    demo = QuickDemoTraining()
    success = demo.run_demo()
    sys.exit(0 if success else 1)
