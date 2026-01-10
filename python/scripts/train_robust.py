#!/usr/bin/env python3
"""
Train WGAN-GP model on JUnit test data with comprehensive logging and visualization
Uses data from 193 JUnit tests (66 original + 46 extended + 45 MathOps + 36 RobustTests)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import Generator, Critic
from utils import compute_gradient_penalty
from junit_datasets import JUnitDataset

def setup_device():
    """Setup computation device (GPU if available, else CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    return device

def load_data(data_path: str, device: torch.device):
    """Load and prepare JUnit test data"""
    print(f"\n📂 Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load CSV data
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} test samples")
    
    # Create dataset
    dataset = JUnitDataset(data_path)
    print(f"  Dataset size: {len(dataset)} samples")
    print(f"  Feature shape: {dataset.data.shape}")
    print(f"  Data range: [{dataset.data.min():.4f}, {dataset.data.max():.4f}]")
    
    return dataset

def create_dataloaders(dataset, batch_size: int = 32):
    """Create train and test dataloaders"""
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    from torch.utils.data import random_split
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n🔄 DataLoaders created:")
    print(f"  Train samples: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"  Test samples: {len(test_dataset)} ({len(test_loader)} batches)")
    
    return train_loader, test_loader

def create_models(z_dim: int = 100, output_dim: int = 3, device=None):
    """Create Generator and Critic models"""
    print(f"\n🧠 Creating models...")
    
    generator = Generator(z_dim=z_dim, output_dim=output_dim)
    critic = Critic(input_dim=output_dim)
    
    generator = generator.to(device)
    critic = critic.to(device)
    
    # Count parameters
    gen_params = sum(p.numel() for p in generator.parameters())
    crit_params = sum(p.numel() for p in critic.parameters())
    
    print(f"  Generator: {gen_params:,} parameters")
    print(f"  Critic: {crit_params:,} parameters")
    print(f"  Total: {gen_params + crit_params:,} parameters")
    
    return generator, critic

def train_epoch(generator, critic, train_loader, opt_g, opt_c, device, 
                lambda_gp=10, n_critic=5):
    """Train one epoch"""
    generator.train()
    critic.train()
    
    total_d_loss = 0.0
    total_g_loss = 0.0
    batch_count = 0
    
    for batch_idx, real_data in enumerate(train_loader):
        real_data = real_data.to(device)
        batch_size = real_data.size(0)
        
        # ==================== Train Critic ====================
        for _ in range(n_critic):
            opt_c.zero_grad()
            
            # Real data
            real_output = critic(real_data)
            
            # Fake data
            z = torch.randn(batch_size, 100, device=device)  # z_dim = 100
            fake_data = generator(z)
            fake_output = critic(fake_data.detach())
            
            # Gradient penalty
            gp = compute_gradient_penalty(critic, real_data, fake_data, device, batch_size, lambda_gp)
            
            # Critic loss (Wasserstein distance + gradient penalty)
            d_loss = -torch.mean(real_output) + torch.mean(fake_output) + gp
            d_loss.backward()
            opt_c.step()
        
        # ==================== Train Generator ====================
        opt_g.zero_grad()
        
        z = torch.randn(batch_size, 100, device=device)
        fake_data = generator(z)
        fake_output = critic(fake_data)
        
        g_loss = -torch.mean(fake_output)
        g_loss.backward()
        opt_g.step()
        
        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()
        batch_count += 1
    
    avg_d_loss = total_d_loss / batch_count
    avg_g_loss = total_g_loss / batch_count
    
    return avg_d_loss, avg_g_loss

def evaluate(generator, critic, test_loader, device):
    """Evaluate on test set"""
    generator.eval()
    critic.eval()
    
    total_d_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for real_data in test_loader:
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            real_output = critic(real_data)
            
            z = torch.randn(batch_size, 100, device=device)
            fake_data = generator(z)
            fake_output = critic(fake_data)
            
            d_loss = -torch.mean(real_output) + torch.mean(fake_output)
            total_d_loss += d_loss.item()
            batch_count += 1
    
    return total_d_loss / batch_count

def train_wgan(epochs: int = 50, batch_size: int = 32, 
               learning_rate: float = 0.0002, lambda_gp: float = 10):
    """Main training loop"""
    print("\n" + "="*70)
    print("🚀 WGAN-GP Training with 193 JUnit Tests")
    print("="*70)
    
    # Setup
    device = setup_device()
    
    # Data
    data_path = "data/junit_data_cleaned.csv"
    dataset = load_data(data_path, device)
    train_loader, test_loader = create_dataloaders(dataset, batch_size)
    
    # Models
    generator, critic = create_models(device=device)
    
    # Optimizers
    opt_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_c = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    print(f"\n⚙️  Hyperparameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Lambda (gradient penalty): {lambda_gp}")
    print(f"  Critic iterations per epoch: 5")
    
    # Training history
    history = {
        'epoch': [],
        'train_d_loss': [],
        'train_g_loss': [],
        'test_d_loss': [],
        'timestamp': []
    }
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\n{'Epoch':<8} {'D Loss':<12} {'G Loss':<12} {'Val D Loss':<14} {'Time':<10}")
    print("-" * 70)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_start = datetime.now()
        
        # Train
        d_loss, g_loss = train_epoch(generator, critic, train_loader, 
                                     opt_g, opt_c, device, lambda_gp, n_critic=5)
        
        # Evaluate
        val_d_loss = evaluate(generator, critic, test_loader, device)
        
        # Record
        history['epoch'].append(epoch)
        history['train_d_loss'].append(d_loss)
        history['train_g_loss'].append(g_loss)
        history['test_d_loss'].append(val_d_loss)
        history['timestamp'].append(datetime.now().isoformat())
        
        # Print
        elapsed = (datetime.now() - epoch_start).total_seconds()
        print(f"{epoch:<8} {d_loss:<12.4f} {g_loss:<12.4f} {val_d_loss:<14.4f} {elapsed:<10.2f}s")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            checkpoint_path = f"{models_dir}/generator_epoch{epoch}.pth"
            torch.save(generator.state_dict(), checkpoint_path)
    
    print("-" * 70)
    print(f"✅ Training completed in {(datetime.now() - epoch_start).total_seconds():.2f}s")
    
    # Save final models
    torch.save(generator.state_dict(), f"{models_dir}/generator_final.pth")
    torch.save(critic.state_dict(), f"{models_dir}/critic_final.pth")
    
    # Save history
    history_df = pd.DataFrame(history)
    os.makedirs("outputs", exist_ok=True)
    history_df.to_csv("outputs/training_history.csv", index=False)
    print(f"📊 Training history saved to outputs/training_history.csv")
    
    # Plot results
    plot_training_history(history)
    
    return generator, critic, history

def plot_training_history(history):
    """Plot training curves"""
    epochs = history['epoch']
    d_losses = history['train_d_loss']
    g_losses = history['train_g_loss']
    val_losses = history['test_d_loss']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Critic loss
    axes[0].plot(epochs, d_losses, 'b-', label='Train D Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r--', label='Val D Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Critic Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Generator loss
    axes[1].plot(epochs, g_losses, 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Generator Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig('outputs/training_curves.png', dpi=150, bbox_inches='tight')
    print(f"📈 Training curves saved to outputs/training_curves.png")
    plt.close()

def generate_synthetic_tests(generator, num_samples=500, output_path="outputs/synthetic_tests_robust.csv"):
    """Generate synthetic test data"""
    print(f"\n🔄 Generating {num_samples} synthetic tests...")
    
    device = next(generator.parameters()).device
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(num_samples, 100, device=device)
        synthetic_data = generator(z)
        synthetic_data = synthetic_data.cpu().numpy()
    
    # Denormalize if necessary
    synthetic_data = np.clip(synthetic_data, -1, 1)
    
    # Save
    columns = [f'feature_{i}' for i in range(synthetic_data.shape[1])]
    df = pd.DataFrame(synthetic_data, columns=columns)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Saved {num_samples} synthetic tests to {output_path}")
    print(f"   Shape: {synthetic_data.shape}")
    print(f"   Range: [{synthetic_data.min():.4f}, {synthetic_data.max():.4f}]")
    
    return synthetic_data

if __name__ == "__main__":
    print("\n🏁 Starting WGAN training with 193 JUnit tests...\n")
    
    try:
        # Train model
        generator, critic, history = train_wgan(
            epochs=50,
            batch_size=32,
            learning_rate=0.0002,
            lambda_gp=10
        )
        
        # Generate synthetic tests
        synthetic_data = generate_synthetic_tests(generator, num_samples=500)
        
        print("\n" + "="*70)
        print("✅ Training completed successfully!")
        print("="*70)
        print(f"📁 Models saved in: models/")
        print(f"📊 Training history: outputs/training_history.csv")
        print(f"📈 Plots: outputs/training_curves.png")
        print(f"🧪 Synthetic tests: outputs/synthetic_tests_robust.csv")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
