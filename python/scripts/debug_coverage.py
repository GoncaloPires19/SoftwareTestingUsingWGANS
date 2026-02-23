"""
Debug Coverage Calculation Script

This script helps diagnose coverage metric instability by adding detailed logging
during training. Use this to understand why coverage varies so much between epochs.

Usage:
    python python/scripts/debug_coverage.py

What it does:
1. Loads clean dataset and trains for 10 epochs
2. Logs detailed statistics every epoch:
   - Synthetic test value ranges (min/max/mean)
   - Denormalized values before coverage calculation
   - Coverage calculation intermediate steps
   - Comparison with real data distribution

This helps identify if the issue is:
- Denormalization producing out-of-range values
- Coverage formula being incorrect
- Generator producing degenerate samples
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import Generator, Critic
from junit_datasets import JUnitDataset
from torch.utils.data import DataLoader


def load_dataset():
    """Load cleaned dataset"""
    data_path = Path("python/data/junit_data_cleaned_strict.csv")
    print(f"📂 Loading dataset from {data_path}")
    
    df = pd.read_csv(data_path, header=None, names=['methods', 'branches', 'statements'])
    print(f"   Loaded {len(df)} samples")
    print(f"   Real data ranges:")
    print(f"      methods: [{df['methods'].min():.1f}, {df['methods'].max():.1f}], mean={df['methods'].mean():.1f}")
    print(f"      branches: [{df['branches'].min():.1f}, {df['branches'].max():.1f}], mean={df['branches'].mean():.1f}")
    print(f"      statements: [{df['statements'].min():.1f}, {df['statements'].max():.1f}], mean={df['statements'].mean():.1f}")
    
    return df


def load_norm_stats():
    """Load normalization statistics"""
    norm_path = Path("python/data/junit_data_cleaned_norm_stats.json")
    print(f"\n📂 Loading normalization stats from {norm_path}")
    
    with open(norm_path, 'r') as f:
        norm_stats = json.load(f)
    
    print(f"   Min values: {norm_stats['min']}")
    print(f"   Max values: {norm_stats['max']}")
    
    return norm_stats


def denormalize(tensor_norm, norm_stats):
    """
    Denormalize from [-1, 1] to original scale.
    
    Formula: x = ((x_norm + 1) / 2) * (max - min) + min
    """
    data_min = torch.tensor(norm_stats['min'], dtype=torch.float32)
    data_max = torch.tensor(norm_stats['max'], dtype=torch.float32)
    data_range = data_max - data_min
    
    # Denormalize
    tensor_denorm = ((tensor_norm + 1) / 2) * data_range + data_min
    
    return tensor_denorm


def calculate_coverage(synthetic_tests, max_values):
    """
    Calculate coverage metrics.
    
    Args:
        synthetic_tests: numpy array (N, 3) with denormalized values
        max_values: dict with 'methods', 'branches', 'statements' max values
    
    Returns:
        dict with coverage percentages
    """
    methods = synthetic_tests[:, 0]
    branches = synthetic_tests[:, 1]
    statements = synthetic_tests[:, 2]
    
    # Debug: print raw values
    print(f"      Raw synthetic values:")
    print(f"         methods: [{methods.min():.2f}, {methods.max():.2f}], mean={methods.mean():.2f}")
    print(f"         branches: [{branches.min():.2f}, {branches.max():.2f}], mean={branches.mean():.2f}")
    print(f"         statements: [{statements.min():.2f}, {statements.max():.2f}], mean={statements.mean():.2f}")
    
    # Clip to valid range [0, max]
    methods_clipped = np.clip(methods, 0, max_values['methods'])
    branches_clipped = np.clip(branches, 0, max_values['branches'])
    statements_clipped = np.clip(statements, 0, max_values['statements'])
    
    # Debug: check how many were clipped
    n_methods_clipped = np.sum(methods != methods_clipped)
    n_branches_clipped = np.sum(branches != branches_clipped)
    n_statements_clipped = np.sum(statements != statements_clipped)
    
    print(f"      Clipped values: methods={n_methods_clipped}, branches={n_branches_clipped}, statements={n_statements_clipped}")
    
    # Calculate coverage as percentage of max
    line_cov = np.mean(np.minimum(methods_clipped / max_values['methods'], 1.0)) * 100
    branch_cov = np.mean(np.minimum(branches_clipped / max_values['branches'], 1.0)) * 100
    statement_cov = np.mean(np.minimum(statements_clipped / max_values['statements'], 1.0)) * 100
    
    print(f"      Coverage: line={line_cov:.2f}%, branch={branch_cov:.2f}%, statement={statement_cov:.2f}%")
    
    return {
        'line_cov': line_cov,
        'branch_cov': branch_cov,
        'statement_cov': statement_cov
    }


def debug_training(epochs=10, batch_size=16, z_dim=100):
    """Run debug training with detailed logging"""
    
    print("=" * 70)
    print("DEBUG COVERAGE CALCULATION")
    print("=" * 70)
    
    # Load data
    df_real = load_dataset()
    norm_stats = load_norm_stats()
    
    # Max values for coverage calculation
    max_values = {
        'methods': 800.0,
        'branches': 1000.0,
        'statements': 1000.0
    }
    print(f"\n📊 Max values for coverage: {max_values}")
    
    # Setup models
    print(f"\n🤖 Setting up models...")
    device = torch.device('cpu')
    generator = Generator(z_dim=z_dim, output_dim=3).to(device)
    critic = Critic(input_dim=3).to(device)
    
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.00005, betas=(0.5, 0.999))
    optimizer_c = torch.optim.Adam(critic.parameters(), lr=0.00005, betas=(0.5, 0.999))
    
    # Load dataset
    dataset = JUnitDataset(
        csv_path='python/data/junit_data_cleaned_strict.csv'
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"   Generator: {sum(p.numel() for p in generator.parameters())} params")
    print(f"   Critic: {sum(p.numel() for p in critic.parameters())} params")
    print(f"   Dataloader: {len(dataloader)} batches of {batch_size}")
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"STARTING DEBUG TRAINING - {epochs} EPOCHS")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        generator.train()
        critic.train()
        
        epoch_gen_losses = []
        epoch_critic_losses = []
        
        # Train for one epoch
        for batch_idx, real_data in enumerate(dataloader):
            real_data = real_data.to(device)
            batch_size_current = real_data.size(0)
            
            # Train Critic
            for _ in range(5):  # n_critic iterations
                z = torch.randn(batch_size_current, z_dim, device=device)
                fake_data = generator(z)
                
                critic_real = critic(real_data)
                critic_fake = critic(fake_data.detach())
                
                # Gradient penalty (simplified for debug)
                alpha = torch.rand(batch_size_current, 1, device=device)
                interpolates = alpha * real_data + (1 - alpha) * fake_data.detach()
                interpolates.requires_grad_(True)
                critic_interpolates = critic(interpolates)
                
                gradients = torch.autograd.grad(
                    outputs=critic_interpolates,
                    inputs=interpolates,
                    grad_outputs=torch.ones_like(critic_interpolates),
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                
                critic_loss = -critic_real.mean() + critic_fake.mean() + 5.0 * gradient_penalty
                
                optimizer_c.zero_grad()
                critic_loss.backward()
                optimizer_c.step()
            
            # Train Generator
            z = torch.randn(batch_size_current, z_dim, device=device)
            fake_data = generator(z)
            gen_loss = -critic(fake_data).mean()
            
            optimizer_g.zero_grad()
            gen_loss.backward()
            optimizer_g.step()
            
            epoch_gen_losses.append(gen_loss.item())
            epoch_critic_losses.append(critic_loss.item())
        
        # Calculate average losses
        avg_gen_loss = np.mean(epoch_gen_losses)
        avg_critic_loss = np.mean(epoch_critic_losses)
        
        # Generate synthetic samples for evaluation
        generator.eval()
        with torch.no_grad():
            z = torch.randn(256, z_dim, device=device)
            synthetic_norm = generator(z)
        
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{epochs}")
        print(f"{'='*70}")
        print(f"   Losses: Gen={avg_gen_loss:.4f}, Critic={avg_critic_loss:.4f}")
        
        # Debug normalized synthetic data
        print(f"\n   Normalized synthetic data (should be in [-1, 1]):")
        print(f"      Range: [{synthetic_norm.min().item():.2f}, {synthetic_norm.max().item():.2f}]")
        print(f"      Mean: {synthetic_norm.mean(dim=0).cpu().numpy()}")
        
        # Denormalize
        synthetic_denorm = denormalize(synthetic_norm, norm_stats)
        synthetic_np = synthetic_denorm.cpu().numpy()
        
        print(f"\n   Denormalized synthetic data:")
        # Calculate coverage with detailed logging
        coverage = calculate_coverage(synthetic_np, max_values)
        
        # Calculate diversity
        unique_rows = len(np.unique(synthetic_np, axis=0))
        diversity = unique_rows / len(synthetic_np)
        print(f"      Diversity: {diversity:.4f} ({unique_rows}/{len(synthetic_np)} unique)")
        
        # Compare with real data distribution
        print(f"\n   Distribution comparison (synthetic vs real):")
        for i, name in enumerate(['methods', 'branches', 'statements']):
            real_mean = df_real[name].mean()
            synthetic_mean = synthetic_np[:, i].mean()
            ratio = synthetic_mean / real_mean if real_mean > 0 else 0
            print(f"      {name}: synthetic={synthetic_mean:.1f} vs real={real_mean:.1f} (ratio={ratio:.2f}x)")
    
    print(f"\n{'='*70}")
    print("DEBUG TRAINING COMPLETED")
    print(f"{'='*70}")
    print("\nKey observations to check:")
    print("1. Are normalized values in [-1, 1]? (should be YES)")
    print("2. Are denormalized values in [0, 1000]? (should be mostly YES)")
    print("3. Does coverage correlate with synthetic mean values? (should be YES)")
    print("4. Is diversity > 0.5? (should be YES for healthy training)")
    print("5. Are synthetic/real ratios reasonable? (should be 0.5-2.0x)")


if __name__ == "__main__":
    debug_training(epochs=10, batch_size=16, z_dim=100)
