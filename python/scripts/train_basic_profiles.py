"""
Train with basic profile configurations (AGGRESSIVE, EXPERIMENTAL, CONSERVATIVE).
Wrapper around train_profile_aware.py that converts basic profiles to advanced profile parameters.
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
from datetime import datetime

from model import Generator, Critic
from junit_datasets import JUnitDataset
from config import ModelConfig, set_random_seed
from utils import compute_gradient_penalty

# Profile definitions
PROFILES = {
    'aggressive': {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'lambda_gp': 2.0,
        'n_critic': 5,
        'description': 'Fast training, may be unstable'
    },
    'experimental': {
        'epochs': 80,
        'batch_size': 12,
        'learning_rate': 0.00008,
        'lambda_gp': 7.5,
        'n_critic': 8,
        'description': 'Improved hybrid configuration'
    },
    'conservative': {
        'epochs': 150,
        'batch_size': 8,
        'learning_rate': 0.000025,
        'lambda_gp': 10.0,
        'n_critic': 12,
        'description': 'Maximum stability, very slow'
    },
    'balanced': {
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.00005,
        'lambda_gp': 5.0,
        'n_critic': 7,
        'description': 'Production-ready, stable training'
    }
}

def train_basic_profile(profile_name, epochs=None, seed=42, device='cpu'):
    """Train with a basic profile configuration."""
    
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Choose from: {list(PROFILES.keys())}")
    
    # Get profile config
    profile = PROFILES[profile_name]
    if epochs is None:
        epochs = profile['epochs']
    
    # Set random seed
    set_random_seed(seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f'outputs/{profile_name}_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    print(f"{'='*70}")
    print(f"TRAINING: {profile_name.upper()}")
    print(f"{'='*70}")
    print(f"Description: {profile['description']}")
    print(f"Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {profile['batch_size']}")
    print(f"  Learning rate: {profile['learning_rate']}")
    print(f"  Lambda GP: {profile['lambda_gp']}")
    print(f"  Critic iterations: {profile['n_critic']}")
    print(f"  Seed: {seed}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Load dataset
    dataset_path = 'python/data/junit_data_cleaned_strict.csv'
    norm_stats_path = 'python/data/junit_data_cleaned_norm_stats.json'
    
    dataset = JUnitDataset(dataset_path, norm_stats_path)
    dataloader = DataLoader(
        dataset,
        batch_size=profile['batch_size'],
        shuffle=True,
        drop_last=True
    )
    
    print(f"Dataset: {len(dataset)} samples")
    print(f"Batches per epoch: {len(dataloader)}\n")
    
    # Initialize models
    model_config = ModelConfig()
    generator = Generator(z_dim=model_config.z_dim, output_dim=model_config.output_dim).to(device)
    critic = Critic(input_dim=model_config.output_dim).to(device)
    
    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=profile['learning_rate'], betas=(0.5, 0.999))
    optimizer_c = optim.Adam(critic.parameters(), lr=profile['learning_rate'], betas=(0.5, 0.999))
    
    # Load normalization stats for coverage
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    num_features = norm_stats['num_features']
    max_values = np.array([norm_stats['features'][f'feature_{i}']['max'] for i in range(num_features)])
    min_values = np.array([norm_stats['features'][f'feature_{i}']['min'] for i in range(num_features)])
    
    def denormalize(normalized_data):
        data_01 = (normalized_data + 1) / 2
        return data_01 * (max_values - min_values) + min_values
    
    # Training history
    history = []
    
    print("Starting training...\n")
    
    for epoch in range(epochs):
        generator.train()
        critic.train()
        
        epoch_g_loss = 0
        epoch_c_loss = 0
        num_batches = 0
        
        for real_data in dataloader:
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            
            # Train Critic
            for _ in range(profile['n_critic']):
                optimizer_c.zero_grad()
                
                z = torch.randn(batch_size, model_config.z_dim).to(device)
                fake_data = generator(z)
                
                critic_real = critic(real_data)
                critic_fake = critic(fake_data.detach())
                
                gp = compute_gradient_penalty(
                    critic, real_data, fake_data, device, batch_size,
                    lambda_gp=profile['lambda_gp']
                )
                
                critic_loss = critic_fake.mean() - critic_real.mean() + gp
                critic_loss.backward()
                optimizer_c.step()
                
                epoch_c_loss += critic_loss.item()
            
            # Train Generator
            optimizer_g.zero_grad()
            
            z = torch.randn(batch_size, model_config.z_dim).to(device)
            fake_data = generator(z)
            
            critic_fake = critic(fake_data)
            gen_loss = -critic_fake.mean()
            
            gen_loss.backward()
            optimizer_g.step()
            
            epoch_g_loss += gen_loss.item()
            num_batches += 1
        
        # Calculate metrics
        avg_g_loss = epoch_g_loss / num_batches
        avg_c_loss = epoch_c_loss / (num_batches * profile['n_critic'])
        
        # Coverage metrics
        with torch.no_grad():
            z_eval = torch.randn(1000, model_config.z_dim).to(device)
            fake_eval_norm = generator(z_eval).cpu().numpy()
            fake_eval = denormalize(fake_eval_norm)
        
        max_achieved = fake_eval.max(axis=0)
        coverage_pct = (max_achieved / max_values * 100).mean()
        diversity = len(np.unique(fake_eval.round(1), axis=0)) / len(fake_eval)
        
        history.append({
            'epoch': epoch + 1,
            'gen_loss': avg_g_loss,
            'critic_loss': avg_c_loss,
            'coverage': coverage_pct,
            'diversity': diversity,
            'max_methods': max_achieved[0],
            'max_branches': max_achieved[1],
            'max_statements': max_achieved[2]
        })
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"G Loss: {avg_g_loss:7.4f} | "
                  f"C Loss: {avg_c_loss:7.4f} | "
                  f"Coverage: {coverage_pct:.2f}% | "
                  f"Max: [{max_achieved[0]:.0f}, {max_achieved[1]:.0f}, {max_achieved[2]:.0f}]")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'critic': critic.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_c': optimizer_c.state_dict(),
            }, models_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Save final models
    torch.save(generator.state_dict(), models_dir / 'generator_final.pth')
    torch.save(critic.state_dict(), models_dir / 'critic_final.pth')
    
    # Save history
    df_history = pd.DataFrame(history)
    df_history.to_csv(output_dir / 'training_history.csv', index=False)
    
    # Save config
    config_dict = {
        'profile': profile_name,
        'profile_config': profile,
        'epochs': epochs,
        'seed': seed,
        'device': device,
        'timestamp': timestamp
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    final_metrics = history[-1]
    print(f"\n{'='*70}")
    print(f"✅ Training completed!")
    print(f"   Final coverage: {final_metrics['coverage']:.2f}%")
    print(f"   Final diversity: {final_metrics['diversity']:.3f}")
    print(f"   Final max values: [{final_metrics['max_methods']:.0f}, "
          f"{final_metrics['max_branches']:.0f}, {final_metrics['max_statements']:.0f}]")
    print(f"   Output: {output_dir}")
    print(f"{'='*70}\n")
    
    return output_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=str, required=True,
                        choices=list(PROFILES.keys()),
                        help='Training profile to use')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from profile)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    train_basic_profile(args.profile, args.epochs, args.seed, args.device)
