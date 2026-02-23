"""
Continue training from a saved checkpoint for additional epochs.
Useful for extending training to see if max coverage improves.
"""

import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
import sys
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent))

from model import Generator, Critic
from junit_datasets import JUnitDataset
from torch.utils.data import DataLoader
from config import ModelConfig, TrainingConfig
from utils import compute_gradient_penalty

def train_continuation(
    generator_path,
    critic_path,
    dataset_path,
    norm_stats_path,
    additional_epochs=100,
    checkpoint_interval=10,
    output_dir=None,
    device='cpu'
):
    """
    Continue training from saved models.
    
    Args:
        generator_path: Path to saved generator model
        critic_path: Path to saved critic model
        dataset_path: Path to training dataset CSV
        norm_stats_path: Path to normalization stats JSON
        additional_epochs: Number of additional epochs to train
        checkpoint_interval: Save checkpoint every N epochs
        output_dir: Directory to save results (auto-generated if None)
        device: 'cpu' or 'cuda'
    """
    # Initialize configs
    model_config = ModelConfig()
    train_config = TrainingConfig()
    train_config.epochs = additional_epochs
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/continuation_{timestamp}")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"Continuation Training Configuration:")
    print(f"  Additional epochs: {additional_epochs}")
    print(f"  Checkpoint interval: {checkpoint_interval}")
    print(f"  Output directory: {output_dir}")
    print(f"  Device: {device}\n")
    
    # Load dataset
    dataset = JUnitDataset(dataset_path, norm_stats_path)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Batch size: {train_config.batch_size}")
    print(f"Batches per epoch: {len(dataloader)}\n")
    
    # Initialize models
    generator = Generator(
        z_dim=model_config.z_dim,
        output_dim=model_config.output_dim
    ).to(device)
    
    critic = Critic(
        input_dim=model_config.output_dim
    ).to(device)
    
    # Load pretrained weights
    print("Loading pretrained models...")
    gen_checkpoint = torch.load(generator_path, map_location=device, weights_only=False)
    if isinstance(gen_checkpoint, dict):
        if 'generator' in gen_checkpoint:
            generator.load_state_dict(gen_checkpoint['generator'])
            start_epoch = gen_checkpoint.get('epoch', 0) + 1
        elif 'model_state_dict' in gen_checkpoint:
            generator.load_state_dict(gen_checkpoint['model_state_dict'])
            start_epoch = gen_checkpoint.get('epoch', 0) + 1
        else:
            generator.load_state_dict(gen_checkpoint)
            start_epoch = 1
    else:
        generator.load_state_dict(gen_checkpoint)
        start_epoch = 1
    
    critic_checkpoint = torch.load(critic_path, map_location=device, weights_only=False)
    if isinstance(critic_checkpoint, dict):
        if 'critic' in critic_checkpoint:
            critic.load_state_dict(critic_checkpoint['critic'])
        elif 'model_state_dict' in critic_checkpoint:
            critic.load_state_dict(critic_checkpoint['model_state_dict'])
        else:
            critic.load_state_dict(critic_checkpoint)
    else:
        critic.load_state_dict(critic_checkpoint)
    
    print(f"✓ Models loaded (resuming from epoch {start_epoch})\n")
    
    # Initialize optimizers
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2)
    )
    
    optimizer_c = optim.Adam(
        critic.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2)
    )
    
    # Load normalization stats for metrics
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    # Extract min/max from features
    num_features = norm_stats['num_features']
    max_values = np.array([norm_stats['features'][f'feature_{i}']['max'] for i in range(num_features)])
    min_values = np.array([norm_stats['features'][f'feature_{i}']['min'] for i in range(num_features)])
    
    def denormalize(normalized_data):
        """Denormalize from [-1, 1] to original scale."""
        data_01 = (normalized_data + 1) / 2
        return data_01 * (max_values - min_values) + min_values
    
    # Training history
    history = []
    
    print("Starting continuation training...\n")
    
    for epoch in range(additional_epochs):
        epoch_num = start_epoch + epoch
        generator.train()
        critic.train()
        
        epoch_g_loss = 0
        epoch_c_loss = 0
        num_batches = 0
        
        for real_data in dataloader:
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            
            # Train Critic
            for _ in range(train_config.n_critic):
                optimizer_c.zero_grad()
                
                z = torch.randn(batch_size, model_config.z_dim).to(device)
                fake_data = generator(z)
                
                critic_real = critic(real_data)
                critic_fake = critic(fake_data.detach())
                
                gp = compute_gradient_penalty(
                    critic, real_data, fake_data, device, batch_size, 
                    lambda_gp=train_config.lambda_gp
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
        
        # Calculate epoch metrics
        avg_g_loss = epoch_g_loss / num_batches
        avg_c_loss = epoch_c_loss / (num_batches * train_config.n_critic)
        
        # Calculate simple coverage metrics
        with torch.no_grad():
            z_eval = torch.randn(1000, model_config.z_dim).to(device)
            fake_eval_norm = generator(z_eval).cpu().numpy()
            fake_eval = denormalize(fake_eval_norm)
        
        max_achieved = fake_eval.max(axis=0)
        coverage_pct = (max_achieved / max_values * 100).mean()
        diversity = len(np.unique(fake_eval.round(1), axis=0)) / len(fake_eval)
        
        history.append({
            'epoch': epoch_num,
            'gen_loss': avg_g_loss,
            'critic_loss': avg_c_loss,
            'coverage': coverage_pct,
            'diversity': diversity,
            'max_methods': max_achieved[0],
            'max_branches': max_achieved[1],
            'max_statements': max_achieved[2]
        })
        
        print(f"Epoch {epoch_num:3d} | "
              f"G Loss: {avg_g_loss:7.4f} | "
              f"C Loss: {avg_c_loss:7.4f} | "
              f"Coverage: {coverage_pct:.2f}% | "
              f"Max: [{max_achieved[0]:.0f}, {max_achieved[1]:.0f}, {max_achieved[2]:.0f}]")
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == additional_epochs:
            checkpoint_path = models_dir / f"checkpoint_epoch_{epoch_num}.pth"
            torch.save({
                'epoch': epoch_num,
                'generator': generator.state_dict(),
                'critic': critic.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_c': optimizer_c.state_dict(),
                'config': {
                    'model': model_config.__dict__,
                    'training': train_config.__dict__
                }
            }, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path.name}")
    
    # Save final models
    torch.save(generator.state_dict(), models_dir / "generator_final.pth")
    torch.save(critic.state_dict(), models_dir / "critic_final.pth")
    
    # Save training history
    df_history = pd.DataFrame(history)
    df_history.to_csv(output_dir / "training_history.csv", index=False)
    
    final_metrics = history[-1]
    print(f"\n✓ Continuation training completed!")
    print(f"  Final coverage: {final_metrics['coverage']:.2f}%")
    print(f"  Final diversity: {final_metrics['diversity']:.3f}")
    print(f"  Final max values: [{final_metrics['max_methods']:.0f}, {final_metrics['max_branches']:.0f}, {final_metrics['max_statements']:.0f}]")
    print(f"  Models saved to: {models_dir}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Continue WGAN training from checkpoint")
    parser.add_argument('--generator', type=str, required=True,
                        help='Path to saved generator model')
    parser.add_argument('--critic', type=str, required=True,
                        help='Path to saved critic model')
    parser.add_argument('--dataset', type=str,
                        default='python/data/junit_data_cleaned_strict.csv',
                        help='Path to training dataset')
    parser.add_argument('--norm-stats', type=str,
                        default='python/data/junit_data_cleaned_norm_stats.json',
                        help='Path to normalization stats')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of additional epochs')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (auto-generated if not specified)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    train_continuation(
        generator_path=args.generator,
        critic_path=args.critic,
        dataset_path=args.dataset,
        norm_stats_path=args.norm_stats,
        additional_epochs=args.epochs,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        device=args.device
    )

if __name__ == '__main__':
    main()
