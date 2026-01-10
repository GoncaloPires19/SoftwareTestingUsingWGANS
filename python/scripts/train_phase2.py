import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import Generator, Critic

# Configuration
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 50
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
Z_DIM = 100

device = torch.device('cpu')

def load_and_normalize_data(csv_path):
    """Load and normalize data to [-1, 1]"""
    df = pd.read_csv(csv_path, header=None)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    data = torch.tensor(df.values, dtype=torch.float32)
    
    # Normalize to [-1, 1]
    data_min = data.min(dim=0)[0]
    data_max = data.max(dim=0)[0]
    data_range = data_max - data_min
    data_range[data_range == 0] = 1
    
    normalized = 2 * (data - data_min) / data_range - 1
    normalized = torch.where(torch.isnan(normalized), torch.zeros_like(normalized), normalized)
    normalized = torch.where(torch.isinf(normalized), torch.zeros_like(normalized), normalized)
    
    return normalized

def load_combined_data():
    print("Loading combined dataset...")
    # Load cleaned real data
    real_data = load_and_normalize_data('data/junit_data_cleaned.csv')
    # Load synthetic data
    synthetic_data = load_and_normalize_data('synthetic_tests.csv')
    
    # Combine datasets
    combined_data = torch.cat([real_data, synthetic_data], dim=0)
    
    print(f"Real data: {real_data.shape[0]} samples")
    print(f"Synthetic data: {synthetic_data.shape[0]} samples")
    print(f"Combined data: {combined_data.shape[0]} samples")
    
    return combined_data

def calculate_gradient_penalty(critic, real_data, fake_data, batch_size):
    alpha = torch.rand(batch_size, 1, device=device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    d_interpolated = critic(interpolated)
    fake = torch.ones(batch_size, 1, device=device, requires_grad=True)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = torch.mean((gradients_norm - 1) ** 2)
    
    return gradient_penalty

def train():
    # Load data
    combined_data = load_combined_data()
    dataset = TensorDataset(combined_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize models
    generator = Generator(Z_DIM, 3).to(device)
    critic = Critic(3).to(device)
    
    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    optimizer_c = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    
    print("Starting Phase 2 Training on Combined Dataset...")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total samples: {len(combined_data)}")
    
    for epoch in range(NUM_EPOCHS):
        total_loss_c = 0
        total_loss_g = 0
        batch_count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for real_data, in pbar:
            real_data = real_data.to(device).float()
            batch_size = real_data.size(0)
            
            # Train Critic
            for _ in range(CRITIC_ITERATIONS):
                z = torch.randn(batch_size, Z_DIM, device=device)
                fake_data = generator(z)
                
                critic_real = critic(real_data)
                critic_fake = critic(fake_data.detach())
                
                gradient_penalty = calculate_gradient_penalty(critic, real_data, fake_data.detach(), batch_size)
                loss_c = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gradient_penalty
                
                optimizer_c.zero_grad()
                loss_c.backward()
                optimizer_c.step()
                
                total_loss_c += loss_c.item()
            
            # Train Generator
            z = torch.randn(batch_size, Z_DIM, device=device)
            fake_data = generator(z)
            critic_fake = critic(fake_data)
            loss_g = -torch.mean(critic_fake)
            
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            
            total_loss_g += loss_g.item()
            batch_count += 1
            
            pbar.set_postfix({
                'Loss_D': f'{total_loss_c/(batch_count*CRITIC_ITERATIONS):.4f}',
                'Loss_G': f'{total_loss_g/batch_count:.4f}'
            })
        
        avg_loss_c = total_loss_c / (batch_count * CRITIC_ITERATIONS)
        avg_loss_g = total_loss_g / batch_count
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Critic Loss: {avg_loss_c:.4f}, Generator Loss: {avg_loss_g:.4f}")
        
        # Save checkpoints
        torch.save(generator.state_dict(), f'models/generator_phase2_epoch{epoch+1}.pth')
        torch.save(critic.state_dict(), f'models/critic_phase2_epoch{epoch+1}.pth')

if __name__ == '__main__':
    train()
