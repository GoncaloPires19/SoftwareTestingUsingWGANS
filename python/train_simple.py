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
    print("Loading data...")
    real_data = load_and_normalize_data('data/junit_data_cleaned.csv')
    print(f"Loaded {real_data.shape[0]} samples")
    
    dataset = TensorDataset(real_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize models
    generator = Generator(Z_DIM, 3).to(device)
    critic = Critic(3).to(device)
    
    # Optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real_samples,) in enumerate(dataloader):
            real_samples = real_samples.to(device)
            batch_size = real_samples.shape[0]
            
            # Train Critic
            for _ in range(CRITIC_ITERATIONS):
                critic_optimizer.zero_grad()
                
                z = torch.randn(batch_size, Z_DIM, device=device)
                fake_samples = generator(z)
                
                real_output = critic(real_samples)
                fake_output = critic(fake_samples.detach())
                
                gp = calculate_gradient_penalty(critic, real_samples, fake_samples.detach(), batch_size)
                critic_loss = fake_output.mean() - real_output.mean() + LAMBDA_GP * gp
                
                critic_loss.backward()
                critic_optimizer.step()
            
            # Train Generator
            gen_optimizer.zero_grad()
            z = torch.randn(batch_size, Z_DIM, device=device)
            fake_samples = generator(z)
            fake_output = critic(fake_samples)
            gen_loss = -fake_output.mean()
            
            gen_loss.backward()
            gen_optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | D_loss: {critic_loss.item():.4f} | G_loss: {gen_loss.item():.4f}")
        
        # Save checkpoint
        torch.save(generator.state_dict(), f'models/generator_epoch{epoch+1}.pth')
    
    print("Training completed!")

if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)
    train()
