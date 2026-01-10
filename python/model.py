"""
WGAN-GP Neural Network Models

This module contains the Generator and Critic (Discriminator) architectures
for the Wasserstein GAN with Gradient Penalty (WGAN-GP) implementation.

Architecture:
    - Generator: Maps random noise (z_dim) to synthetic test features (output_dim)
    - Critic: Evaluates authenticity of real vs synthetic test data
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator network for WGAN-GP.
    
    Maps random noise vectors to synthetic JUnit test features.
    
    Args:
        z_dim (int): Dimension of input noise vector (default: 100)
        output_dim (int): Dimension of output synthetic features (default: 3)
    
    Architecture:
        100 -> 256 (LeakyReLU) -> 512 (LeakyReLU) -> output_dim (Linear)
    """
    def __init__(self, z_dim, output_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, z):
        """
        Generate synthetic features from noise vector.
        
        Args:
            z (torch.Tensor): Noise vector of shape (batch_size, z_dim)
        
        Returns:
            torch.Tensor: Synthetic features of shape (batch_size, output_dim)
        """
        x = self.leaky_relu(self.fc1(z))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Critic(nn.Module):
    """
    Critic (Discriminator) network for WGAN-GP.
    
    Evaluates how realistic synthetic test features are.
    Unlike traditional GAN discriminators, outputs a score (not probability).
    
    Args:
        input_dim (int): Dimension of input features (default: 3)
        output_dim (int): Dimension of output (default: 1, not used but kept for compatibility)
    
    Architecture:
        input_dim -> 512 (LeakyReLU) -> 256 (LeakyReLU) -> 1 (Linear)
    """
    def __init__(self, input_dim, output_dim=1):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        """
        Evaluate realism of features.
        
        Args:
            x (torch.Tensor): Features to evaluate, shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Realism score of shape (batch_size, 1)
                         Higher = more realistic, Lower = more synthetic
        """
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
