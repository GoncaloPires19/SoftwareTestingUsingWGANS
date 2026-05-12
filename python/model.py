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


def _critic_linear(in_features: int, out_features: int, use_spectral_norm: bool) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    if use_spectral_norm:
        return nn.utils.spectral_norm(layer)
    return layer


def _weights_init(m: nn.Module) -> None:
    """He/Kaiming init for Linear layers — improves convergence speed."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)



class Generator(nn.Module):
    """
    Generator network for WGAN-GP.
    
    Maps random noise vectors to synthetic JUnit test features.
    
    Args:
        z_dim (int): Dimension of input noise vector (default: 100)
        output_dim (int): Dimension of output synthetic features (default: 4)
    
    Architecture:
        100 -> 256 (LeakyReLU) -> 512 (LeakyReLU) -> output_dim (Linear)
    
    Output features:
        - methods: number of test methods
        - branches: number of branches covered
        - statements: number of statements covered
        - method_coverage: ratio of method coverage
    """
    def __init__(self, z_dim, output_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()  # Ensures output in [-1, 1] for normalized data

        self.apply(_weights_init)
        
    def forward(self, z):
        """
        Generate synthetic features from noise vector.
        
        Args:
            z (torch.Tensor): Noise vector of shape (batch_size, z_dim)
        
        Returns:
            torch.Tensor: Synthetic features of shape (batch_size, output_dim)
                         Values are in [-1, 1] range (tanh activation)
        """
        x = self.leaky_relu(self.fc1(z))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        x = self.tanh(x)  # Constrain output to [-1, 1] for normalized data
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
    def __init__(self, input_dim, output_dim=1, use_spectral_norm=False):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.use_spectral_norm = use_spectral_norm
        
        self.fc1 = _critic_linear(input_dim, 512, use_spectral_norm)
        self.fc2 = _critic_linear(512, 256, use_spectral_norm)
        self.fc3 = _critic_linear(256, 1, use_spectral_norm)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.apply(_weights_init)
        
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
