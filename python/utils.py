"""
WGAN-GP Utility Functions

This module provides helper functions for WGAN-GP training, particularly
the gradient penalty computation which is crucial for training stability.

Reference: Improved Training of Wasserstein GANs (https://arxiv.org/abs/1704.00028)
"""

import torch
import torch.nn as nn


def compute_gradient_penalty(critic, real_data, fake_data, device, batch_size, lambda_gp=10):
    """
    Compute gradient penalty for WGAN-GP.
    
    The gradient penalty enforces the Lipschitz constraint on the critic
    by penalizing gradients with norm > 1 on interpolated samples.
    
    Args:
        critic (nn.Module): Critic network
        real_data (torch.Tensor): Real samples of shape (batch_size, num_features)
        fake_data (torch.Tensor): Fake samples of shape (batch_size, num_features)
        device (torch.device): Device to compute on (cpu or cuda)
        batch_size (int): Batch size
        lambda_gp (float): Gradient penalty coefficient (default: 10)
    
    Returns:
        torch.Tensor: Gradient penalty loss
    
    Process:
        1. Create interpolated samples between real and fake
        2. Compute critic output on interpolated samples
        3. Compute gradients with respect to interpolated samples
        4. Penalize gradients with norm != 1
    """
    alpha = torch.rand(batch_size, 1, device=device)
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
    
    d_interpolates = critic(interpolates)
    fake = torch.ones(batch_size, 1, device=device, requires_grad=True)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    
    return gradient_penalty
