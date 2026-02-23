"""
WGAN-GP Utility Functions

This module provides helper functions for WGAN-GP training, particularly
the gradient penalty computation which is crucial for training stability.

Also includes diagnostic utilities for monitoring gradient statistics during training.

Reference: Improved Training of Wasserstein GANs (https://arxiv.org/abs/1704.00028)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


def get_model_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient statistics for a model during training.
    
    Useful for detecting training issues:
    - Low gradient norms: Training is too conservative
    - High gradient norms: Training is unstable
    - NaN/Inf: Numerical issues in backpropagation
    
    Args:
        model (nn.Module): Neural network model to analyze
    
    Returns:
        Dict[str, float]: Dictionary containing:
            - 'mean_grad_norm': Mean gradient norm across all parameters
            - 'max_grad_norm': Maximum gradient norm
            - 'min_grad_norm': Minimum gradient norm (excluding zeros)
            - 'grad_norms': List of per-parameter gradient norms
    
    Example:
        >>> stats = get_model_gradient_stats(critic)
        >>> print(f"Grad norms: mean={stats['mean_grad_norm']:.4f}, max={stats['max_grad_norm']:.4f}")
    """
    grad_norms = []
    
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            if not (torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm))):
                grad_norms.append(grad_norm)
    
    if len(grad_norms) == 0:
        return {
            'mean_grad_norm': 0.0,
            'max_grad_norm': 0.0,
            'min_grad_norm': 0.0,
        }
    
    grad_norms_tensor = torch.tensor(grad_norms)
    
    return {
        'mean_grad_norm': torch.mean(grad_norms_tensor).item(),
        'max_grad_norm': torch.max(grad_norms_tensor).item(),
        'min_grad_norm': torch.min(grad_norms_tensor).item(),
    }


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
    alpha = torch.rand(int(batch_size), 1, device=device)
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
    
    d_interpolates = critic(interpolates)
    fake = torch.ones(int(batch_size), 1, device=device, requires_grad=True)
    
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
