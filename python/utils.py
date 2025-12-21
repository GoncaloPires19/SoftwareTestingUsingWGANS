import torch
import torch.nn as nn

def compute_gradient_penalty(critic, real_data, fake_data, device, batch_size, lambda_gp=10):
    """Compute gradient penalty for WGAN-GP"""
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
