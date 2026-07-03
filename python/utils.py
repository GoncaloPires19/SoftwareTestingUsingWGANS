"""
WGAN-GP Utility Functions

Gradient penalty, feature matching loss, diversity scoring, and Wasserstein
distance — all used by the training loop and diagnostic scripts.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Gradient penalty (core WGAN-GP constraint)
# ---------------------------------------------------------------------------

def compute_gradient_penalty(
    critic: nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    device: torch.device,
    batch_size: int,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Compute WGAN-GP gradient penalty (Gulrajani et al., 2017).

    Enforces the Lipschitz-1 constraint on the critic by penalising gradients
    that deviate from unit norm on linear interpolates between real and fake samples.

    Args:
        critic: The critic network.
        real_data: Real samples — shape (B, D).
        fake_data: Fake samples — shape (B, D). Must already be detached from the
            generator graph to avoid unintended gradient flow.
        device: Computation device.
        batch_size: Expected batch size (actual size may be smaller for the last batch).
        lambda_gp: Penalty coefficient. Default 10 follows the original paper.

    Returns:
        Scalar gradient penalty tensor (with graph attached for backward).
    """
    current_batch_size = real_data.size(0)

    alpha = torch.rand(current_batch_size, 1, device=device)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

    d_interpolates = critic(interpolates)

    grad_outputs = torch.ones(current_batch_size, 1, device=device, requires_grad=False)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)

    # Numerically stable L2 norm — epsilon prevents NaN when gradient is exactly 0
    grad_norms = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    return ((grad_norms - 1) ** 2).mean() * lambda_gp


# ---------------------------------------------------------------------------
# Feature matching loss
# ---------------------------------------------------------------------------

def compute_feature_matching_loss(
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
) -> torch.Tensor:
    """
    Feature matching loss: penalise mismatch in first and second moments.

    Encourages the generator to reproduce the marginal mean and standard deviation
    of each feature, which directly combats mode collapse and variance collapse.

    Loss = mean(|mean(fake) - mean(real)|) + mean(|std(fake) - std(real)|)

    Args:
        real_data: Real samples — shape (B, D).
        fake_data: Fake samples — shape (B, D).

    Returns:
        Scalar loss tensor.
    """
    real_mean = real_data.mean(dim=0)
    fake_mean = fake_data.mean(dim=0)

    # Unbiased std with correction=1; clamp denominator to avoid NaN on constant features
    real_std = real_data.std(dim=0, correction=1).clamp(min=1e-8)
    fake_std = fake_data.std(dim=0, correction=1)

    mean_loss = torch.abs(fake_mean - real_mean).mean()
    std_loss = torch.abs(fake_std - real_std).mean()

    return mean_loss + std_loss


# ---------------------------------------------------------------------------
# Diversity score
# ---------------------------------------------------------------------------

def compute_diversity_score(
    synthetic: np.ndarray,
    real: np.ndarray,
) -> float:
    """
    Compute a diversity score in [0, 1] comparing synthetic vs real variance.

    Score = mean( clip(std_synth / std_real, 0, 2) ) / 2

    A score of 0.5 means synthetic variance matches real variance exactly.
    Values below 0.3 indicate mode collapse; values above 0.7 indicate
    over-dispersion.

    Args:
        synthetic: Synthetic samples — shape (N, D).
        real: Real samples — shape (M, D).

    Returns:
        Diversity score in [0, 1].
    """
    real_std = real.std(axis=0) + 1e-8
    synth_std = synthetic.std(axis=0)
    ratio = np.clip(synth_std / real_std, 0.0, 2.0)
    return float(ratio.mean() / 2.0)


# ---------------------------------------------------------------------------
# Wasserstein distance (per-feature average)
# ---------------------------------------------------------------------------

def compute_wasserstein_distance(
    synthetic: np.ndarray,
    real: np.ndarray,
) -> float:
    """
    Compute the mean 1-D Wasserstein distance across all features.

    Uses scipy's exact solver (sort-based), which is exact for 1-D marginals.
    Lower values indicate that synthetic samples better approximate the real
    data distribution.

    Args:
        synthetic: Synthetic samples — shape (N, D).
        real: Real samples — shape (M, D).

    Returns:
        Mean Wasserstein-1 distance across features.
    """
    from scipy.stats import wasserstein_distance  # lazy import — scipy optional at module level

    n_features = real.shape[1] if real.ndim > 1 else 1
    distances = []

    if real.ndim == 1:
        return float(wasserstein_distance(real, synthetic))

    for i in range(n_features):
        distances.append(wasserstein_distance(real[:, i], synthetic[:, i]))

    return float(np.mean(distances))


# ---------------------------------------------------------------------------
# Composite quality score (early stopping metric)
# ---------------------------------------------------------------------------

def compute_composite_score(
    diversity: float,
    wasserstein: float,
    w1_reference: float = 1.0,
    diversity_weight: float = 0.4,
    quality_weight: float = 0.6,
) -> float:
    """
    Composite score that balances distribution quality and sample diversity.

    This is the recommended early stopping metric — it prevents picking
    epoch 1 (random noise, high diversity but terrible quality) as the best
    model by penalising high Wasserstein distance.

    Score = quality_weight * quality + diversity_weight * diversity

    where quality = clip(1 - W1 / w1_reference, 0, 1)

    Args:
        diversity: Diversity score in [0, 1] (higher = more diverse).
        wasserstein: Current Wasserstein-1 distance (lower = better quality).
        w1_reference: Reference W1 (typically from epoch 1 or untrained model).
            Used to normalise quality to [0, 1]. Defaults to 1.0.
        diversity_weight: Weight for diversity term. Default 0.4.
        quality_weight: Weight for quality term. Default 0.6.

    Returns:
        Composite score in [0, 1] — higher is better.
    """
    quality = float(np.clip(1.0 - wasserstein / max(w1_reference, 1e-8), 0.0, 1.0))
    return quality_weight * quality + diversity_weight * diversity


# ---------------------------------------------------------------------------
# Gradient statistics (for training diagnostics)
# ---------------------------------------------------------------------------

def get_model_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """
    Compute L2 gradient norm statistics for all model parameters.

    Uses Python built-ins after extracting scalar norms to avoid creating
    additional PyTorch tensors during the backward pass.

    Returns:
        Dictionary with 'mean_grad_norm', 'max_grad_norm', 'min_grad_norm'.
        All values are 0.0 if no gradients are available yet.
    """
    grad_norms: list[float] = []

    for param in model.parameters():
        if param.grad is not None:
            norm = param.grad.data.norm(2).item()
            if not (math.isnan(norm) or math.isinf(norm)):
                grad_norms.append(norm)

    if not grad_norms:
        return {"mean_grad_norm": 0.0, "max_grad_norm": 0.0, "min_grad_norm": 0.0}

    return {
        "mean_grad_norm": sum(grad_norms) / len(grad_norms),
        "max_grad_norm": max(grad_norms),
        "min_grad_norm": min(grad_norms),
    }
