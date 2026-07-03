"""
WGAN-GP Neural Network Models

Generator and Critic architectures for WGAN-GP.
Hidden layer dimensions are fully configurable.

Key design choices:
- Generator: Tanh output forces [-1, 1] — real data MUST be normalised to same range.
- Critic: Linear output (no sigmoid) as required by Wasserstein loss.
- MinibatchStdDev: optional, proven technique against mode collapse. Appends the
  mean per-feature standard deviation across the batch as an extra input to the
  critic's final layer, giving the critic batch-diversity information.
"""

from __future__ import annotations

import warnings
from typing import List

import torch
import torch.nn as nn


def _weights_init(m: nn.Module) -> None:
    """He/Kaiming init for Linear layers — improves convergence with LeakyReLU."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _maybe_spectral(layer: nn.Linear, use_spectral_norm: bool) -> nn.Module:
    return nn.utils.spectral_norm(layer) if use_spectral_norm else layer


class Generator(nn.Module):
    """
    WGAN-GP Generator.

    Maps latent noise vectors to synthetic JUnit test metrics.
    Output is bounded to [-1, 1] via Tanh.

    Args:
        z_dim: Latent space dimension. Default 100.
        output_dim: Number of output features (e.g. 3 = methods, branches, statements).
        hidden_dims: Hidden layer sizes. Defaults to [256, 512].
        use_batch_norm: Add BatchNorm1d after each hidden layer.
    """

    def __init__(
        self,
        z_dim: int = 100,
        output_dim: int = 3,
        hidden_dims: List[int] | None = None,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.output_dim = output_dim

        if hidden_dims is None:
            hidden_dims = [256, 512]

        layers: list[nn.Module] = []
        in_dim = z_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Critic(nn.Module):
    """
    WGAN-GP Critic (Discriminator).

    Outputs a scalar Wasserstein score — no sigmoid activation.

    MinibatchStdDev: when enabled, appends the mean standard deviation of the
    current batch's hidden representation as an extra feature to the final
    linear layer. This gives the critic batch-level diversity information and
    is the most effective technique against mode collapse in practice.

    Spectral normalization is theoretically incompatible with gradient penalty
    and MUST NOT be combined with lambda_gp > 0.

    Args:
        input_dim: Input feature count. Must match Generator output_dim.
        hidden_dims: Hidden layer sizes. Defaults to [512, 256].
        use_spectral_norm: Wrap linear layers with spectral normalization.
            Raises a warning if True (incompatible with GP).
        use_minibatch_std: Append batch std deviation before the output layer.
            Recommended — combats mode collapse without breaking gradient penalty.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: List[int] | None = None,
        use_spectral_norm: bool = False,
        use_minibatch_std: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_spectral_norm = use_spectral_norm
        self.use_minibatch_std = use_minibatch_std

        if use_spectral_norm:
            warnings.warn(
                "use_spectral_norm=True is theoretically incompatible with WGAN-GP gradient "
                "penalty. This over-constrains the Lipschitz space and harms convergence.",
                UserWarning,
                stacklevel=2,
            )

        if hidden_dims is None:
            hidden_dims = [512, 256]

        # Feature extraction trunk
        trunk_layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            trunk_layers.append(_maybe_spectral(nn.Linear(in_dim, h), use_spectral_norm))
            trunk_layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)

        # Output layer — receives trunk output + optional minibatch std scalar
        out_in_dim = hidden_dims[-1] + (1 if use_minibatch_std else 0)
        self.output_layer = _maybe_spectral(nn.Linear(out_in_dim, 1), use_spectral_norm)

        self.apply(_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)

        if self.use_minibatch_std:
            # Mean std across the batch for each feature, then take the mean across features
            # → scalar that measures how diverse this batch is
            batch_std = features.std(dim=0).mean()
            std_feature = batch_std.expand(features.size(0), 1)
            features = torch.cat([features, std_feature], dim=1)

        return self.output_layer(features)
