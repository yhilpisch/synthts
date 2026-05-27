"""Generator, Discriminator, and GAN modules in PyTorch."""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Dense generator: noise -> synthetic data."""

    def __init__(self, latent_dim: int=1, hidden_units: int=32,
                 output_dim: int=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """Dense discriminator: data -> probability (real vs fake)."""

    def __init__(self, input_dim: int=1, hidden_units: int=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GAN(nn.Module):
    """Composite module holding generator and discriminator.

    The GAN itself is only used for generator-training steps where
    the discriminator weights are frozen.
    """

    def __init__(self, generator: Generator, discriminator: Discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate data and pass through frozen discriminator."""
        fake = self.generator(z)
        return self.discriminator(fake)
