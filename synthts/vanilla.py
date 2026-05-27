"""Vanilla GAN: Generator, Discriminator, and training loop."""

import numpy as np
import torch
import torch.nn as nn
from numpy.random import default_rng


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
    """Composite module holding generator and discriminator."""

    def __init__(self, generator: Generator, discriminator: Discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        fake = self.generator(z)
        return self.discriminator(fake)


def train_gan(generator, discriminator, data_scaled,
              epochs: int=5001, batch_size: int=32,
              lr_d: float=0.001, lr_g: float=0.0001,
              latent_dim: int=1, seed: int=100,
              log_interval: int=1000, device: str='cpu'):
    """Train a vanilla GAN on standardised 1-D data.

    Parameters
    ----------
    generator : Generator
    discriminator : Discriminator
    data_scaled : np.ndarray
        Standardised real data, shape (n, 1).
    epochs : int
    batch_size : int
    lr_d : float
        Discriminator learning rate.
    lr_g : float
        Generator learning rate.
    latent_dim : int
    seed : int
    log_interval : int
    device : str

    Returns
    -------
    dict
        Keys 'g_loss', 'd_loss_real', 'd_loss_fake'.
    """
    rng = default_rng(seed)
    torch.manual_seed(seed)

    gen = generator.to(device)
    disc = discriminator.to(device)

    bce = nn.BCELoss()
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g)

    ones = torch.ones(batch_size, 1, device=device)
    zeros = torch.zeros(batch_size, 1, device=device)

    n = len(data_scaled)
    data_t = torch.tensor(data_scaled, dtype=torch.float32,
                          device=device)

    history = {'g_loss': [], 'd_loss_real': [], 'd_loss_fake': []}

    for epoch in range(epochs):
        # --- train discriminator ---
        opt_d.zero_grad()
        idx = rng.integers(0, n, batch_size)
        real = data_t[idx]
        d_real = disc(real)
        loss_real = bce(d_real, ones)
        loss_real.backward()

        noise = torch.randn(batch_size, latent_dim, device=device)
        with torch.no_grad():
            fake = gen(noise)
        d_fake = disc(fake.detach())
        loss_fake = bce(d_fake, zeros)
        loss_fake.backward()
        opt_d.step()

        # --- train generator ---
        opt_g.zero_grad()
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake = gen(noise)
        d_fake = disc(fake)
        loss_g = bce(d_fake, ones)
        loss_g.backward()
        opt_g.step()

        if epoch % log_interval == 0:
            print(f'Epoch {epoch:5d} | '
                  f'D_real: {loss_real.item():.4f} '
                  f'D_fake: {loss_fake.item():.4f} '
                  f'G: {loss_g.item():.4f}')

    return history


def generate_samples(generator, n_samples: int, latent_dim: int=1,
                     device: str='cpu') -> np.ndarray:
    """Produce synthetic samples from a trained generator."""
    gen = generator.to(device)
    gen.eval()
    with torch.no_grad():
        noise = torch.randn(n_samples, latent_dim, device=device)
        samples = gen(noise)
    return samples.cpu().numpy()
