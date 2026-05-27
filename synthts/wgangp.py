"""WGAN-GP: Critic, Generator, gradient penalty, and training loop."""

import numpy as np
import torch
import torch.nn as nn
from numpy.random import default_rng


class WGANGenerator(nn.Module):
    """Dense generator for WGAN-GP (same architecture as vanilla)."""

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


class Critic(nn.Module):
    """Wasserstein critic: data -> unbounded score (no sigmoid)."""

    def __init__(self, input_dim: int=1, hidden_units: int=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _gradient_penalty(critic, real, fake, device: str='cpu',
                      lambda_gp: float=10.0):
    """Compute gradient penalty for WGAN-GP.

    Interpolates between real and fake samples, computes critic
    gradient, and penalises deviations from unit norm.
    """
    alpha = torch.rand(real.shape[0], 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(
        True)
    score = critic(interpolated)
    grad = torch.autograd.grad(
        outputs=score, inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True,
    )[0]
    grad_norm = grad.view(grad.shape[0], -1).norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()
    return lambda_gp * penalty


def train_wgangp(generator, critic, data_scaled,
                 epochs: int=5001, batch_size: int=32,
                 n_critic: int=5, lr_g: float=0.0001,
                 lr_c: float=0.0001, latent_dim: int=1,
                 lambda_gp: float=10.0, seed: int=100,
                 log_interval: int=1000, device: str='cpu'):
    """Train a WGAN-GP on standardised 1-D data.

    Parameters
    ----------
    generator : WGANGenerator
    critic : Critic
    data_scaled : np.ndarray
        Standardised real data, shape (n, 1).
    epochs : int
    batch_size : int
    n_critic : int
        Critic updates per generator update.
    lr_g : float
        Generator learning rate.
    lr_c : float
        Critic learning rate.
    latent_dim : int
    lambda_gp : float
        Gradient-penalty coefficient.
    seed : int
    log_interval : int
    device : str

    Returns
    -------
    dict
        Keys 'g_loss', 'c_loss', 'w_distance'.
    """
    rng = default_rng(seed)
    torch.manual_seed(seed)

    gen = generator.to(device)
    crit = critic.to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g,
                             betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(crit.parameters(), lr=lr_c,
                             betas=(0.5, 0.9))

    n = len(data_scaled)
    data_t = torch.tensor(data_scaled, dtype=torch.float32,
                          device=device)

    history = {'g_loss': [], 'c_loss': [], 'w_distance': []}

    for epoch in range(epochs):
        # --- train critic (n_critic steps) ---
        for _ in range(n_critic):
            opt_c.zero_grad()
            idx = rng.integers(0, n, batch_size)
            real = data_t[idx]
            noise = torch.randn(batch_size, latent_dim,
                                device=device)
            with torch.no_grad():
                fake = gen(noise)
            score_real = crit(real)
            score_fake = crit(fake)
            gp = _gradient_penalty(crit, real, fake, device, lambda_gp)
            loss_c = (score_fake.mean() - score_real.mean()
                      + gp)
            loss_c.backward()
            opt_c.step()

        # --- train generator ---
        opt_g.zero_grad()
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake = gen(noise)
        score_fake = crit(fake)
        loss_g = -score_fake.mean()
        loss_g.backward()
        opt_g.step()

        w_dist = (score_real.mean() - score_fake.mean()).item()

        if epoch % log_interval == 0:
            print(f'Epoch {epoch:5d} | '
                  f'C: {loss_c.item():.4f} '
                  f'G: {loss_g.item():.4f} '
                  f'W: {w_dist:.4f}')

    return history


def generate_samples(generator, n_samples: int, latent_dim: int=1,
                     device: str='cpu') -> np.ndarray:
    """Produce synthetic samples from a trained WGAN-GP generator."""
    gen = generator.to(device)
    gen.eval()
    with torch.no_grad():
        noise = torch.randn(n_samples, latent_dim, device=device)
        samples = gen(noise)
    return samples.cpu().numpy()
