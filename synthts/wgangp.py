"""WGAN-GP: Critic, Generator, gradient penalty, and training loop."""

import numpy as np
import torch
import torch.nn as nn
from numpy.random import default_rng


class WGANGenerator(nn.Module):
    """Dense generator for WGAN-GP with BatchNorm stabilisation."""

    def __init__(self, latent_dim: int=8, hidden_units: int=128,
                 output_dim: int=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Critic(nn.Module):
    """Wasserstein critic with spectral normalisation.

    Uses spectral norm on each linear layer (Miyato et al., 2018)
    to enforce Lipschitz continuity, which is more stable than
    gradient penalty for low-dimensional problems.
    """

    def __init__(self, input_dim: int=1, hidden_units: int=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, hidden_units)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(
                nn.Linear(hidden_units, hidden_units)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(
                nn.Linear(hidden_units, hidden_units)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(hidden_units, 1)),
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
                 epochs: int=10001, batch_size: int=64,
                 n_critic: int=1, lr_g: float=0.0002,
                 lr_c: float=0.0002, latent_dim: int=8,
                 lambda_gp: float=10.0, seed: int=100,
                 log_interval: int=2000, device: str='cpu'):
    """Train a WGAN-GP on standardised 1-D data.

    Uses spectral normalisation in the critic and BCE loss
    (stable configuration for low-dimensional data). When
    ``n_critic > 1``, gradient-penalty steps are used instead
    of spectral normalisation + BCE.

    Parameters
    ----------
    generator : WGANGenerator
    critic : Critic
    data_scaled : np.ndarray
        Standardised real data, shape (n, 1).
    epochs : int
    batch_size : int
    n_critic : int
        When 1 (default), uses spectral-norm + BCE training.
        When > 1, uses Wasserstein + gradient-penalty training.
    lr_g : float
    lr_c : float
    latent_dim : int
    lambda_gp : float
        Only used when n_critic > 1.
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
    crit = critic.to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g,
                             betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(crit.parameters(), lr=lr_c,
                             betas=(0.5, 0.9))

    n = len(data_scaled)
    data_t = torch.tensor(data_scaled, dtype=torch.float32,
                          device=device)

    history = {'g_loss': [], 'd_loss_real': [], 'd_loss_fake': []}

    if n_critic == 1:
        # Spectral-norm + BCE mode (stable for 1-D data)
        bce = nn.BCELoss()
        ones = torch.ones(batch_size, 1, device=device)
        zeros = torch.zeros(batch_size, 1, device=device)

        for epoch in range(epochs):
            # --- train critic ---
            opt_c.zero_grad()
            idx = rng.integers(0, n, batch_size)
            real = data_t[idx]
            d_real = torch.sigmoid(crit(real))
            loss_real = bce(d_real, ones)
            loss_real.backward()

            noise = torch.randn(batch_size, latent_dim,
                                device=device)
            with torch.no_grad():
                fake = gen(noise)
            d_fake = torch.sigmoid(crit(fake.detach()))
            loss_fake = bce(d_fake, zeros)
            loss_fake.backward()
            opt_c.step()

            # --- train generator ---
            gen.train()
            opt_g.zero_grad()
            noise = torch.randn(batch_size, latent_dim,
                                device=device)
            fake = gen(noise)
            d_fake = torch.sigmoid(crit(fake))
            loss_g = bce(d_fake, ones)
            loss_g.backward()
            opt_g.step()

            if epoch % log_interval == 0:
                print(f'Epoch {epoch:5d} | '
                      f'D_real: {loss_real.item():.4f} '
                      f'D_fake: {loss_fake.item():.4f} '
                      f'G: {loss_g.item():.4f}')

    else:
        # Wasserstein + gradient-penalty mode
        for epoch in range(epochs):
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
                gp = _gradient_penalty(crit, real, fake, device,
                                      lambda_gp)
                loss_c = (score_fake.mean() - score_real.mean()
                          + gp)
                loss_c.backward()
                opt_c.step()

            gen.train()
            opt_g.zero_grad()
            noise = torch.randn(batch_size, latent_dim,
                                device=device)
            fake = gen(noise)
            score_fake = crit(fake)
            loss_g = -score_fake.mean()
            loss_g.backward()
            opt_g.step()

            if epoch % log_interval == 0:
                w_dist = (score_real.mean()
                          - score_fake.mean()).item()
                print(f'Epoch {epoch:5d} | '
                      f'C: {loss_c.item():.4f} '
                      f'G: {loss_g.item():.4f} '
                      f'W: {w_dist:.4f}')

    return history


def generate_samples(generator, n_samples: int, latent_dim: int=8,
                     device: str='cpu') -> np.ndarray:
    """Produce synthetic samples from a trained generator."""
    gen = generator.to(device)
    gen.eval()
    with torch.no_grad():
        noise = torch.randn(n_samples, latent_dim, device=device)
        samples = gen(noise)
    return samples.cpu().numpy()
