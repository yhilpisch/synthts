"""Sequential GAN: LSTM generator with per-step noise and CNN discriminator."""

import numpy as np
import torch
import torch.nn as nn
from numpy.random import default_rng

from synthts._validation import (
    backward, dimensions, finite_nonnegative, integer, step,
    temporal_lag, training_settings,
)


class SeqGenerator(nn.Module):
    """LSTM generator with global latent and per-step noise.

    Takes a global latent vector and generates a sequence of
    returns by feeding the latent (repeated) concatenated with
    independent per-step noise at each time step.  The per-step
    noise gives the LSTM variation within a sequence, which is
    essential for producing realistic, non-monotonic returns.
    """

    def __init__(self, latent_dim: int=4, noise_dim: int=2,
                 hidden_units: int=48, output_dim: int=1,
                 seq_len: int=63, num_layers: int=1):
        dimensions(
            latent_dim=latent_dim,
            noise_dim=noise_dim,
            hidden_units=hidden_units,
            output_dim=output_dim,
            seq_len=seq_len,
            num_layers=num_layers,
        )
        super().__init__()
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.hidden_units = hidden_units
        self.seq_len = seq_len
        self.lstm = nn.LSTM(latent_dim + noise_dim, hidden_units,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate a batch of return sequences.

        Parameters
        ----------
        z : torch.Tensor
            Global noise vector, shape (batch, latent_dim).

        Returns
        -------
        torch.Tensor
            Return sequences, shape (batch, seq_len, output_dim).
        """
        batch = z.shape[0]
        # repeat global latent across time steps
        global_vec = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        # sample independent noise at each step
        step_noise = torch.randn(
            batch, self.seq_len, self.noise_dim, device=z.device)
        inp = torch.cat([global_vec, step_noise], dim=2)
        out, _ = self.lstm(inp)
        out = self.fc(out)
        return out


class SeqDiscriminator(nn.Module):
    """1-D CNN discriminator with spectral normalisation.

    Two SN-Conv1d layers with LeakyReLU, adaptive avg pooling,
    and an SN-linear + sigmoid head.  The CNN is faster than an
    LSTM discriminator and captures local temporal patterns
    without becoming too powerful for the generator.
    """

    def __init__(self, input_dim: int=1, hidden_units: int=32):
        dimensions(
            input_dim=input_dim,
            hidden_units=hidden_units,
        )
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(input_dim, hidden_units, 3, padding=1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(
                nn.Conv1d(hidden_units, hidden_units, 3, padding=1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(hidden_units, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of sequences.

        Parameters
        ----------
        x : torch.Tensor
            Return sequences, shape (batch, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Probabilities, shape (batch, 1).
        """
        # conv1d expects (batch, channels, length)
        return self.net(x.permute(0, 2, 1))


def _acf_torch(x: torch.Tensor, max_lag: int=5) -> torch.Tensor:
    """Average temporal ACF for a batch of sequences."""
    temporal_lag(x.shape[1], max_lag)
    x = x - x.mean(dim=1, keepdim=True)
    var = x.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-6)
    values = []
    for lag in range(1, max_lag + 1):
        cov = (x[:, :-lag, :] * x[:, lag:, :]).mean(dim=1,
                                                     keepdim=True)
        values.append((cov / var).mean())
    return torch.stack(values)


def _stat_loss(real_seq: torch.Tensor, fake_seq: torch.Tensor,
               max_lag: int=5) -> torch.Tensor:
    """Differentiable marginal and temporal calibration loss."""
    real = real_seq.reshape(-1, real_seq.shape[-1])
    fake = fake_seq.reshape(-1, fake_seq.shape[-1])
    q = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                     device=fake_seq.device)

    mean_loss = (fake.mean(dim=0) - real.mean(dim=0)).pow(2).mean()
    std_loss = (fake.std(dim=0) - real.std(dim=0)).pow(2).mean()
    quant_loss = (torch.quantile(fake, q, dim=0)
                  - torch.quantile(real, q, dim=0)).pow(2).mean()
    acf_loss = (_acf_torch(fake_seq, max_lag=max_lag)
                - _acf_torch(real_seq, max_lag=max_lag)).pow(2).mean()
    acf2_loss = (_acf_torch(fake_seq.pow(2), max_lag=max_lag)
                 - _acf_torch(real_seq.pow(2), max_lag=max_lag)
                 ).pow(2).mean()
    return mean_loss + std_loss + quant_loss + acf_loss + acf2_loss


def train_seq_gan(generator, discriminator, data_scaled,
                  seq_len: int=63, epochs: int=5001,
                  batch_size: int=64, lr_d: float=0.0001,
                  lr_g: float=0.0002, latent_dim: int=4,
                  label_smooth: float=0.1, n_gen_steps: int=2,
                  stat_weight: float=5.0, stat_max_lag: int=5,
                  seed: int=100, log_interval: int=1000,
                  device: str='cpu'):
    """Train a sequential GAN on standardised 1-D return data.

    Real data is sliced into overlapping windows of length
    `seq_len`.  The training uses one-sided label smoothing
    (real labels set to ``1 - label_smooth``) and multiple
    generator steps per discriminator step to keep the
    generator competitive.

    Parameters
    ----------
    generator : SeqGenerator
    discriminator : SeqDiscriminator
    data_scaled : np.ndarray
        Standardised real data, shape (n, 1).
    seq_len : int
    epochs : int
    batch_size : int
    lr_d : float
    lr_g : float
    latent_dim : int
    label_smooth : float
        One-sided smoothing for real labels.
    n_gen_steps : int
        Generator optimisation steps per discriminator step.
    stat_weight : float
        Weight for auxiliary marginal and temporal statistic matching.
    stat_max_lag : int
        Maximum lag for auxiliary ACF matching.
    seed : int
    log_interval : int
    device : str

    Returns
    -------
    dict
        Keys 'g_loss', 'd_loss_real', 'd_loss_fake'.
    """
    data_scaled = training_settings(
        data_scaled, batch_size, log_interval, epochs=epochs)
    integer('latent_dim', latent_dim)
    integer('seq_len', seq_len)
    integer('n_gen_steps', n_gen_steps)
    finite_nonnegative(lr_g=lr_g, lr_d=lr_d, stat_weight=stat_weight,
                       label_smooth=label_smooth)
    if label_smooth >= 1:
        raise ValueError('label_smooth must be smaller than 1')
    if generator.seq_len != seq_len or generator.latent_dim != latent_dim:
        raise ValueError('generator dimensions must match training settings')
    if stat_weight:
        temporal_lag(seq_len, stat_max_lag)
    rng = default_rng(seed)
    torch.manual_seed(seed)

    gen = generator.to(device)
    disc = discriminator.to(device)

    bce = nn.BCELoss()
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d,
                             betas=(0.5, 0.9))
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g,
                             betas=(0.5, 0.9))

    n = len(data_scaled)
    data_t = torch.tensor(data_scaled, dtype=torch.float32,
                          device=device)
    n_windows = n - seq_len + 1
    if n_windows <= 0:
        raise ValueError('seq_len must not exceed the number of returns')
    windows = data_t.unfold(0, seq_len, 1).permute(0, 2, 1)

    smooth_ones = torch.full((batch_size, 1), 1.0 - label_smooth,
                             device=device)
    zeros = torch.zeros(batch_size, 1, device=device)

    history = {'g_loss': [], 'd_loss_real': [], 'd_loss_fake': []}

    for epoch in range(epochs):
        # --- train discriminator ---
        opt_d.zero_grad()
        idx = rng.integers(0, n_windows, batch_size)
        real_seq = windows[idx]
        d_real = disc(real_seq)
        loss_real = bce(d_real, smooth_ones)
        backward(loss_real)

        noise = torch.randn(batch_size, latent_dim, device=device)
        with torch.no_grad():
            fake_seq = gen(noise)
        d_fake = disc(fake_seq.detach())
        loss_fake = bce(d_fake, zeros)
        backward(loss_fake)
        step(opt_d)

        # --- train generator (multiple steps) ---
        for _ in range(n_gen_steps):
            gen.train()
            opt_g.zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_seq = gen(noise)
            d_fake = disc(fake_seq)
            loss_g = bce(d_fake, torch.ones(batch_size, 1,
                                            device=device))
            if stat_weight:
                idx_g = rng.integers(0, n_windows, batch_size)
                stat = _stat_loss(windows[idx_g], fake_seq,
                                  max_lag=stat_max_lag)
                loss_g = loss_g + stat_weight * stat
            backward(loss_g)
            step(opt_g)

        history['g_loss'].append(loss_g.item())
        history['d_loss_real'].append(loss_real.item())
        history['d_loss_fake'].append(loss_fake.item())

        if epoch % log_interval == 0:
            print(f'Epoch {epoch:5d} | '
                  f'D_real: {loss_real.item():.4f} '
                  f'D_fake: {loss_fake.item():.4f} '
                  f'G: {loss_g.item():.4f}')

    return history


def generate_sequences(generator, n_sequences: int,
                       latent_dim: int=4,
                       seed: int=None,
                       device: str='cpu') -> np.ndarray:
    """Produce synthetic return sequences from a trained generator.

    Returns
    -------
    np.ndarray
        Shape (n_sequences, seq_len, output_dim).
    """
    if seed is not None:
        torch.manual_seed(seed)
    gen = generator.to(device)
    gen.eval()
    with torch.no_grad():
        noise = torch.randn(n_sequences, latent_dim, device=device)
        seqs = gen(noise)
    return seqs.cpu().numpy()
