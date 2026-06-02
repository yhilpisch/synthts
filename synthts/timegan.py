"""TimeGAN model for synthetic financial return sequences."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import default_rng


class TimeGANEmbedder(nn.Module):
    """Embed observed sequences into latent temporal states."""

    def __init__(self, input_dim: int=1, hidden_units: int=64,
                 latent_dim: int=24, num_layers: int=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.latent_dim = latent_dim
        self.rnn = nn.GRU(input_dim, hidden_units,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return torch.tanh(self.fc(out))


class TimeGANRecovery(nn.Module):
    """Recover observations from latent temporal states."""

    def __init__(self, latent_dim: int=24, hidden_units: int=64,
                 output_dim: int=1, num_layers: int=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.rnn = nn.GRU(latent_dim, hidden_units,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h)
        return self.fc(out)


class TimeGANGenerator(nn.Module):
    """Generate latent temporal states from per-step noise."""

    def __init__(self, noise_dim: int=8, hidden_units: int=64,
                 latent_dim: int=24, num_layers: int=1):
        super().__init__()
        self.noise_dim = noise_dim
        self.hidden_units = hidden_units
        self.latent_dim = latent_dim
        self.rnn = nn.GRU(noise_dim, hidden_units,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(z)
        return torch.tanh(self.fc(out))


class TimeGANSupervisor(nn.Module):
    """Learn one-step latent dynamics."""

    def __init__(self, latent_dim: int=24, hidden_units: int=64,
                 num_layers: int=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.rnn = nn.GRU(latent_dim, hidden_units,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, latent_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h)
        return torch.tanh(self.fc(out))


class TimeGANDiscriminator(nn.Module):
    """Latent sequence discriminator returning logits."""

    def __init__(self, latent_dim: int=24, hidden_units: int=64,
                 num_layers: int=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.rnn = nn.GRU(latent_dim, hidden_units,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h)
        pooled = out.mean(dim=1)
        return self.fc(pooled)


def _make_windows(data_scaled: np.ndarray, seq_len: int,
                  device: str) -> torch.Tensor:
    """Create overlapping training windows."""
    n_windows = len(data_scaled) - seq_len + 1
    if n_windows <= 0:
        raise ValueError('seq_len must not exceed the number of returns')
    data_t = torch.tensor(data_scaled, dtype=torch.float32,
                          device=device)
    return data_t.unfold(0, seq_len, 1).permute(0, 2, 1)


def _sample_noise(batch_size: int, seq_len: int, noise_dim: int,
                  device: str) -> torch.Tensor:
    """Sample per-step TimeGAN noise."""
    return torch.randn(batch_size, seq_len, noise_dim, device=device)


def _supervised_loss(h: torch.Tensor,
                     h_hat: torch.Tensor) -> torch.Tensor:
    """One-step latent prediction loss."""
    return F.mse_loss(h[:, 1:, :], h_hat[:, :-1, :])


def _moment_loss(real: torch.Tensor,
                 fake: torch.Tensor) -> torch.Tensor:
    """Moment matching loss from the TimeGAN paper."""
    real_f = real.reshape(-1, real.shape[-1])
    fake_f = fake.reshape(-1, fake.shape[-1])
    mean_loss = torch.abs(real_f.mean(dim=0) - fake_f.mean(dim=0)).mean()
    real_std = real_f.var(dim=0, unbiased=False).sqrt()
    fake_std = fake_f.var(dim=0, unbiased=False).sqrt()
    std_loss = torch.abs(real_std - fake_std).mean()
    return mean_loss + std_loss


def _acf_torch(x: torch.Tensor, max_lag: int=10) -> torch.Tensor:
    """Average temporal ACF for a batch of sequences."""
    x = x - x.mean(dim=1, keepdim=True)
    var = x.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-6)
    values = []
    for lag in range(1, max_lag + 1):
        cov = (x[:, :-lag, :] * x[:, lag:, :]).mean(dim=1,
                                                     keepdim=True)
        values.append((cov / var).mean())
    return torch.stack(values)


def _stat_loss(real: torch.Tensor, fake: torch.Tensor,
               max_lag: int=10) -> torch.Tensor:
    """Differentiable tail and temporal calibration loss."""
    real_f = real.reshape(-1, real.shape[-1])
    fake_f = fake.reshape(-1, fake.shape[-1])
    q = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                     device=fake.device)
    quant_loss = (torch.quantile(fake_f, q, dim=0)
                  - torch.quantile(real_f, q, dim=0)).pow(2).mean()
    acf_loss = (_acf_torch(fake, max_lag=max_lag)
                - _acf_torch(real, max_lag=max_lag)).pow(2).mean()
    acf2_loss = (_acf_torch(fake.pow(2), max_lag=max_lag)
                 - _acf_torch(real.pow(2), max_lag=max_lag)
                 ).pow(2).mean()
    return quant_loss + acf_loss + acf2_loss


def train_timegan(embedder, recovery, generator, supervisor,
                  discriminator, data_scaled,
                  seq_len: int=63, embedding_epochs: int=500,
                  supervisor_epochs: int=500, joint_epochs: int=1000,
                  batch_size: int=128, noise_dim: int=8,
                  lr: float=0.001, gamma: float=1.0,
                  recon_weight: float=10.0,
                  supervised_weight: float=100.0,
                  moment_weight: float=100.0,
                  stat_weight: float=10.0,
                  stat_max_lag: int=10,
                  seed: int=100, log_interval: int=500,
                  device: str='cpu') -> dict:
    """Train TimeGAN with embedding, supervisor, and joint phases."""
    rng = default_rng(seed)
    torch.manual_seed(seed)

    e = embedder.to(device)
    r = recovery.to(device)
    g = generator.to(device)
    s = supervisor.to(device)
    d = discriminator.to(device)
    windows = _make_windows(data_scaled, seq_len, device)
    n_windows = len(windows)

    opt_er = torch.optim.Adam(
        list(e.parameters()) + list(r.parameters()), lr=lr)
    opt_s = torch.optim.Adam(s.parameters(), lr=lr)
    opt_gs = torch.optim.Adam(
        list(g.parameters()) + list(s.parameters()), lr=lr)
    opt_d = torch.optim.Adam(d.parameters(), lr=lr)
    opt_e = torch.optim.Adam(
        list(e.parameters()) + list(r.parameters()), lr=lr)

    bce = nn.BCEWithLogitsLoss()
    history = {
        'recon_loss': [], 'supervised_loss': [],
        'g_loss': [], 'd_loss': [], 'moment_loss': [],
        'stat_loss': [],
    }

    for epoch in range(embedding_epochs):
        idx = rng.integers(0, n_windows, batch_size)
        x = windows[idx]
        opt_er.zero_grad()
        x_tilde = r(e(x))
        loss = F.mse_loss(x_tilde, x)
        loss.backward()
        opt_er.step()
        history['recon_loss'].append(loss.item())
        if epoch % log_interval == 0:
            print(f'Embedding {epoch:5d} | R: {loss.item():.4f}')

    for epoch in range(supervisor_epochs):
        idx = rng.integers(0, n_windows, batch_size)
        x = windows[idx]
        opt_s.zero_grad()
        h = e(x).detach()
        loss = _supervised_loss(h, s(h))
        loss.backward()
        opt_s.step()
        history['supervised_loss'].append(loss.item())
        if epoch % log_interval == 0:
            print(f'Supervisor {epoch:4d} | S: {loss.item():.4f}')

    for epoch in range(joint_epochs):
        idx = rng.integers(0, n_windows, batch_size)
        x = windows[idx]
        z = _sample_noise(batch_size, seq_len, noise_dim, device)

        opt_d.zero_grad()
        h = e(x).detach()
        e_hat = g(z).detach()
        h_hat = s(e_hat).detach()
        y_real = d(h)
        y_fake = d(h_hat)
        y_fake_e = d(e_hat)
        d_loss = (
            bce(y_real, torch.ones_like(y_real))
            + bce(y_fake, torch.zeros_like(y_fake))
            + gamma * bce(y_fake_e, torch.zeros_like(y_fake_e))
        )
        d_loss.backward()
        opt_d.step()

        z = _sample_noise(batch_size, seq_len, noise_dim, device)
        opt_gs.zero_grad()
        e_hat = g(z)
        h_hat = s(e_hat)
        x_hat = r(h_hat)
        h_real = e(x)
        y_fake = d(h_hat)
        y_fake_e = d(e_hat)
        g_adv = bce(y_fake, torch.ones_like(y_fake))
        g_adv = g_adv + gamma * bce(y_fake_e, torch.ones_like(y_fake_e))
        g_sup = _supervised_loss(h_real.detach(), s(h_real.detach()))
        g_mom = _moment_loss(x, x_hat)
        g_stat = _stat_loss(x, x_hat, max_lag=stat_max_lag)
        g_loss = (g_adv + supervised_weight * g_sup
                  + moment_weight * g_mom + stat_weight * g_stat)
        g_loss.backward()
        opt_gs.step()

        opt_e.zero_grad()
        h_real = e(x)
        x_tilde = r(h_real)
        e_recon = F.mse_loss(x_tilde, x)
        e_sup = _supervised_loss(h_real, s(h_real))
        e_loss = recon_weight * e_recon + supervised_weight * e_sup
        e_loss.backward()
        opt_e.step()

        history['g_loss'].append(g_loss.item())
        history['d_loss'].append(d_loss.item())
        history['moment_loss'].append(g_mom.item())
        history['stat_loss'].append(g_stat.item())
        history['recon_loss'].append(e_recon.item())
        history['supervised_loss'].append(g_sup.item())
        if epoch % log_interval == 0:
            print(f'Joint {epoch:8d} | D: {d_loss.item():.4f} '
                  f'G: {g_loss.item():.4f} M: {g_mom.item():.4f}')

    return history


def generate_timegan_sequences(generator, supervisor, recovery,
                               n_sequences: int, seq_len: int=63,
                               noise_dim: int=8, seed: int=None,
                               device: str='cpu') -> np.ndarray:
    """Generate synthetic sequences from trained TimeGAN components."""
    if seed is not None:
        torch.manual_seed(seed)
    g = generator.to(device)
    s = supervisor.to(device)
    r = recovery.to(device)
    g.eval()
    s.eval()
    r.eval()
    with torch.no_grad():
        z = _sample_noise(n_sequences, seq_len, noise_dim, device)
        h_hat = s(g(z))
        seqs = r(h_hat)
    return seqs.cpu().numpy()
