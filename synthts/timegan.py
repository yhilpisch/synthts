"""TimeGAN model for synthetic financial return sequences."""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import default_rng


class TimeGANEmbedder(nn.Module):
    """Embed observed sequences into latent temporal states."""

    def __init__(self, input_dim: int=1, hidden_units: int=64,
                 latent_dim: int=24, num_layers: int=1,
                 latent_activation: str='tanh'):
        super().__init__()
        if latent_activation not in ('tanh', 'layernorm'):
            raise ValueError(
                "latent_activation must be 'tanh' or 'layernorm'")
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.latent_dim = latent_dim
        self.latent_activation = latent_activation
        self.rnn = nn.GRU(input_dim, hidden_units,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, latent_dim)
        if latent_activation == 'layernorm':
            self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        h = self.fc(out)
        if self.latent_activation == 'tanh':
            return torch.tanh(h)
        return self.layer_norm(h)


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
                 latent_dim: int=24, num_layers: int=1,
                 latent_activation: str='tanh'):
        super().__init__()
        if latent_activation not in ('tanh', 'layernorm'):
            raise ValueError(
                "latent_activation must be 'tanh' or 'layernorm'")
        self.noise_dim = noise_dim
        self.hidden_units = hidden_units
        self.latent_dim = latent_dim
        self.latent_activation = latent_activation
        self.rnn = nn.GRU(noise_dim, hidden_units,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, latent_dim)
        if latent_activation == 'layernorm':
            self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(z)
        h = self.fc(out)
        if self.latent_activation == 'tanh':
            return torch.tanh(h)
        return self.layer_norm(h)


class TimeGANSupervisor(nn.Module):
    """Learn one-step latent dynamics."""

    def __init__(self, latent_dim: int=24, hidden_units: int=64,
                 num_layers: int=1,
                 latent_activation: str='tanh'):
        super().__init__()
        if latent_activation not in ('tanh', 'layernorm'):
            raise ValueError(
                "latent_activation must be 'tanh' or 'layernorm'")
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.latent_activation = latent_activation
        self.rnn = nn.GRU(latent_dim, hidden_units,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, latent_dim)
        if latent_activation == 'layernorm':
            self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h)
        y = self.fc(out)
        if self.latent_activation == 'tanh':
            return torch.tanh(y)
        return self.layer_norm(y)


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


class TimeGANObsDiscriminator(nn.Module):
    """Observation-space discriminator with spectral normalisation.

    Classifies real vs recovered return windows directly in
    observation space, providing adversarial feedback that
    penalises unrealistic cumulative-return shape and narrow
    path dispersion.
    """

    def __init__(self, input_dim: int=1,
                 hidden_channels: int=32,
                 num_layers: int=2):
        super().__init__()
        self.input_dim = input_dim
        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            out_ch = hidden_channels * (2 ** i)
            layers.append(nn.utils.spectral_norm(
                nn.Conv1d(in_ch, out_ch, 3, padding=1)))
            layers.append(nn.LeakyReLU(0.2))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        h = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        h = self.conv(h)  # (batch, out_ch, seq_len)
        h = h.mean(dim=2)  # (batch, out_ch)
        return self.fc(h)


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------

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


def _mean_loss(real: torch.Tensor,
               fake: torch.Tensor) -> torch.Tensor:
    """Mean matching loss."""
    real_f = real.reshape(-1, real.shape[-1])
    fake_f = fake.reshape(-1, fake.shape[-1])
    return torch.abs(
        real_f.mean(dim=0) - fake_f.mean(dim=0)).mean()


def _std_loss(real: torch.Tensor,
              fake: torch.Tensor) -> torch.Tensor:
    """Standard deviation matching loss."""
    real_f = real.reshape(-1, real.shape[-1])
    fake_f = fake.reshape(-1, fake.shape[-1])
    real_std = real_f.var(dim=0, unbiased=False).sqrt()
    fake_std = fake_f.var(dim=0, unbiased=False).sqrt()
    return torch.abs(real_std - fake_std).mean()


def _moment_loss(real: torch.Tensor,
                 fake: torch.Tensor) -> torch.Tensor:
    """Moment matching loss from the TimeGAN paper."""
    return _mean_loss(real, fake) + _std_loss(real, fake)


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


def _acf_loss(real: torch.Tensor, fake: torch.Tensor,
              max_lag: int=10) -> torch.Tensor:
    """Return ACF matching loss."""
    return (_acf_torch(fake, max_lag=max_lag)
            - _acf_torch(real, max_lag=max_lag)).pow(2).mean()


def _squared_acf_loss(real: torch.Tensor, fake: torch.Tensor,
                      max_lag: int=10) -> torch.Tensor:
    """Squared-return ACF matching loss (volatility clustering)."""
    return (_acf_torch(fake.pow(2), max_lag=max_lag)
            - _acf_torch(real.pow(2), max_lag=max_lag)
            ).pow(2).mean()


def _abs_acf_loss(real: torch.Tensor, fake: torch.Tensor,
                  max_lag: int=10) -> torch.Tensor:
    """Absolute-return ACF matching loss."""
    return (_acf_torch(fake.abs(), max_lag=max_lag)
            - _acf_torch(real.abs(), max_lag=max_lag)
            ).pow(2).mean()


def _central_quantile_loss(real: torch.Tensor,
                           fake: torch.Tensor) -> torch.Tensor:
    """Central quantile matching loss (0.25, 0.5, 0.75)."""
    real_f = real.reshape(-1, real.shape[-1])
    fake_f = fake.reshape(-1, fake.shape[-1])
    q = torch.tensor([0.25, 0.5, 0.75], device=fake.device)
    return (torch.quantile(fake_f, q, dim=0)
            - torch.quantile(real_f, q, dim=0)).pow(2).mean()


def _tail_quantile_loss(real: torch.Tensor,
                        fake: torch.Tensor) -> torch.Tensor:
    """Tail quantile matching loss (0.01, 0.05, 0.95, 0.99)."""
    real_f = real.reshape(-1, real.shape[-1])
    fake_f = fake.reshape(-1, fake.shape[-1])
    q = torch.tensor([0.01, 0.05, 0.95, 0.99], device=fake.device)
    return (torch.quantile(fake_f, q, dim=0)
            - torch.quantile(real_f, q, dim=0)).pow(2).mean()


def _terminal_cumulative_quantile_loss(
        real: torch.Tensor,
        fake: torch.Tensor) -> torch.Tensor:
    """Terminal cumulative-return quantile matching loss.

    Matches the distribution of the cumulative return at the end
    of each window, ensuring synthetic paths have the right
    fan width.
    """
    real_cum = real.cumsum(dim=1)[:, -1, :]
    fake_cum = fake.cumsum(dim=1)[:, -1, :]
    q = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                     device=fake.device)
    return (torch.quantile(fake_cum, q, dim=0)
            - torch.quantile(real_cum, q, dim=0)).pow(2).mean()


def _soft_drawdown_quantile_loss(
        real: torch.Tensor,
        fake: torch.Tensor) -> torch.Tensor:
    """Soft drawdown quantile matching loss.

    Uses the minimum cumulative return in each window as a
    differentiable proxy for maximum drawdown.  Matching
    quantiles of this proxy ensures synthetic paths reproduce
    deep drawdown tails.
    """
    real_min_cum = real.cumsum(dim=1).min(dim=1).values
    fake_min_cum = fake.cumsum(dim=1).min(dim=1).values
    q = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                     device=fake.device)
    return (torch.quantile(fake_min_cum, q, dim=0)
            - torch.quantile(real_min_cum, q, dim=0)).pow(2).mean()


def _stat_loss(real: torch.Tensor, fake: torch.Tensor,
               max_lag: int=10) -> torch.Tensor:
    """Differentiable tail and temporal calibration loss.

    Kept for backward compatibility; delegates to the modular
    components with the original unweighted sum.
    """
    return (_central_quantile_loss(real, fake)
            + _tail_quantile_loss(real, fake)
            + _acf_loss(real, fake, max_lag=max_lag)
            + _squared_acf_loss(real, fake, max_lag=max_lag))


def compute_financial_losses(
        real: torch.Tensor,
        fake: torch.Tensor,
        max_lag: int=10) -> dict[str, torch.Tensor]:
    """Compute all modular financial loss components.

    Returns a dictionary of named scalar losses, each
    independently weighted in the generator objective.
    """
    return {
        'mean': _mean_loss(real, fake),
        'std': _std_loss(real, fake),
        'central_quantile': _central_quantile_loss(real, fake),
        'tail_quantile': _tail_quantile_loss(real, fake),
        'acf': _acf_loss(real, fake, max_lag=max_lag),
        'abs_acf': _abs_acf_loss(real, fake, max_lag=max_lag),
        'squared_acf': _squared_acf_loss(real, fake, max_lag=max_lag),
        'terminal_quantile': _terminal_cumulative_quantile_loss(
            real, fake),
        'drawdown_quantile': _soft_drawdown_quantile_loss(
            real, fake),
    }


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def _snapshot_models(*modules: nn.Module) -> dict[str, dict]:
    """Deep-copy state dicts for checkpointing."""
    return {m.__class__.__name__: copy.deepcopy(m.state_dict())
            for m in modules}


def _restore_models(snapshot: dict[str, dict],
                    *modules: nn.Module) -> None:
    """Restore model state dicts from a checkpoint snapshot."""
    for m in modules:
        name = m.__class__.__name__
        if name in snapshot:
            m.load_state_dict(snapshot[name])


def train_timegan(
        embedder, recovery, generator, supervisor,
        discriminator, data_scaled,
        seq_len: int=63,
        embedding_epochs: int=500,
        supervisor_epochs: int=500,
        joint_epochs: int=1000,
        batch_size: int=128,
        noise_dim: int=8,
        lr: float=0.001,
        gamma: float=1.0,
        recon_weight: float=10.0,
        supervised_weight: float=100.0,
        moment_weight: float=100.0,
        stat_weight: float=10.0,
        stat_max_lag: int=10,
        # Modular financial loss weights
        mean_weight: float=1.0,
        std_weight: float=1.0,
        central_quantile_weight: float=1.0,
        tail_quantile_weight: float=2.0,
        acf_weight: float=1.0,
        abs_acf_weight: float=1.0,
        squared_acf_weight: float=2.0,
        terminal_quantile_weight: float=3.0,
        drawdown_quantile_weight: float=3.0,
        # Observation discriminator
        obs_discriminator: Optional[nn.Module]=None,
        obs_adv_weight: float=1.0,
        # Checkpoint scoring
        checkpoint_interval: int=0,
        # LR scheduling
        lr_schedule: str='none',
        seed: int=100,
        log_interval: int=500,
        device: str='cpu') -> dict:
    """Train TimeGAN with embedding, supervisor, and joint phases.

    Parameters
    ----------
    checkpoint_interval : int
        Evaluate validation score every N joint epochs and keep the
        best model.  0 disables checkpointing.
    lr_schedule : str
        Learning rate schedule for joint training.
        'none', 'cosine', or 'step'.
    obs_discriminator : nn.Module, optional
        Observation-space discriminator.  When provided, its
        adversarial signal is added to the generator and embedder
        objectives.
    """
    rng = default_rng(seed)
    torch.manual_seed(seed)

    e = embedder.to(device)
    r = recovery.to(device)
    g = generator.to(device)
    s = supervisor.to(device)
    d = discriminator.to(device)
    obs_d = (obs_discriminator.to(device)
             if obs_discriminator is not None else None)
    windows = _make_windows(data_scaled, seq_len, device)
    n_windows = len(windows)

    # Optimisers
    opt_er = torch.optim.Adam(
        list(e.parameters()) + list(r.parameters()), lr=lr)
    opt_s = torch.optim.Adam(s.parameters(), lr=lr)
    opt_gs = torch.optim.Adam(
        list(g.parameters()) + list(s.parameters()), lr=lr)
    opt_d = torch.optim.Adam(d.parameters(), lr=lr)
    opt_e = torch.optim.Adam(
        list(e.parameters()) + list(r.parameters()), lr=lr)
    if obs_d is not None:
        opt_obs_d = torch.optim.Adam(obs_d.parameters(), lr=lr)

    # LR schedulers for joint phase
    sched_gs = sched_d = sched_e = sched_obs_d = None
    if lr_schedule == 'cosine':
        sched_gs = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_gs, T_max=joint_epochs)
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_d, T_max=joint_epochs)
        sched_e = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_e, T_max=joint_epochs)
        if obs_d is not None:
            sched_obs_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_obs_d, T_max=joint_epochs)
    elif lr_schedule == 'step':
        step_size = max(1, joint_epochs // 3)
        sched_gs = torch.optim.lr_scheduler.StepLR(
            opt_gs, step_size=step_size, gamma=0.5)
        sched_d = torch.optim.lr_scheduler.StepLR(
            opt_d, step_size=step_size, gamma=0.5)
        sched_e = torch.optim.lr_scheduler.StepLR(
            opt_e, step_size=step_size, gamma=0.5)
        if obs_d is not None:
            sched_obs_d = torch.optim.lr_scheduler.StepLR(
                opt_obs_d, step_size=step_size, gamma=0.5)

    # Checkpoint scoring state
    best_snapshot = None
    best_val_score = float('inf')
    best_epoch = 0
    if checkpoint_interval > 0:
        val_rng = default_rng(seed + 999_999)
        val_idx = val_rng.integers(0, n_windows, batch_size)
        val_x = windows[val_idx]

    bce = nn.BCEWithLogitsLoss()
    history: dict[str, list] = {
        'recon_loss': [], 'supervised_loss': [],
        'g_loss': [], 'd_loss': [], 'moment_loss': [],
        'stat_loss': [],
    }
    # Component-level loss history
    for name in ('mean', 'std', 'central_quantile', 'tail_quantile',
                 'acf', 'abs_acf', 'squared_acf',
                 'terminal_quantile', 'drawdown_quantile'):
        history[f'{name}_loss'] = []
    if obs_d is not None:
        history['obs_d_loss'] = []
        history['obs_adv_loss'] = []

    # ------------------------------------------------------------------
    # Phase 1: Embedding pretraining
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Phase 2: Supervisor pretraining
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Phase 3: Joint adversarial training
    # ------------------------------------------------------------------
    for epoch in range(joint_epochs):
        idx = rng.integers(0, n_windows, batch_size)
        x = windows[idx]
        z = _sample_noise(batch_size, seq_len, noise_dim, device)

        # --- Discriminator update ---
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

        # --- Observation discriminator update ---
        if obs_d is not None:
            opt_obs_d.zero_grad()
            x_hat_det = r(s(g(z.detach()))).detach()
            obs_y_real = obs_d(x)
            obs_y_fake = obs_d(x_hat_det)
            obs_d_loss = (
                bce(obs_y_real, torch.ones_like(obs_y_real))
                + bce(obs_y_fake, torch.zeros_like(obs_y_fake))
            )
            obs_d_loss.backward()
            opt_obs_d.step()

        # --- Generator + Supervisor update ---
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

        # Modular financial losses
        losses = compute_financial_losses(x, x_hat, max_lag=stat_max_lag)
        g_mom = losses['mean'] + losses['std']
        g_stat = (central_quantile_weight * losses['central_quantile']
                  + tail_quantile_weight * losses['tail_quantile']
                  + acf_weight * losses['acf']
                  + abs_acf_weight * losses['abs_acf']
                  + squared_acf_weight * losses['squared_acf']
                  + terminal_quantile_weight * losses['terminal_quantile']
                  + drawdown_quantile_weight * losses['drawdown_quantile'])

        # Observation adversarial loss for generator
        g_obs_adv = torch.tensor(0.0, device=device)
        if obs_d is not None:
            obs_y_fake_g = obs_d(x_hat)
            g_obs_adv = bce(obs_y_fake_g, torch.ones_like(obs_y_fake_g))

        g_loss = (g_adv
                  + obs_adv_weight * g_obs_adv
                  + supervised_weight * g_sup
                  + moment_weight * g_mom
                  + stat_weight * g_stat)
        g_loss.backward()
        opt_gs.step()

        # --- Embedder update ---
        opt_e.zero_grad()
        h_real = e(x)
        x_tilde = r(h_real)
        e_recon = F.mse_loss(x_tilde, x)
        e_sup = _supervised_loss(h_real, s(h_real))
        e_loss = recon_weight * e_recon + supervised_weight * e_sup
        e_loss.backward()
        opt_e.step()

        # --- Record history ---
        history['g_loss'].append(g_loss.item())
        history['d_loss'].append(d_loss.item())
        history['moment_loss'].append(g_mom.item())
        history['stat_loss'].append(g_stat.item())
        history['recon_loss'].append(e_recon.item())
        history['supervised_loss'].append(g_sup.item())
        for name, val in losses.items():
            history[f'{name}_loss'].append(val.item())
        if obs_d is not None:
            history['obs_d_loss'].append(obs_d_loss.item())
            history['obs_adv_loss'].append(g_obs_adv.item())

        # --- LR scheduling ---
        if sched_gs is not None:
            sched_gs.step()
            sched_d.step()
            sched_e.step()
            if sched_obs_d is not None:
                sched_obs_d.step()

        # --- Checkpoint scoring ---
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            with torch.no_grad():
                e.eval(); r.eval(); g.eval(); s.eval()
                z_val = _sample_noise(batch_size, seq_len,
                                      noise_dim, device)
                x_hat_val = r(s(g(z_val)))
                val_mom = _moment_loss(val_x, x_hat_val)
                val_losses = compute_financial_losses(
                    val_x, x_hat_val, max_lag=stat_max_lag)
                val_score = (val_mom.item()
                             + val_losses['tail_quantile'].item()
                             + val_losses['squared_acf'].item()
                             + val_losses['terminal_quantile'].item()
                             + val_losses['drawdown_quantile'].item())
                if not np.isnan(val_score) and val_score < best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    best_snapshot = _snapshot_models(e, r, g, s, d)
                    if obs_d is not None:
                        best_snapshot.update(
                            _snapshot_models(obs_d))
                e.train(); r.train(); g.train(); s.train()
                history.setdefault('val_score', []).append(val_score)

        if epoch % log_interval == 0:
            parts = (f'Joint {epoch:8d} | D: {d_loss.item():.4f} '
                     f'G: {g_loss.item():.4f} M: {g_mom.item():.4f}')
            if obs_d is not None:
                parts += f' ObsD: {obs_d_loss.item():.4f}'
            print(parts)

    # Restore best checkpoint
    if checkpoint_interval > 0 and best_snapshot is not None:
        modules = [e, r, g, s, d]
        if obs_d is not None:
            modules.append(obs_d)
        _restore_models(best_snapshot, *modules)
        history['best_epoch'] = best_epoch
        history['best_val_score'] = best_val_score

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
