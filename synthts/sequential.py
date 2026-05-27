"""Sequential GAN: LSTM generator with per-step noise and CNN discriminator."""

import numpy as np
import torch
import torch.nn as nn
from numpy.random import default_rng


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


def train_seq_gan(generator, discriminator, data_scaled,
                  seq_len: int=63, epochs: int=5001,
                  batch_size: int=64, lr_d: float=0.0001,
                  lr_g: float=0.0002, latent_dim: int=4,
                  label_smooth: float=0.1, n_gen_steps: int=2,
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
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d,
                             betas=(0.5, 0.9))
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g,
                             betas=(0.5, 0.9))

    n = len(data_scaled)
    data_t = torch.tensor(data_scaled, dtype=torch.float32,
                          device=device)
    n_windows = n - seq_len + 1
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
        loss_real.backward()

        noise = torch.randn(batch_size, latent_dim, device=device)
        with torch.no_grad():
            fake_seq = gen(noise)
        d_fake = disc(fake_seq.detach())
        loss_fake = bce(d_fake, zeros)
        loss_fake.backward()
        opt_d.step()

        # --- train generator (multiple steps) ---
        for _ in range(n_gen_steps):
            gen.train()
            opt_g.zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_seq = gen(noise)
            d_fake = disc(fake_seq)
            loss_g = bce(d_fake, torch.ones(batch_size, 1,
                                            device=device))
            loss_g.backward()
            opt_g.step()

        if epoch % log_interval == 0:
            print(f'Epoch {epoch:5d} | '
                  f'D_real: {loss_real.item():.4f} '
                  f'D_fake: {loss_fake.item():.4f} '
                  f'G: {loss_g.item():.4f}')

    return history


def generate_sequences(generator, n_sequences: int,
                       latent_dim: int=4,
                       device: str='cpu') -> np.ndarray:
    """Produce synthetic return sequences from a trained generator.

    Returns
    -------
    np.ndarray
        Shape (n_sequences, seq_len, output_dim).
    """
    gen = generator.to(device)
    gen.eval()
    with torch.no_grad():
        noise = torch.randn(n_sequences, latent_dim, device=device)
        seqs = gen(noise)
    return seqs.cpu().numpy()
