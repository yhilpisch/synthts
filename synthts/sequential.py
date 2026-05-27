"""Sequential GAN: LSTM generator and discriminator for return sequences."""

import numpy as np
import torch
import torch.nn as nn
from numpy.random import default_rng


class SeqGenerator(nn.Module):
    """LSTM generator: single noise vector -> return sequence.

    Takes a latent vector and autoregressively generates a sequence
    of returns of length `seq_len`.
    """

    def __init__(self, latent_dim: int=1, hidden_units: int=64,
                 output_dim: int=1, seq_len: int=252,
                 num_layers: int=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.seq_len = seq_len
        self.lstm = nn.LSTM(latent_dim, hidden_units,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate a batch of return sequences.

        Parameters
        ----------
        z : torch.Tensor
            Noise vector, shape (batch, latent_dim).

        Returns
        -------
        torch.Tensor
            Return sequences, shape (batch, seq_len, output_dim).
        """
        batch = z.shape[0]
        # expand latent to full sequence as input signal
        inp = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(inp)
        out = self.fc(out)
        return out


class SeqDiscriminator(nn.Module):
    """LSTM discriminator: return sequence -> probability (real vs fake)."""

    def __init__(self, input_dim: int=1, hidden_units: int=64,
                 num_layers: int=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_units,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_units, 1),
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
        out, _ = self.lstm(x)
        # use last time-step hidden state
        out = self.fc(out[:, -1, :])
        return out


def train_seq_gan(generator, discriminator, data_scaled,
                  seq_len: int=252, epochs: int=5001,
                  batch_size: int=32, lr_d: float=0.001,
                  lr_g: float=0.0001, latent_dim: int=1,
                  seed: int=100, log_interval: int=1000,
                  device: str='cpu'):
    """Train a sequential GAN on standardised 1-D return data.

    Real data is sliced into overlapping windows of length `seq_len`.

    Parameters
    ----------
    generator : SeqGenerator
    discriminator : SeqDiscriminator
    data_scaled : np.ndarray
        Standardised real data, shape (n, 1).
    seq_len : int
        Window length for training sequences.
    epochs : int
    batch_size : int
    lr_d : float
    lr_g : float
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
    # build overlapping windows
    n_windows = n - seq_len + 1
    windows = data_t.unfold(0, seq_len, 1).permute(0, 2, 1)
    # shape: (n_windows, seq_len, 1)

    history = {'g_loss': [], 'd_loss_real': [], 'd_loss_fake': []}

    for epoch in range(epochs):
        # --- train discriminator ---
        opt_d.zero_grad()
        idx = rng.integers(0, n_windows, batch_size)
        real_seq = windows[idx]
        d_real = disc(real_seq)
        loss_real = bce(d_real, ones)
        loss_real.backward()

        noise = torch.randn(batch_size, latent_dim, device=device)
        with torch.no_grad():
            fake_seq = gen(noise)
        d_fake = disc(fake_seq.detach())
        loss_fake = bce(d_fake, zeros)
        loss_fake.backward()
        opt_d.step()

        # --- train generator ---
        opt_g.zero_grad()
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_seq = gen(noise)
        d_fake = disc(fake_seq)
        loss_g = bce(d_fake, ones)
        loss_g.backward()
        opt_g.step()

        if epoch % log_interval == 0:
            print(f'Epoch {epoch:5d} | '
                  f'D_real: {loss_real.item():.4f} '
                  f'D_fake: {loss_fake.item():.4f} '
                  f'G: {loss_g.item():.4f}')

    return history


def generate_sequences(generator, n_sequences: int,
                       latent_dim: int=1,
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
