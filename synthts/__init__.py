"""synthts — synthetic financial return series via GAN variants.

The package exposes four generative-model families plus their
training helpers and a shared evaluation toolkit:

- `synthts.vanilla` — baseline GAN.
- `synthts.wgangp`  — WGAN-GP in spectral-norm + BCE mode.
- `synthts.sequential` — LSTM/CNN sequential GAN.
- `synthts.timegan`  — TimeGAN with embedder, recovery, supervisor,
  generator, and (optional) observation-space discriminator.
- `synthts.evaluate` — marginal and path-level diagnostics.

Requires Python >= 3.10.
"""

from synthts._version import __version__
from synthts.vanilla import (
    GAN,
    Discriminator,
    Generator,
    generate_samples,
    train_gan,
)
from synthts.wgangp import (
    Critic,
    WGANGenerator,
    generate_samples as wgan_generate_samples,
    train_wgangp,
)
from synthts.sequential import (
    SeqDiscriminator,
    SeqGenerator,
    generate_sequences,
    train_seq_gan,
)
from synthts.timegan import (
    TimeGANDiscriminator,
    TimeGANEmbedder,
    TimeGANGenerator,
    TimeGANObsDiscriminator,
    TimeGANRecovery,
    TimeGANSupervisor,
    compute_financial_losses,
    generate_timegan_sequences,
    train_timegan,
)

__all__ = [
    "__version__",
    # vanilla
    "Generator",
    "Discriminator",
    "GAN",
    "train_gan",
    "generate_samples",
    # wgangp
    "WGANGenerator",
    "Critic",
    "train_wgangp",
    "wgan_generate_samples",
    # sequential
    "SeqGenerator",
    "SeqDiscriminator",
    "train_seq_gan",
    "generate_sequences",
    # timegan
    "TimeGANEmbedder",
    "TimeGANRecovery",
    "TimeGANGenerator",
    "TimeGANSupervisor",
    "TimeGANDiscriminator",
    "TimeGANObsDiscriminator",
    "train_timegan",
    "generate_timegan_sequences",
    "compute_financial_losses",
]
