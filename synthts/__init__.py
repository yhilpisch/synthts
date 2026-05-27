# Synthetic Financial Time Series — synthts package

from synthts.vanilla import (  # noqa: F401
    Generator, Discriminator, GAN,
    train_gan, generate_samples,
)
from synthts.wgangp import (  # noqa: F401
    WGANGenerator, Critic,
    train_wgangp, generate_samples as wgan_generate_samples,
)
