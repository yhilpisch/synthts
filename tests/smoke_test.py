"""End-to-end smoke test: train a tiny GAN and generate samples.

Run as a script:
    python -m tests.smoke_test
or via pytest:
    pytest tests/smoke_test.py
"""

import numpy as np
import pytest

from synthts.data import ReturnScaler
from synthts.vanilla import Generator, Discriminator, train_gan, generate_samples
from synthts.evaluate import ks_test, pass_rate


def test_smoke_cubic():
    """Train a minimal GAN on standardised x**3 data."""
    x = np.linspace(-2, 2, 100)
    y = x ** 3
    scaler = ReturnScaler()
    y_ = scaler.fit_transform(y)

    g = Generator(latent_dim=1, hidden_units=16, output_dim=1)
    d = Discriminator(input_dim=1, hidden_units=16)
    train_gan(g, d, y_.reshape(-1, 1),
              epochs=51, batch_size=16,
              seed=42, log_interval=100)

    samples = generate_samples(g, len(y), latent_dim=1)
    synth = scaler.inverse_transform(samples)
    assert synth.shape == (len(y),)
    assert not np.any(np.isnan(synth))


def test_smoke_random_returns():
    """Train a minimal GAN on synthetic random returns."""
    np.random.seed(7)
    rets = np.random.randn(200) * 0.01
    scaler = ReturnScaler()
    rets_ = scaler.fit_transform(rets)

    g = Generator(latent_dim=1, hidden_units=16, output_dim=1)
    d = Discriminator(input_dim=1, hidden_units=16)
    train_gan(g, d, rets_.reshape(-1, 1),
              epochs=51, batch_size=16,
              seed=42, log_interval=100)

    N = 10
    pvals = []
    for _ in range(N):
        samples = generate_samples(g, len(rets_), latent_dim=1)
        synth = scaler.inverse_transform(samples)
        pvals.append(ks_test(rets, synth)['pvalue'])
    pvals = np.array(pvals)
    # with so few epochs, we just check the pipeline completes
    assert len(pvals) == N
    assert (pvals >= 0).all() and (pvals <= 1).all()


if __name__ == '__main__':
    test_smoke_cubic()
    test_smoke_random_returns()
    print('All smoke tests passed.')
