"""Tests for synthts.vanilla training loop."""

import numpy as np
import torch
import pytest

from synthts.vanilla import Generator, Discriminator, train_gan, generate_samples


@pytest.fixture()
def simple_data():
    """Standardised cubic-function data, shape (n, 1)."""
    from synthts.data import ReturnScaler
    x = np.linspace(-2, 2, 200)
    y = x ** 3
    scaler = ReturnScaler()
    return scaler.fit_transform(y), scaler


class TestTrainGan:
    def test_runs_without_error(self, simple_data):
        data_scaled, _ = simple_data
        g = Generator(latent_dim=1, hidden_units=16, output_dim=1)
        d = Discriminator(input_dim=1, hidden_units=16)
        history = train_gan(g, d, data_scaled,
                            epochs=10, batch_size=8,
                            seed=42, log_interval=100)
        assert history is not None

    def test_generator_weights_change(self, simple_data):
        data_scaled, _ = simple_data
        g = Generator(latent_dim=1, hidden_units=16, output_dim=1)
        d = Discriminator(input_dim=1, hidden_units=16)
        w0 = g.net[0].weight.data.clone()
        train_gan(g, d, data_scaled,
                  epochs=10, batch_size=8, seed=42,
                  log_interval=100)
        w1 = g.net[0].weight.data
        assert not torch.equal(w0, w1)

    def test_discriminator_weights_change(self, simple_data):
        data_scaled, _ = simple_data
        g = Generator(latent_dim=1, hidden_units=16, output_dim=1)
        d = Discriminator(input_dim=1, hidden_units=16)
        w0 = d.net[0].weight.data.clone()
        train_gan(g, d, data_scaled,
                  epochs=10, batch_size=8, seed=42,
                  log_interval=100)
        w1 = d.net[0].weight.data
        assert not torch.equal(w0, w1)

    def test_reproducibility_with_same_seed(self, simple_data):
        data_scaled, _ = simple_data
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            g = Generator(latent_dim=1, hidden_units=16, output_dim=1)
            d = Discriminator(input_dim=1, hidden_units=16)
            train_gan(g, d, data_scaled,
                      epochs=5, batch_size=8, seed=42,
                      log_interval=100)
            results.append(g.net[0].weight.data.clone())
        assert torch.equal(results[0], results[1])


class TestGenerateSamples:
    def test_output_shape(self):
        g = Generator(latent_dim=1, hidden_units=16, output_dim=1)
        samples = generate_samples(g, n_samples=50, latent_dim=1)
        assert samples.shape == (50, 1)

    def test_output_is_numpy(self):
        g = Generator(latent_dim=1, hidden_units=16, output_dim=1)
        samples = generate_samples(g, n_samples=10, latent_dim=1)
        assert isinstance(samples, np.ndarray)

    def test_custom_latent_dim(self):
        g = Generator(latent_dim=3, hidden_units=16, output_dim=2)
        samples = generate_samples(g, n_samples=20, latent_dim=3)
        assert samples.shape == (20, 2)
