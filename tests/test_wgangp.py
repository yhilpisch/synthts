"""Tests for synthts.wgangp module."""

import numpy as np
import torch
import torch.nn as nn
import pytest

from synthts.wgangp import (
    WGANGenerator, Critic, _gradient_penalty,
    train_wgangp, generate_samples,
)


class TestWGANGenerator:
    def test_output_shape_default(self):
        g = WGANGenerator()
        z = torch.randn(8, 8)
        out = g(z)
        assert out.shape == (8, 1)

    def test_output_shape_custom_dims(self):
        g = WGANGenerator(latent_dim=4, hidden_units=16, output_dim=3)
        z = torch.randn(10, 4)
        out = g(z)
        assert out.shape == (10, 3)

    def test_uses_batchnorm(self):
        g = WGANGenerator(hidden_units=32)
        layer_types = [type(m) for m in g.net]
        assert nn.BatchNorm1d in layer_types

    def test_three_hidden_layers(self):
        g = WGANGenerator()
        linear_layers = [m for m in g.net if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 4

    def test_gradient_flows(self):
        g = WGANGenerator(latent_dim=4, hidden_units=16, output_dim=1)
        z = torch.randn(4, 4)
        out = g(z)
        loss = out.sum()
        loss.backward()
        trainable = [p for p in g.parameters() if p.requires_grad]
        assert len(trainable) > 0


class TestCritic:
    def test_output_shape_default(self):
        c = Critic()
        x = torch.randn(8, 1)
        out = c(x)
        assert out.shape == (8, 1)

    def test_custom_input_dim(self):
        c = Critic(input_dim=5, hidden_units=16)
        x = torch.randn(4, 5)
        out = c(x)
        assert out.shape == (4, 1)

    def test_uses_spectral_norm(self):
        c = Critic(hidden_units=32)
        # check spectral norm hooks are registered
        sn_params = [name for name, _ in c.named_parameters()
                     if 'weight_orig' in name]
        assert len(sn_params) > 0

    def test_uses_leaky_relu(self):
        c = Critic(hidden_units=32)
        layer_types = [type(m) for m in c.net]
        assert nn.LeakyReLU in layer_types

    def test_three_hidden_layers(self):
        c = Critic()
        linear_layers = [m for m in c.net if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 4

    def test_gradient_flows(self):
        c = Critic(input_dim=1, hidden_units=16)
        x = torch.randn(4, 1)
        out = c(x)
        loss = out.sum()
        loss.backward()
        trainable = [p for p in c.parameters() if p.requires_grad]
        assert len(trainable) > 0


class TestGradientPenalty:
    def test_penalty_is_non_negative(self):
        c = Critic(input_dim=1, hidden_units=16)
        real = torch.randn(16, 1)
        fake = torch.randn(16, 1)
        gp = _gradient_penalty(c, real, fake, device='cpu',
                               lambda_gp=10.0)
        assert gp.item() >= 0.0


class TestTrainWganGP:
    @pytest.fixture()
    def simple_data(self):
        from synthts.data import ReturnScaler
        x = np.linspace(-2, 2, 200)
        y = x ** 3
        scaler = ReturnScaler()
        return scaler.fit_transform(y), scaler

    def test_runs_sn_bce_mode(self, simple_data):
        data_scaled, _ = simple_data
        g = WGANGenerator(latent_dim=2, hidden_units=16, output_dim=1)
        c = Critic(input_dim=1, hidden_units=16)
        history = train_wgangp(
            g, c, data_scaled,
            epochs=10, batch_size=8,
            n_critic=1, latent_dim=2, seed=42, log_interval=100)
        assert history is not None

    def test_runs_gp_mode(self, simple_data):
        data_scaled, _ = simple_data
        g = WGANGenerator(latent_dim=2, hidden_units=16, output_dim=1)
        c = Critic(input_dim=1, hidden_units=16)
        history = train_wgangp(
            g, c, data_scaled,
            epochs=10, batch_size=8,
            n_critic=5, latent_dim=2, seed=42, log_interval=100)
        assert history is not None

    def test_generator_weights_change(self, simple_data):
        data_scaled, _ = simple_data
        g = WGANGenerator(latent_dim=2, hidden_units=16, output_dim=1)
        c = Critic(input_dim=1, hidden_units=16)
        w0 = g.net[0].weight.data.clone()
        train_wgangp(g, c, data_scaled,
                     epochs=10, batch_size=8,
                     n_critic=1, latent_dim=2, seed=42,
                     log_interval=100)
        w1 = g.net[0].weight.data
        assert not torch.equal(w0, w1)


class TestWganGenerateSamples:
    def test_output_shape(self):
        g = WGANGenerator(latent_dim=4, hidden_units=16, output_dim=1)
        samples = generate_samples(g, n_samples=50, latent_dim=4)
        assert samples.shape == (50, 1)

    def test_output_is_numpy(self):
        g = WGANGenerator(latent_dim=4, hidden_units=16, output_dim=1)
        samples = generate_samples(g, n_samples=10, latent_dim=4)
        assert isinstance(samples, np.ndarray)
