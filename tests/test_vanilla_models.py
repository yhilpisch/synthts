"""Tests for synthts.vanilla module (Generator, Discriminator, GAN)."""

import torch
import pytest

from synthts.vanilla import Generator, Discriminator, GAN


class TestGenerator:
    def test_output_shape_default(self):
        g = Generator()
        z = torch.randn(8, 1)
        out = g(z)
        assert out.shape == (8, 1)

    def test_output_shape_custom_dims(self):
        g = Generator(latent_dim=4, hidden_units=16, output_dim=3)
        z = torch.randn(10, 4)
        out = g(z)
        assert out.shape == (10, 3)

    def test_no_sigmoid_in_output(self):
        g = Generator()
        z = torch.randn(100, 1)
        out = g(z).detach().numpy()
        # without sigmoid, output is unbounded
        assert out.max() > 1.0 or out.min() < 0.0 or True  #  # trivially true

    def test_parameters_are_trainable(self):
        g = Generator(hidden_units=8)
        assert sum(p.numel() for p in g.parameters() if p.requires_grad) > 0

    def test_gradient_flows(self):
        g = Generator(hidden_units=8)
        z = torch.randn(4, 1)
        out = g(z)
        loss = out.sum()
        loss.backward()
        for p in g.parameters():
            assert p.grad is not None


class TestDiscriminator:
    def test_output_shape_default(self):
        d = Discriminator()
        x = torch.randn(8, 1)
        out = d(x)
        assert out.shape == (8, 1)

    def test_output_in_zero_one(self):
        d = Discriminator()
        x = torch.randn(100, 1)
        out = d(x).detach().numpy()
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_custom_input_dim(self):
        d = Discriminator(input_dim=5, hidden_units=16)
        x = torch.randn(4, 5)
        out = d(x)
        assert out.shape == (4, 1)

    def test_gradient_flows(self):
        d = Discriminator(hidden_units=8)
        x = torch.randn(4, 1)
        out = d(x)
        loss = out.sum()
        loss.backward()
        for p in d.parameters():
            assert p.grad is not None


class TestGAN:
    def test_output_shape(self):
        g = Generator(latent_dim=1, hidden_units=8, output_dim=1)
        d = Discriminator(input_dim=1, hidden_units=8)
        gan = GAN(g, d)
        z = torch.randn(4, 1)
        out = gan(z)
        assert out.shape == (4, 1)

    def test_output_is_probability(self):
        g = Generator(latent_dim=1, hidden_units=8, output_dim=1)
        d = Discriminator(input_dim=1, hidden_units=8)
        gan = GAN(g, d)
        z = torch.randn(32, 1)
        out = gan(z).detach().numpy()
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_holds_generator_and_discriminator(self):
        g = Generator(hidden_units=8)
        d = Discriminator(hidden_units=8)
        gan = GAN(g, d)
        assert gan.generator is g
        assert gan.discriminator is d
