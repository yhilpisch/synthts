"""Tests for synthts.sequential module."""

import numpy as np
import torch
import pytest

from synthts.sequential import (
    SeqGenerator, SeqDiscriminator,
    train_seq_gan, generate_sequences,
)


class TestSeqGenerator:
    def test_output_shape(self):
        g = SeqGenerator(latent_dim=4, hidden_units=32,
                         output_dim=1, seq_len=50)
        z = torch.randn(8, 4)
        out = g(z)
        assert out.shape == (8, 50, 1)

    def test_default_seq_len(self):
        g = SeqGenerator(latent_dim=1, hidden_units=32)
        z = torch.randn(4, 1)
        out = g(z)
        assert out.shape == (4, 252, 1)

    def test_gradient_flows(self):
        g = SeqGenerator(latent_dim=1, hidden_units=16, seq_len=20,
                         num_layers=1)
        z = torch.randn(4, 1)
        out = g(z)
        loss = out.sum()
        loss.backward()
        for p in g.parameters():
            assert p.grad is not None

    def test_multivariate_output(self):
        g = SeqGenerator(latent_dim=2, hidden_units=32,
                         output_dim=3, seq_len=30)
        z = torch.randn(5, 2)
        out = g(z)
        assert out.shape == (5, 30, 3)


class TestSeqDiscriminator:
    def test_output_shape(self):
        d = SeqDiscriminator(input_dim=1, hidden_units=32)
        x = torch.randn(8, 50, 1)
        out = d(x)
        assert out.shape == (8, 1)

    def test_output_in_zero_one(self):
        d = SeqDiscriminator(input_dim=1, hidden_units=32)
        x = torch.randn(16, 30, 1)
        out = d(x).detach().numpy()
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_gradient_flows(self):
        d = SeqDiscriminator(input_dim=1, hidden_units=16,
                             num_layers=1)
        x = torch.randn(4, 20, 1)
        out = d(x)
        loss = out.sum()
        loss.backward()
        for p in d.parameters():
            assert p.grad is not None


class TestTrainSeqGan:
    @pytest.fixture()
    def simple_data(self):
        from synthts.data import ReturnScaler
        np.random.seed(7)
        rets = np.random.randn(500) * 0.01
        scaler = ReturnScaler()
        return scaler.fit_transform(rets), scaler

    def test_runs_without_error(self, simple_data):
        data_scaled, _ = simple_data
        g = SeqGenerator(latent_dim=1, hidden_units=32,
                         output_dim=1, seq_len=30, num_layers=1)
        d = SeqDiscriminator(input_dim=1, hidden_units=32,
                             num_layers=1)
        history = train_seq_gan(
            g, d, data_scaled,
            seq_len=30, epochs=10, batch_size=8,
            latent_dim=1, seed=42, log_interval=100)
        assert history is not None

    def test_generator_weights_change(self, simple_data):
        data_scaled, _ = simple_data
        g = SeqGenerator(latent_dim=1, hidden_units=16,
                         output_dim=1, seq_len=30, num_layers=1)
        d = SeqDiscriminator(input_dim=1, hidden_units=16,
                             num_layers=1)
        w0 = list(g.parameters())[0].data.clone()
        train_seq_gan(g, d, data_scaled,
                      seq_len=30, epochs=10, batch_size=8,
                      seed=42, log_interval=100)
        w1 = list(g.parameters())[0].data
        assert not torch.equal(w0, w1)


class TestGenerateSequences:
    def test_output_shape(self):
        g = SeqGenerator(latent_dim=1, hidden_units=16,
                         output_dim=1, seq_len=30, num_layers=1)
        seqs = generate_sequences(g, n_sequences=5, latent_dim=1)
        assert seqs.shape == (5, 30, 1)

    def test_output_is_numpy(self):
        g = SeqGenerator(latent_dim=1, hidden_units=16,
                         output_dim=1, seq_len=20, num_layers=1)
        seqs = generate_sequences(g, n_sequences=3, latent_dim=1)
        assert isinstance(seqs, np.ndarray)
