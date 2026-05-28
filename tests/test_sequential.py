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
        g = SeqGenerator(latent_dim=4, noise_dim=2,
                         hidden_units=32, output_dim=1,
                         seq_len=50, num_layers=1)
        z = torch.randn(8, 4)
        out = g(z)
        assert out.shape == (8, 50, 1)

    def test_default_params(self):
        g = SeqGenerator()
        assert g.latent_dim == 4
        assert g.noise_dim == 2
        assert g.seq_len == 63
        z = torch.randn(4, 4)
        out = g(z)
        assert out.shape == (4, 63, 1)

    def test_gradient_flows(self):
        g = SeqGenerator(latent_dim=2, noise_dim=1,
                         hidden_units=16, seq_len=20,
                         num_layers=1)
        z = torch.randn(4, 2)
        out = g(z)
        loss = out.sum()
        loss.backward()
        for p in g.parameters():
            assert p.grad is not None

    def test_multivariate_output(self):
        g = SeqGenerator(latent_dim=2, noise_dim=1,
                         hidden_units=32, output_dim=3,
                         seq_len=30, num_layers=1)
        z = torch.randn(5, 2)
        out = g(z)
        assert out.shape == (5, 30, 3)

    def test_noise_dim_varies_output(self):
        g = SeqGenerator(latent_dim=2, noise_dim=2,
                         hidden_units=16, seq_len=30,
                         num_layers=1)
        g.eval()
        z = torch.randn(1, 2)
        with torch.no_grad():
            out1 = g(z).numpy()
            out2 = g(z).numpy()
        # same latent, different per-step noise -> different output
        assert not np.allclose(out1, out2)


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
        d = SeqDiscriminator(input_dim=1, hidden_units=16)
        x = torch.randn(4, 20, 1)
        out = d(x)
        loss = out.sum()
        loss.backward()
        for p in d.parameters():
            assert p.grad is not None

    def test_has_spectral_norm(self):
        d = SeqDiscriminator(input_dim=1, hidden_units=32)
        # check that SN parameters exist (weight_orig)
        sn_params = [n for n, _ in d.named_parameters()
                     if 'weight_orig' in n]
        assert len(sn_params) > 0

    def test_cnn_architecture(self):
        d = SeqDiscriminator(input_dim=1, hidden_units=16)
        # should contain Conv1d layers
        conv_layers = [m for m in d.modules()
                       if isinstance(m, torch.nn.Conv1d)]
        assert len(conv_layers) == 2


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
        g = SeqGenerator(latent_dim=2, noise_dim=1,
                         hidden_units=16, output_dim=1,
                         seq_len=30, num_layers=1)
        d = SeqDiscriminator(input_dim=1, hidden_units=16)
        history = train_seq_gan(
            g, d, data_scaled,
            seq_len=30, epochs=10, batch_size=8,
            latent_dim=2, seed=42, log_interval=100)
        assert history is not None
        assert len(history['g_loss']) == 10

    def test_generator_weights_change(self, simple_data):
        data_scaled, _ = simple_data
        g = SeqGenerator(latent_dim=2, noise_dim=1,
                         hidden_units=16, output_dim=1,
                         seq_len=30, num_layers=1)
        d = SeqDiscriminator(input_dim=1, hidden_units=16)
        w0 = list(g.parameters())[0].data.clone()
        train_seq_gan(g, d, data_scaled,
                      seq_len=30, epochs=10, batch_size=8,
                      latent_dim=2, seed=42, log_interval=100)
        w1 = list(g.parameters())[0].data
        assert not torch.equal(w0, w1)

    def test_label_smoothing(self, simple_data):
        data_scaled, _ = simple_data
        g = SeqGenerator(latent_dim=2, noise_dim=1,
                         hidden_units=16, output_dim=1,
                         seq_len=30, num_layers=1)
        d = SeqDiscriminator(input_dim=1, hidden_units=16)
        history = train_seq_gan(
            g, d, data_scaled,
            seq_len=30, epochs=10, batch_size=8,
            latent_dim=2, label_smooth=0.1,
            seed=42, log_interval=100)
        assert 'g_loss' in history

    def test_short_data_raises(self):
        data_scaled = np.random.randn(10, 1)
        g = SeqGenerator(latent_dim=2, noise_dim=1,
                         hidden_units=16, output_dim=1,
                         seq_len=30, num_layers=1)
        d = SeqDiscriminator(input_dim=1, hidden_units=16)
        with pytest.raises(ValueError):
            train_seq_gan(g, d, data_scaled,
                          seq_len=30, epochs=1, batch_size=8,
                          latent_dim=2, seed=42, log_interval=100)


class TestGenerateSequences:
    def test_output_shape(self):
        g = SeqGenerator(latent_dim=2, noise_dim=1,
                         hidden_units=16, output_dim=1,
                         seq_len=30, num_layers=1)
        seqs = generate_sequences(g, n_sequences=5, latent_dim=2)
        assert seqs.shape == (5, 30, 1)

    def test_output_is_numpy(self):
        g = SeqGenerator(latent_dim=2, noise_dim=1,
                         hidden_units=16, output_dim=1,
                         seq_len=20, num_layers=1)
        seqs = generate_sequences(g, n_sequences=3, latent_dim=2)
        assert isinstance(seqs, np.ndarray)

    def test_different_latent_different_output(self):
        g = SeqGenerator(latent_dim=2, noise_dim=1,
                         hidden_units=16, output_dim=1,
                         seq_len=30, num_layers=1)
        g.eval()
        with torch.no_grad():
            s1 = generate_sequences(g, 1, latent_dim=2)
            s2 = generate_sequences(g, 1, latent_dim=2)
        assert not np.allclose(s1, s2)

    def test_seed_reproducible_output(self):
        g = SeqGenerator(latent_dim=2, noise_dim=1,
                         hidden_units=16, output_dim=1,
                         seq_len=30, num_layers=1)
        s1 = generate_sequences(g, 2, latent_dim=2, seed=10)
        s2 = generate_sequences(g, 2, latent_dim=2, seed=10)
        assert np.allclose(s1, s2)
