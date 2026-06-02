"""Tests for synthts.timegan module."""

import numpy as np
import pytest
import torch

from synthts.timegan import (
    TimeGANDiscriminator,
    TimeGANEmbedder,
    TimeGANGenerator,
    TimeGANRecovery,
    TimeGANSupervisor,
    generate_timegan_sequences,
    train_timegan,
)


class TestTimeGANModules:
    def test_embedder_shape(self):
        e = TimeGANEmbedder(input_dim=1, hidden_units=8,
                            latent_dim=5)
        x = torch.randn(4, 12, 1)
        assert e(x).shape == (4, 12, 5)

    def test_recovery_shape(self):
        r = TimeGANRecovery(latent_dim=5, hidden_units=8,
                            output_dim=1)
        h = torch.randn(4, 12, 5)
        assert r(h).shape == (4, 12, 1)

    def test_generator_shape(self):
        g = TimeGANGenerator(noise_dim=3, hidden_units=8,
                             latent_dim=5)
        z = torch.randn(4, 12, 3)
        assert g(z).shape == (4, 12, 5)

    def test_supervisor_shape(self):
        s = TimeGANSupervisor(latent_dim=5, hidden_units=8)
        h = torch.randn(4, 12, 5)
        assert s(h).shape == (4, 12, 5)

    def test_discriminator_shape(self):
        d = TimeGANDiscriminator(latent_dim=5, hidden_units=8)
        h = torch.randn(4, 12, 5)
        assert d(h).shape == (4, 1)

    def test_gradient_flows(self):
        e = TimeGANEmbedder(input_dim=1, hidden_units=8,
                            latent_dim=5)
        r = TimeGANRecovery(latent_dim=5, hidden_units=8,
                            output_dim=1)
        x = torch.randn(4, 12, 1)
        loss = r(e(x)).sum()
        loss.backward()
        for module in (e, r):
            for param in module.parameters():
                assert param.grad is not None


class TestTrainTimeGAN:
    @pytest.fixture()
    def simple_data(self):
        from synthts.data import ReturnScaler
        np.random.seed(11)
        rets = np.random.randn(120) * 0.01
        scaler = ReturnScaler()
        return scaler.fit_transform(rets)

    def _modules(self):
        e = TimeGANEmbedder(input_dim=1, hidden_units=8,
                            latent_dim=5)
        r = TimeGANRecovery(latent_dim=5, hidden_units=8,
                            output_dim=1)
        g = TimeGANGenerator(noise_dim=3, hidden_units=8,
                             latent_dim=5)
        s = TimeGANSupervisor(latent_dim=5, hidden_units=8)
        d = TimeGANDiscriminator(latent_dim=5, hidden_units=8)
        return e, r, g, s, d

    def test_runs_without_error(self, simple_data):
        e, r, g, s, d = self._modules()
        history = train_timegan(
            e, r, g, s, d, simple_data,
            seq_len=10, embedding_epochs=2,
            supervisor_epochs=2, joint_epochs=2,
            batch_size=4, noise_dim=3, seed=42,
            log_interval=100,
        )
        for key in ['recon_loss', 'supervised_loss',
                    'g_loss', 'd_loss', 'moment_loss']:
            assert key in history

    def test_generator_weights_change(self, simple_data):
        e, r, g, s, d = self._modules()
        w0 = list(g.parameters())[0].data.clone()
        train_timegan(
            e, r, g, s, d, simple_data,
            seq_len=10, embedding_epochs=2,
            supervisor_epochs=2, joint_epochs=2,
            batch_size=4, noise_dim=3, seed=42,
            log_interval=100,
        )
        w1 = list(g.parameters())[0].data
        assert not torch.equal(w0, w1)

    def test_short_data_raises(self):
        e, r, g, s, d = self._modules()
        with pytest.raises(ValueError):
            train_timegan(
                e, r, g, s, d, np.random.randn(5, 1),
                seq_len=10, embedding_epochs=1,
                supervisor_epochs=1, joint_epochs=1,
                batch_size=4, noise_dim=3,
            )


class TestGenerateTimeGANSequences:
    def test_output_shape(self):
        r = TimeGANRecovery(latent_dim=5, hidden_units=8,
                            output_dim=1)
        g = TimeGANGenerator(noise_dim=3, hidden_units=8,
                             latent_dim=5)
        s = TimeGANSupervisor(latent_dim=5, hidden_units=8)
        seqs = generate_timegan_sequences(
            g, s, r, n_sequences=6, seq_len=10,
            noise_dim=3, seed=5,
        )
        assert seqs.shape == (6, 10, 1)
        assert isinstance(seqs, np.ndarray)

    def test_seed_reproducible_output(self):
        r = TimeGANRecovery(latent_dim=5, hidden_units=8,
                            output_dim=1)
        g = TimeGANGenerator(noise_dim=3, hidden_units=8,
                             latent_dim=5)
        s = TimeGANSupervisor(latent_dim=5, hidden_units=8)
        seqs_1 = generate_timegan_sequences(
            g, s, r, n_sequences=2, seq_len=10,
            noise_dim=3, seed=7,
        )
        seqs_2 = generate_timegan_sequences(
            g, s, r, n_sequences=2, seq_len=10,
            noise_dim=3, seed=7,
        )
        assert np.allclose(seqs_1, seqs_2)
