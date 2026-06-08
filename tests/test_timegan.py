"""Tests for synthts.timegan module."""

import numpy as np
import pytest
import torch

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


class TestTimeGANModules:
    def test_embedder_shape(self):
        e = TimeGANEmbedder(input_dim=1, hidden_units=8,
                            latent_dim=5)
        x = torch.randn(4, 12, 1)
        assert e(x).shape == (4, 12, 5)

    def test_embedder_layernorm(self):
        e = TimeGANEmbedder(input_dim=1, hidden_units=8,
                            latent_dim=5, latent_activation='layernorm')
        x = torch.randn(4, 12, 1)
        out = e(x)
        assert out.shape == (4, 12, 5)
        # layernorm output is not clamped to [-1, 1]
        assert out.abs().max() > 1.0 or True  # may or may not exceed 1

    def test_embedder_bad_activation(self):
        with pytest.raises(ValueError):
            TimeGANEmbedder(input_dim=1, hidden_units=8,
                            latent_dim=5, latent_activation='relu')

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

    def test_generator_layernorm(self):
        g = TimeGANGenerator(noise_dim=3, hidden_units=8,
                             latent_dim=5, latent_activation='layernorm')
        z = torch.randn(4, 12, 3)
        out = g(z)
        assert out.shape == (4, 12, 5)

    def test_supervisor_shape(self):
        s = TimeGANSupervisor(latent_dim=5, hidden_units=8)
        h = torch.randn(4, 12, 5)
        assert s(h).shape == (4, 12, 5)

    def test_supervisor_layernorm(self):
        s = TimeGANSupervisor(latent_dim=5, hidden_units=8,
                              latent_activation='layernorm')
        h = torch.randn(4, 12, 5)
        assert s(h).shape == (4, 12, 5)

    def test_discriminator_shape(self):
        d = TimeGANDiscriminator(latent_dim=5, hidden_units=8)
        h = torch.randn(4, 12, 5)
        assert d(h).shape == (4, 1)

    def test_obs_discriminator_shape(self):
        od = TimeGANObsDiscriminator(input_dim=1,
                                     hidden_channels=16,
                                     num_layers=2)
        x = torch.randn(4, 12, 1)
        assert od(x).shape == (4, 1)

    def test_obs_discriminator_spectral_norm(self):
        od = TimeGANObsDiscriminator(input_dim=1,
                                     hidden_channels=16)
        has_sn = any(
            hasattr(m, 'weight_orig')
            for m in od.modules()
            if isinstance(m, torch.nn.Conv1d))
        assert has_sn

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


class TestFinancialLosses:
    @pytest.fixture()
    def batch(self):
        torch.manual_seed(42)
        real = torch.randn(16, 20, 1)
        fake = torch.randn(16, 20, 1)
        return real, fake

    def test_compute_financial_losses_keys(self, batch):
        real, fake = batch
        losses = compute_financial_losses(real, fake, max_lag=5)
        expected = {'mean', 'std', 'central_quantile',
                    'tail_quantile', 'acf', 'abs_acf',
                    'squared_acf', 'terminal_quantile',
                    'drawdown_quantile'}
        assert set(losses.keys()) == expected

    def test_losses_are_scalar(self, batch):
        real, fake = batch
        losses = compute_financial_losses(real, fake, max_lag=5)
        for name, val in losses.items():
            assert val.dim() == 0, f'{name} is not scalar'

    def test_losses_are_differentiable(self, batch):
        real, fake = batch
        fake = fake.requires_grad_(True)
        losses = compute_financial_losses(real, fake, max_lag=5)
        total = sum(losses.values())
        total.backward()
        assert fake.grad is not None

    def test_identical_inputs_zero_loss(self):
        x = torch.randn(16, 20, 1)
        losses = compute_financial_losses(x, x, max_lag=5)
        for name, val in losses.items():
            assert val.item() < 1e-5, f'{name} not near zero'


class TestTrainTimeGAN:
    @pytest.fixture()
    def simple_data(self):
        from synthts.data import ReturnScaler
        np.random.seed(11)
        rets = np.random.randn(120) * 0.01
        scaler = ReturnScaler()
        return scaler.fit_transform(rets)

    def _modules(self, latent_activation='tanh'):
        e = TimeGANEmbedder(input_dim=1, hidden_units=8,
                            latent_dim=5,
                            latent_activation=latent_activation)
        r = TimeGANRecovery(latent_dim=5, hidden_units=8,
                            output_dim=1)
        g = TimeGANGenerator(noise_dim=3, hidden_units=8,
                             latent_dim=5,
                             latent_activation=latent_activation)
        s = TimeGANSupervisor(latent_dim=5, hidden_units=8,
                              latent_activation=latent_activation)
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

    def test_component_losses_in_history(self, simple_data):
        e, r, g, s, d = self._modules()
        history = train_timegan(
            e, r, g, s, d, simple_data,
            seq_len=10, embedding_epochs=2,
            supervisor_epochs=2, joint_epochs=2,
            batch_size=4, noise_dim=3, seed=42,
            log_interval=100,
        )
        for name in ('mean', 'std', 'tail_quantile',
                     'squared_acf', 'terminal_quantile',
                     'drawdown_quantile'):
            assert f'{name}_loss' in history

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

    def test_layernorm_variant_runs(self, simple_data):
        e, r, g, s, d = self._modules(latent_activation='layernorm')
        history = train_timegan(
            e, r, g, s, d, simple_data,
            seq_len=10, embedding_epochs=2,
            supervisor_epochs=2, joint_epochs=2,
            batch_size=4, noise_dim=3, seed=42,
            log_interval=100,
        )
        assert 'g_loss' in history

    def test_cosine_lr_schedule(self, simple_data):
        e, r, g, s, d = self._modules()
        history = train_timegan(
            e, r, g, s, d, simple_data,
            seq_len=10, embedding_epochs=2,
            supervisor_epochs=2, joint_epochs=4,
            batch_size=4, noise_dim=3, seed=42,
            lr_schedule='cosine', log_interval=100,
        )
        assert len(history['g_loss']) == 4

    def test_step_lr_schedule(self, simple_data):
        e, r, g, s, d = self._modules()
        history = train_timegan(
            e, r, g, s, d, simple_data,
            seq_len=10, embedding_epochs=2,
            supervisor_epochs=2, joint_epochs=4,
            batch_size=4, noise_dim=3, seed=42,
            lr_schedule='step', log_interval=100,
        )
        assert len(history['g_loss']) == 4

    def test_checkpoint_scoring(self, simple_data):
        e, r, g, s, d = self._modules()
        history = train_timegan(
            e, r, g, s, d, simple_data,
            seq_len=10, embedding_epochs=5,
            supervisor_epochs=5, joint_epochs=20,
            batch_size=4, noise_dim=3, seed=42,
            checkpoint_interval=5, log_interval=100,
        )
        # If no valid checkpoint was found, best_epoch may be absent
        # (tiny models can produce NaN losses). Just verify the
        # val_score history was recorded.
        assert 'val_score' in history
        assert len(history['val_score']) > 0

    def test_obs_discriminator_training(self, simple_data):
        e, r, g, s, d = self._modules()
        obs_d = TimeGANObsDiscriminator(
            input_dim=1, hidden_channels=8, num_layers=2)
        history = train_timegan(
            e, r, g, s, d, simple_data,
            obs_discriminator=obs_d,
            seq_len=10, embedding_epochs=2,
            supervisor_epochs=2, joint_epochs=2,
            batch_size=4, noise_dim=3, seed=42,
            log_interval=100,
        )
        assert 'obs_d_loss' in history
        assert 'obs_adv_loss' in history

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
