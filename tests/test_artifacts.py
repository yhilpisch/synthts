"""Portable inference and preprocessing regression tests."""

import numpy as np
import pytest
import torch

from synthts.artifacts import save_bundle, load_bundle
from synthts.data import ReturnScaler
from synthts.sequential import SeqGenerator
from synthts.timegan import (
    TimeGANGenerator, TimeGANRecovery, TimeGANSupervisor,
    generate_timegan_sequences,
)
from synthts.vanilla import Generator
from synthts.wgangp import WGANGenerator


@pytest.mark.parametrize('cls', [Generator, WGANGenerator, SeqGenerator])
def test_generator_bundle_round_trip(tmp_path, cls):
    torch.manual_seed(7)
    model = cls(hidden_units=4)
    scaler = ReturnScaler()
    data = np.array([0.01, -0.02, 0.03])
    scaler.fit_transform(data)
    path = tmp_path / 'inference.pt'
    save_bundle(path, {'generator': model}, scaler, {'seed': 7})
    restored = load_bundle(path)
    z = torch.randn(3, model.config['latent_dim'])
    torch.manual_seed(5)
    expected = model(z).detach().numpy()
    torch.manual_seed(5)
    actual = restored['models']['generator'](z).detach().numpy()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(restored['scaler'].transform(data),
                                  scaler.transform(data))
    assert restored['metadata'] == {'seed': 7}


def test_timegan_inference_bundle_round_trip(tmp_path):
    models = {'generator': TimeGANGenerator(2, 4, 3),
              'supervisor': TimeGANSupervisor(3, 4),
              'recovery': TimeGANRecovery(3, 4, 1)}
    scaler = ReturnScaler()
    scaler.fit_transform(np.arange(10.))
    path = tmp_path / 'timegan.pt'
    save_bundle(path, models, scaler, {'protocol': 'synthetic test'})
    restored = load_bundle(path)['models']
    expected = generate_timegan_sequences(**models, n_sequences=2,
                                          seq_len=12, noise_dim=2, seed=8)
    actual = generate_timegan_sequences(**restored, n_sequences=2,
                                        seq_len=12, noise_dim=2, seed=8)
    np.testing.assert_array_equal(actual, expected)
