"""Numerical regressions for the publication corrections."""

import copy

import numpy as np
import pytest
import torch

from synthts._validation import backward, step
from synthts.evaluate import max_drawdown
from synthts.timegan import (
    TimeGANDiscriminator, TimeGANEmbedder, TimeGANGenerator,
    TimeGANRecovery, TimeGANSupervisor, train_timegan,
)
from synthts.vanilla import Generator, Discriminator, train_gan
from synthts.wgangp import WGANGenerator, Critic, train_wgangp


@pytest.mark.parametrize('n_critic', [None, 1, 2])
def test_dense_histories_are_finite(n_critic):
    data = np.random.default_rng(1).normal(size=(32, 1))
    if n_critic is None:
        history = train_gan(Generator(hidden_units=4),
                            Discriminator(hidden_units=4), data, epochs=3)
    else:
        history = train_wgangp(WGANGenerator(hidden_units=4),
                               Critic(hidden_units=4), data, epochs=3,
                               n_critic=n_critic)
    assert all(len(values) == 3 for values in history.values())
    assert all(np.isfinite(values).all() for values in history.values())
    if n_critic == 2:
        assert set(history) == {
            'g_loss', 'critic_loss', 'wasserstein_distance',
            'gradient_penalty'}
        np.testing.assert_allclose(
            history['critic_loss'],
            np.subtract(history['gradient_penalty'],
                        history['wasserstein_distance']), atol=1e-6)


@pytest.mark.parametrize('returns, expected', [
    ([-0.10], np.expm1(-0.10)),
    ([-0.10, 0.05], np.expm1(-0.10)),
    (np.log([1.1, 0.8, 1.1]), -0.2),
    ([1000.0, -0.10], np.expm1(-0.10)),
])
def test_drawdown_includes_initial_wealth(returns, expected):
    assert max_drawdown(np.array(returns)) == pytest.approx(expected)


def tiny_timegan():
    return (TimeGANEmbedder(1, 4, 3), TimeGANRecovery(3, 4, 1),
            TimeGANGenerator(2, 4, 3), TimeGANSupervisor(3, 4),
            TimeGANDiscriminator(3, 4))


@pytest.mark.parametrize('weight', ['mean_weight', 'std_weight'])
def test_timegan_moment_weight_changes_objective(weight):
    torch.manual_seed(3)
    models = tiny_timegan()
    data = np.random.default_rng(1).normal(size=(32, 1))
    results = []
    for value in (0, 3):
        modules = copy.deepcopy(models)
        history = train_timegan(
            *modules, data, seq_len=8, stat_max_lag=3,
            embedding_epochs=1, supervisor_epochs=1, joint_epochs=2,
            batch_size=4, noise_dim=2, **{weight: value})
        assert all(np.isfinite(v).all() for v in history.values())
        assert all(torch.isfinite(p).all() for m in modules
                   for p in m.parameters())
        results.append((history, modules))
    assert results[0][0]['g_loss'] != results[1][0]['g_loss']
    assert any(not torch.equal(a, b) for a, b in zip(
        results[0][1][2].parameters(), results[1][1][2].parameters()))


@pytest.mark.parametrize('kwargs', [
    {'seq_len': 10}, {'stat_max_lag': 0}, {'lr_schedule': 'typo'},
    {'checkpoint_interval': -1}, {'batch_size': 0},
])
def test_timegan_rejects_invalid_training_settings(kwargs):
    with pytest.raises(ValueError):
        train_timegan(*tiny_timegan(), np.ones((32, 1)), **kwargs)


def test_nonfinite_loss_and_gradient_fail_before_update():
    p = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([p], lr=1)
    with pytest.raises(FloatingPointError, match='loss'):
        backward(p * torch.tensor(float('nan')))
    p.grad = torch.tensor(float('inf'))
    with pytest.raises(FloatingPointError, match='gradient'):
        step(optimizer)
    assert p.item() == 0
