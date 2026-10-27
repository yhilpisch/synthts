"""Tests for synthts.data module."""

import numpy as np
import pandas as pd
import pytest

from synthts.data import ReturnScaler, compute_log_returns, load_eod


def test_load_eod_drops_only_selected_symbol_gaps(monkeypatch):
    raw = pd.DataFrame(
        {
            'GLD': [100.0, 101.0],
            'SPY': [np.nan, 200.0],
        },
        index=pd.date_range('2026-01-01', periods=2),
    )
    monkeypatch.setattr(pd, 'read_csv', lambda *args, **kwargs: raw)
    result = load_eod(symbol='GLD')
    assert len(result) == 2


# ------------------------------------------------------------------
# compute_log_returns
# ------------------------------------------------------------------

class TestComputeLogReturns:
    def test_output_is_1d_ndarray(self):
        prices = pd.Series([100, 101, 102, 101.5, 103])
        rets = compute_log_returns(prices)
        assert isinstance(rets, np.ndarray)
        assert rets.ndim == 1

    def test_length_matches_input_minus_one(self):
        prices = pd.Series(np.random.rand(50) * 100 + 50)
        rets = compute_log_returns(prices)
        assert len(rets) == len(prices) - 1

    def test_zero_price_raises(self):
        prices = pd.Series([100, 101, 0, 102])  # zero price
        with pytest.raises(ValueError, match='strictly positive'):
            compute_log_returns(prices)

    def test_missing_price_is_dropped(self):
        prices = pd.Series([100, 101, np.nan, 102])
        rets = compute_log_returns(prices)
        assert np.isfinite(rets).all()

    def test_values_are_log_returns(self):
        prices = pd.Series([100.0, 105.0, 110.25])
        rets = compute_log_returns(prices)
        expected = np.log([105 / 100, 110.25 / 105])
        np.testing.assert_allclose(rets, expected, rtol=1e-12)

    def test_constant_prices_yield_zero_returns(self):
        prices = pd.Series([50.0] * 10)
        rets = compute_log_returns(prices)
        np.testing.assert_allclose(rets, 0.0, atol=1e-15)


# ------------------------------------------------------------------
# ReturnScaler
# ------------------------------------------------------------------

class TestReturnScaler:
    @pytest.fixture()
    def scaler(self):
        return ReturnScaler()

    def test_fit_transform_output_shape(self, scaler):
        rets = np.random.randn(100)
        out = scaler.fit_transform(rets)
        assert out.shape == (100, 1)

    def test_fit_transform_2d_input(self, scaler):
        rets = np.random.randn(100).reshape(-1, 1)
        out = scaler.fit_transform(rets)
        assert out.shape == (100, 1)

    def test_roundtrip(self, scaler):
        rets = np.random.randn(200)
        scaled = scaler.fit_transform(rets)
        recovered = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(recovered, rets, rtol=1e-10)

    def test_scaled_zero_mean_unit_var(self, scaler):
        rets = np.random.randn(500)
        scaled = scaler.fit_transform(rets).flatten()
        np.testing.assert_allclose(scaled.mean(), 0.0, atol=0.05)
        np.testing.assert_allclose(scaled.std(ddof=1), 1.0, atol=0.05)

    def test_inverse_transform_1d_input(self, scaler):
        rets = np.random.randn(50)
        scaled = scaler.fit_transform(rets)
        recovered = scaler.inverse_transform(scaled.flatten())
        np.testing.assert_allclose(recovered, rets, rtol=1e-10)


@pytest.mark.parametrize('n_obs', [0, -1, 1.5, True])
def test_load_eod_rejects_invalid_count(n_obs):
    with pytest.raises(ValueError, match='n_obs'):
        load_eod(url='not-read.csv', n_obs=n_obs)


def test_multivariate_scaler_preserves_features_and_training_mean():
    train = np.array([[1., 10.], [3., 20.], [5., 30.]])
    future = np.array([[7., 40.], [9., 50.]])
    scaler = ReturnScaler()
    scaler.fit_transform(train)
    mean = scaler.scaler.mean_.copy()
    recovered = scaler.inverse_transform(scaler.transform(future))
    assert recovered.shape == future.shape
    np.testing.assert_allclose(recovered, future)
    np.testing.assert_array_equal(scaler.scaler.mean_, mean)


def test_returns_do_not_bridge_missing_prices():
    prices = pd.Series([100., 110., np.nan, 121., 133.1])
    np.testing.assert_allclose(compute_log_returns(prices),
                               [np.log(1.1), np.log(1.1)])


def test_extreme_positive_prices_produce_finite_log_returns():
    result = compute_log_returns(pd.Series([1e-300, 1e300]))
    np.testing.assert_allclose(result, [600 * np.log(10)])


@pytest.mark.parametrize('values', [[], [np.nan], [[1., np.inf]]])
def test_scaler_rejects_invalid_data(values):
    with pytest.raises(ValueError):
        ReturnScaler().fit_transform(np.array(values))
