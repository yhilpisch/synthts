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
