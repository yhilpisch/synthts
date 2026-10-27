"""Data loading, return computation, and scaling utilities."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from synthts._validation import integer


EOD_URL = 'https://hilpisch.com/eod_data.csv'


def load_eod(url: str=EOD_URL, symbol: str='GLD',
             n_obs: int=None) -> pd.Series:
    """Load EOD price data for a single symbol.

    Parameters
    ----------
    url : str
        CSV source URL or local path, with unique increasing dates.
    symbol : str
        Column name to extract.
    n_obs : int, optional
        Number of most recent observations to keep.

    Returns
    -------
    pd.Series
        Price series indexed by date.
    """
    if n_obs is not None:
        integer('n_obs', n_obs)
    raw = pd.read_csv(url, index_col=0, parse_dates=True)
    if (not isinstance(raw.index, pd.DatetimeIndex)
            or raw.index.hasnans or not raw.index.is_unique
            or not raw.index.is_monotonic_increasing):
        raise ValueError('CSV dates must be unique and increasing')
    series = raw[symbol].dropna()
    if n_obs is not None:
        series = series.iloc[-n_obs:]
    return series


def compute_log_returns(prices: pd.Series) -> np.ndarray:
    """Compute adjacent log-returns without bridging missing prices.

    A missing price removes both adjacent returns. Differences of logs
    avoid overflow in the ratio of extreme but finite positive prices.

    Returns
    -------
    np.ndarray
        1-D array of finite log-returns.
    """
    observed = prices.dropna()
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError('prices must be finite and strictly positive')
    rets = np.log(prices).diff().dropna()
    return rets.values


class ReturnScaler:
    """Standardise returns, preserving feature columns for multivariate data.

    Transform methods return matrices. Inversion returns a vector for a
    single feature and an observation-by-feature matrix otherwise.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, rets: np.ndarray) -> np.ndarray:
        """Fit scaler and return standardised returns.

        Parameters
        ----------
        rets : np.ndarray
            1-D or 2-D array of log-returns.

        Returns
        -------
        np.ndarray
            Standardised returns, shape (n, features).
        """
        return self.scaler.fit_transform(self._matrix(rets))

    @staticmethod
    def _matrix(values):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or min(values.shape) == 0:
            raise ValueError('returns must have shape (n, features)')
        if not np.isfinite(values).all():
            raise ValueError('returns must contain only finite values')
        return values

    def transform(self, rets: np.ndarray) -> np.ndarray:
        """Scale validation/test returns with training-only parameters."""
        return self.scaler.transform(self._matrix(rets))

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        """Map standardised values back to original scale."""
        result = self.scaler.inverse_transform(self._matrix(scaled))
        return result[:, 0] if result.shape[1] == 1 else result


def split_returns(returns, validation_fraction: float=0.2,
                  test_fraction: float=0.2, seq_len: int=63):
    """Split chronological returns before fitting scalers or making windows.

    No window crosses a partition boundary. The final test partition is
    reserved for evaluation after model selection; do not tune on it.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1 or not np.isfinite(returns).all():
        raise ValueError('returns must be a finite 1-D array')
    integer('seq_len', seq_len)
    fractions = np.array([validation_fraction, test_fraction])
    if (not np.isfinite(fractions).all() or (fractions <= 0).any()
            or fractions.sum() >= 1):
        raise ValueError(
            'validation/test fractions must be positive and sum < 1')
    test_start = len(returns) - int(len(returns) * test_fraction)
    val_start = test_start - int(len(returns) * validation_fraction)
    partitions = np.split(returns, [val_start, test_start])
    if min(map(len, partitions)) < seq_len:
        raise ValueError('each chronological split needs at least seq_len rows')
    return tuple(partitions)
