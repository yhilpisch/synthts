"""Data loading, return computation, and scaling utilities."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


EOD_URL = 'https://hilpisch.com/eod_data.csv'


def load_eod(url: str=EOD_URL, symbol: str='GLD',
             n_obs: int=None) -> pd.Series:
    """Load EOD price data for a single symbol.

    Parameters
    ----------
    url : str
        CSV source URL.
    symbol : str
        Column name to extract.
    n_obs : int, optional
        Number of most recent observations to keep.

    Returns
    -------
    pd.Series
        Price series indexed by date.
    """
    raw = pd.read_csv(url, index_col=0, parse_dates=True).dropna()
    series = raw[symbol]
    if n_obs is not None:
        series = series.iloc[-n_obs:]
    return series


def compute_log_returns(prices: pd.Series) -> np.ndarray:
    """Compute log-returns from a price series.

    Returns
    -------
    np.ndarray
        1-D array of log-returns (NaN-free).
    """
    rets = np.log(prices / prices.shift(1)).dropna()
    return rets.values


class ReturnScaler:
    """Standardise and invert log-returns via sklearn StandardScaler."""

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
            Standardised returns, shape (n, 1).
        """
        if rets.ndim == 1:
            rets = rets.reshape(-1, 1)
        return self.scaler.fit_transform(rets)

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        """Map standardised values back to original scale."""
        if scaled.ndim == 1:
            scaled = scaled.reshape(-1, 1)
        return self.scaler.inverse_transform(scaled).flatten()
