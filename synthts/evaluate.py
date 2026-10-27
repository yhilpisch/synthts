"""Statistical evaluation and visualisation of synthetic data."""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings

from synthts._validation import integer


def _series(values, name='returns'):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError(f'{name} must be a non-empty 1-D array')
    if not np.isfinite(values).all():
        raise ValueError(f'{name} must contain only finite values')
    return values


def _windows(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError('windows must be a non-empty 2-D array')
    if not np.isfinite(values).all():
        raise ValueError('windows must contain only finite values')
    return values


def ks_test(real: np.ndarray, synth: np.ndarray) -> dict:
    """Two-sample Kolmogorov-Smirnov diagnostic.

    The usual p-value assumes independent observations. Serial dependence,
    overlapping windows, and in-sample fitting invalidate that interpretation.
    Non-rejection does not establish equivalence or path realism.

    Returns
    -------
    dict
        Keys 'statistic', 'pvalue'.
    """
    result = stats.ks_2samp(_series(real), _series(synth), method='asymp')
    return {'statistic': result.statistic, 'pvalue': result.pvalue}


def ks_test_batch(real: np.ndarray, synth_df: pd.DataFrame,
                  alpha: float=None) -> np.ndarray:
    """Run KS test for every synthetic column against real data.

    Parameters
    ----------
    real : np.ndarray
        1-D real return array.
    synth_df : pd.DataFrame
        Columns of synthetic returns.
    alpha : float
        Deprecated compatibility parameter; use pass_rate to apply alpha.

    Returns
    -------
    np.ndarray
        Array of p-values.
    """
    if alpha is not None:
        warnings.warn('alpha is unused; apply it in pass_rate instead',
                      DeprecationWarning, stacklevel=2)
    pvals = []
    for col in synth_df.columns:
        pvals.append(ks_test(real, synth_df[col].values)['pvalue'])
    return np.array(pvals)


def pass_rate(pvals: np.ndarray, alpha: float=0.05) -> float:
    """Fraction of p-values above alpha."""
    pvals = _series(pvals, 'pvals')
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one')
    if ((pvals < 0) | (pvals > 1)).any():
        raise ValueError('pvals must lie between zero and one')
    return float((pvals > alpha).mean())


def describe_comparison(data: pd.DataFrame, n_show: int=5) -> pd.DataFrame:
    """Descriptive statistics for real vs synthetic columns."""
    return data.describe().round(4).iloc[:, :n_show]


def mse_sorted(real: np.ndarray, synth: np.ndarray) -> float:
    """Mean squared error between sorted real and synthetic data."""
    real, synth = _series(real), _series(synth)
    if len(real) != len(synth):
        raise ValueError('mse_sorted requires equal-length arrays')
    return ((np.sort(real) - np.sort(synth)) ** 2).mean()


def plot_real_vs_synth(data: pd.DataFrame, n_synth: int=1,
                       alpha: float=0.7):
    """Overlay real and synthetic return paths."""
    data.iloc[:, :n_synth + 1].plot(
        style=['r'] + n_synth * ['b--'], lw=1, alpha=alpha)
    plt.legend()


def plot_histograms(real: np.ndarray, synth: np.ndarray,
                    bins: int=50):
    """Histogram comparison of real vs synthetic returns."""
    plt.hist(real, bins=bins, color='r', alpha=0.7, label='real',
             density=True)
    plt.hist(synth, bins=bins, color='b', alpha=0.7, label='synthetic',
             density=True)
    plt.legend()


def plot_sorted(real: np.ndarray, synth: np.ndarray,
                n_quantiles: int=200):
    """Quantile-quantile comparison of real vs synthetic returns.

    Both series are mapped to a common percentile grid so the
    plot works even when the arrays have different lengths.
    """
    q = np.linspace(0, 1, n_quantiles)
    r_q = np.quantile(real, q)
    s_q = np.quantile(synth, q)
    plt.plot(r_q, s_q, 'b', lw=0.8, label='synth vs real')
    lims = [min(r_q.min(), s_q.min()), max(r_q.max(), s_q.max())]
    plt.plot(lims, lims, 'r--', lw=0.7, label='y = x')
    plt.xlabel('Real quantiles')
    plt.ylabel('Synthetic quantiles')
    plt.legend()


def plot_cumulative(data: pd.DataFrame, n_synth: int=25):
    """Cumulative return paths for real and synthetic series."""
    cols = data.columns[1:n_synth + 1]
    fig, ax = plt.subplots()
    cum = data[cols].cumsum().apply(np.exp)
    for col in cum.columns:
        ax.plot(cum.index, cum[col], 'b--', lw=0.7,
                label='_nolegend_')
    # single legend entry for all synth paths
    ax.plot([], [], 'b--', lw=0.7, label='synthetic')
    mean_path = cum.mean(axis=1)
    ax.plot(mean_path.index, mean_path, 'g', lw=2,
            label='synth mean')
    real_path = data['real'].cumsum().apply(np.exp)
    ax.plot(real_path.index, real_path, 'r', lw=2, label='real')
    ax.legend()


def plot_pvalue_histogram(pvals: np.ndarray, alpha: float=0.05):
    """Histogram of KS test p-values with significance line."""
    plt.hist(pvals, bins=100)
    plt.axvline(alpha, color='r')
    plt.title('KS test p-values')


# ------------------------------------------------------------------
# Temporal evaluation metrics
# ------------------------------------------------------------------

def rolling_windows(returns: np.ndarray, seq_len: int) -> np.ndarray:
    """Create overlapping return windows."""
    returns = _series(returns)
    integer('seq_len', seq_len)
    if seq_len > len(returns):
        raise ValueError('seq_len must not exceed the number of returns')
    return np.lib.stride_tricks.sliding_window_view(returns, seq_len)


def sample_windows(returns: np.ndarray, seq_len: int, n_windows: int,
                   rng_seed: int=None) -> np.ndarray:
    """Sample contiguous return windows."""
    integer('n_windows', n_windows)
    windows = rolling_windows(returns, seq_len)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(windows), n_windows)
    return windows[idx]


def sequence_ks_pvalues(real_windows: np.ndarray,
                        synth_windows: np.ndarray) -> np.ndarray:
    """KS p-values for matched real and synthetic windows."""
    real_windows = _windows(real_windows)
    synth_windows = _windows(synth_windows)
    n = min(len(real_windows), len(synth_windows))
    return np.array([
        ks_test(real_windows[i], synth_windows[i])['pvalue']
        for i in range(n)
    ])

def acf(x: np.ndarray, max_lag: int=20) -> np.ndarray:
    """Biased ACF for 0 <= max_lag < len(x); constant inputs yield zeros."""
    x = _series(x)
    integer('max_lag', max_lag, minimum=0)
    n = len(x)
    if max_lag >= n:
        raise ValueError('max_lag must be smaller than the series length')
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var == 0:
        return np.zeros(max_lag + 1)
    result = np.correlate(x, x, mode='full')[n - 1:]
    result = result[:max_lag + 1] / (var * n)
    return result


def compare_acf(real: np.ndarray, synth: np.ndarray,
                max_lag: int=20) -> dict:
    """Compare ACF of real vs synthetic returns and squared returns.

    Returns
    -------
    dict
        Keys 'acf_diff' (sum of abs differences in ACF),
        'acf2_diff' (same for squared returns).
    """
    real, synth = _series(real), _series(synth)
    a_real = acf(real, max_lag)
    a_synth = acf(synth, max_lag)
    a2_real = acf(real ** 2, max_lag)
    a2_synth = acf(synth ** 2, max_lag)
    return {
        'acf_diff': np.abs(a_real[1:] - a_synth[1:]).sum(),
        'acf2_diff': np.abs(a2_real[1:] - a2_synth[1:]).sum(),
        'real_acf': a_real,
        'synth_acf': a_synth,
        'real_acf2': a2_real,
        'synth_acf2': a2_synth,
    }


def average_acf(windows: np.ndarray, max_lag: int=20) -> np.ndarray:
    """Average ACF over a two-dimensional window array."""
    return np.array([acf(w, max_lag)
                     for w in _windows(windows)]).mean(axis=0)


def compare_acf_windows(real_windows: np.ndarray,
                        synth_windows: np.ndarray,
                        max_lag: int=20) -> dict:
    """Compare average ACFs of real and synthetic windows."""
    real_windows = _windows(real_windows)
    synth_windows = _windows(synth_windows)
    a_real = average_acf(real_windows, max_lag)
    a_synth = average_acf(synth_windows, max_lag)
    a2_real = average_acf(real_windows ** 2, max_lag)
    a2_synth = average_acf(synth_windows ** 2, max_lag)
    return {
        'acf_diff': np.abs(a_real[1:] - a_synth[1:]).sum(),
        'acf2_diff': np.abs(a2_real[1:] - a2_synth[1:]).sum(),
        'real_acf': a_real,
        'synth_acf': a_synth,
        'real_acf2': a2_real,
        'synth_acf2': a2_synth,
    }


def max_drawdown(returns: np.ndarray) -> float:
    """Negative maximum drawdown from log-returns and initial wealth 1.

    Compute in log space to avoid overflow on long or extreme paths.
    """
    returns = _series(returns)
    log_wealth = np.insert(np.cumsum(returns), 0, 0.0)
    peak = np.maximum.accumulate(log_wealth)
    return float(np.expm1((log_wealth - peak).min()))


def drawdown_distribution(returns: np.ndarray,
                          n_bootstrap: int=200,
                          rng_seed: int=None) -> np.ndarray:
    """Bootstrap distribution of maximum drawdown.

    Slices random contiguous windows and computes max drawdown
    for each.
    """
    from numpy.random import default_rng
    returns = _series(returns)
    integer('n_bootstrap', n_bootstrap)
    if len(returns) < 2:
        raise ValueError('returns must contain at least two observations')
    rng = default_rng(rng_seed)
    window = min(252, len(returns) // 2)
    dds = []
    for _ in range(n_bootstrap):
        start = rng.integers(0, len(returns) - window + 1)
        segment = returns[start:start + window]
        dds.append(max_drawdown(segment))
    return np.array(dds)


def drawdown_distribution_windows(windows: np.ndarray) -> np.ndarray:
    """Maximum drawdown for every return window."""
    return np.array([max_drawdown(w) for w in _windows(windows)])


def plot_acf_comparison(real: np.ndarray, synth: np.ndarray,
                        max_lag: int=20):
    """Side-by-side ACF and squared-return ACF comparison."""
    comp = compare_acf(real, synth, max_lag)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    lags = np.arange(1, max_lag + 1)
    axes[0].bar(lags - 0.15, comp['real_acf'][1:], width=0.3,
                color='r', alpha=0.7, label='real')
    axes[0].bar(lags + 0.15, comp['synth_acf'][1:], width=0.3,
                color='b', alpha=0.7, label='synthetic')
    axes[0].set_title('Return ACF')
    axes[0].legend()
    axes[1].bar(lags - 0.15, comp['real_acf2'][1:], width=0.3,
                color='r', alpha=0.7, label='real')
    axes[1].bar(lags + 0.15, comp['synth_acf2'][1:], width=0.3,
                color='b', alpha=0.7, label='synthetic')
    axes[1].set_title('Squared-return ACF')
    axes[1].legend()
    plt.tight_layout()


def plot_acf_window_comparison(real_windows: np.ndarray,
                               synth_windows: np.ndarray,
                               max_lag: int=20):
    """Side-by-side average ACF comparison for return windows."""
    comp = compare_acf_windows(real_windows, synth_windows, max_lag)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    lags = np.arange(1, max_lag + 1)
    axes[0].bar(lags - 0.15, comp['real_acf'][1:], width=0.3,
                color='r', alpha=0.7, label='real')
    axes[0].bar(lags + 0.15, comp['synth_acf'][1:], width=0.3,
                color='b', alpha=0.7, label='synthetic')
    axes[0].set_title('Return ACF')
    axes[0].legend()
    axes[1].bar(lags - 0.15, comp['real_acf2'][1:], width=0.3,
                color='r', alpha=0.7, label='real')
    axes[1].bar(lags + 0.15, comp['synth_acf2'][1:], width=0.3,
                color='b', alpha=0.7, label='synthetic')
    axes[1].set_title('Squared-return ACF')
    axes[1].legend()
    plt.tight_layout()


def plot_drawdown_comparison(real: np.ndarray, synth: np.ndarray,
                             n_bootstrap: int=200, bins: int=30):
    """Histogram overlay of max-drawdown bootstrap distributions."""
    dd_real = drawdown_distribution(real, n_bootstrap, rng_seed=0)
    dd_synth = drawdown_distribution(synth, n_bootstrap, rng_seed=1)
    plt.hist(dd_real, bins=bins, color='r', alpha=0.7,
             label='real', density=True)
    plt.hist(dd_synth, bins=bins, color='b', alpha=0.7,
             label='synthetic', density=True)
    plt.xlabel('Max drawdown')
    plt.legend()


def plot_drawdown_window_comparison(real_windows: np.ndarray,
                                    synth_windows: np.ndarray,
                                    bins: int=30):
    """Histogram overlay of max drawdowns from return windows."""
    dd_real = drawdown_distribution_windows(real_windows)
    dd_synth = drawdown_distribution_windows(synth_windows)
    plt.hist(dd_real, bins=bins, color='r', alpha=0.7,
             label='real', density=True)
    plt.hist(dd_synth, bins=bins, color='b', alpha=0.7,
             label='synthetic', density=True)
    plt.xlabel('Max drawdown')
    plt.legend()
