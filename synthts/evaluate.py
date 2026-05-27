"""Statistical evaluation and visualisation of synthetic data."""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def ks_test(real: np.ndarray, synth: np.ndarray) -> dict:
    """Two-sample Kolmogorov-Smirnov test.

    Returns
    -------
    dict
        Keys 'statistic', 'pvalue'.
    """
    result = stats.kstest(synth, real)
    return {'statistic': result.statistic, 'pvalue': result.pvalue}


def ks_test_batch(real: np.ndarray, synth_df: pd.DataFrame,
                  alpha: float=0.05) -> np.ndarray:
    """Run KS test for every synthetic column against real data.

    Parameters
    ----------
    real : np.ndarray
        1-D real return array.
    synth_df : pd.DataFrame
        Columns of synthetic returns.
    alpha : float
        Significance level.

    Returns
    -------
    np.ndarray
        Array of p-values.
    """
    pvals = []
    for col in synth_df.columns:
        pvals.append(ks_test(real, synth_df[col].values)['pvalue'])
    return np.array(pvals)


def pass_rate(pvals: np.ndarray, alpha: float=0.05) -> float:
    """Fraction of p-values above alpha."""
    return (pvals > alpha).mean()


def describe_comparison(data: pd.DataFrame, n_show: int=5) -> pd.DataFrame:
    """Descriptive statistics for real vs synthetic columns."""
    return data.describe().round(4).iloc[:, :n_show]


def mse_sorted(real: np.ndarray, synth: np.ndarray) -> float:
    """Mean squared error between sorted real and synthetic data."""
    return ((np.sort(real) - np.sort(synth)) ** 2).mean()


def plot_real_vs_synth(data: pd.DataFrame, n_synth: int=1,
                       alpha: float=0.7):
    """Overlay real and synthetic return paths."""
    data.iloc[:, :n_synth + 1].plot(
        style=['r'] + n_synth * ['b--'], lw=1, alpha=alpha)
    plt.legend();


def plot_histograms(real: np.ndarray, synth: np.ndarray,
                    bins: int=50):
    """Histogram comparison of real vs synthetic returns."""
    plt.hist(real, bins=bins, color='r', alpha=0.7, label='real',
             density=True)
    plt.hist(synth, bins=bins, color='b', alpha=0.7, label='synthetic',
             density=True)
    plt.legend();


def plot_sorted(real: np.ndarray, synth: np.ndarray):
    """Sorted-value comparison (QQ-style)."""
    plt.plot(np.sort(real), 'r', lw=1.0, label='real')
    plt.plot(np.sort(synth), 'b--', lw=1.0, label='synthetic')
    plt.legend();


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
    ax.legend();


def plot_pvalue_histogram(pvals: np.ndarray, alpha: float=0.05):
    """Histogram of KS test p-values with significance line."""
    plt.hist(pvals, bins=100)
    plt.axvline(alpha, color='r')
    plt.title('KS test p-values');


# ------------------------------------------------------------------
# Temporal evaluation metrics
# ------------------------------------------------------------------

def acf(x: np.ndarray, max_lag: int=20) -> np.ndarray:
    """Autocorrelation function up to max_lag."""
    n = len(x)
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
    a_real = acf(real, max_lag)
    a_synth = acf(synth, max_lag)
    a2_real = acf(real ** 2, max_lag)
    a2_synth = acf(synth ** 2, max_lag)
    return {
        'acf_diff': np.abs(a_real - a_synth).sum(),
        'acf2_diff': np.abs(a2_real - a2_synth).sum(),
        'real_acf': a_real,
        'synth_acf': a_synth,
        'real_acf2': a2_real,
        'synth_acf2': a2_synth,
    }


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from a 1-D return array."""
    cum = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return dd.min()


def drawdown_distribution(returns: np.ndarray,
                          n_bootstrap: int=200,
                          rng_seed: int=None) -> np.ndarray:
    """Bootstrap distribution of maximum drawdown.

    Slices random contiguous windows and computes max drawdown
    for each.
    """
    from numpy.random import default_rng
    rng = default_rng(rng_seed)
    window = min(252, len(returns) // 2)
    dds = []
    for _ in range(n_bootstrap):
        start = rng.integers(0, len(returns) - window)
        segment = returns[start:start + window]
        dds.append(max_drawdown(segment))
    return np.array(dds)


def plot_acf_comparison(real: np.ndarray, synth: np.ndarray,
                        max_lag: int=20):
    """Side-by-side ACF and squared-return ACF comparison."""
    comp = compare_acf(real, synth, max_lag)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    lags = np.arange(max_lag + 1)
    axes[0].bar(lags - 0.15, comp['real_acf'], width=0.3,
                color='r', alpha=0.7, label='real')
    axes[0].bar(lags + 0.15, comp['synth_acf'], width=0.3,
                color='b', alpha=0.7, label='synthetic')
    axes[0].set_title('Return ACF')
    axes[0].legend()
    axes[1].bar(lags - 0.15, comp['real_acf2'], width=0.3,
                color='r', alpha=0.7, label='real')
    axes[1].bar(lags + 0.15, comp['synth_acf2'], width=0.3,
                color='b', alpha=0.7, label='synthetic')
    axes[1].set_title('Squared-return ACF')
    axes[1].legend()
    plt.tight_layout();


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
    plt.legend();
