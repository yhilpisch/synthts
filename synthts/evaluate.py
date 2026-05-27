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
