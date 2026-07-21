"""Tests for synthts.evaluate module."""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from synthts.evaluate import (
    ks_test, ks_test_batch, pass_rate, describe_comparison,
    mse_sorted, plot_real_vs_synth, plot_histograms, plot_sorted,
    plot_cumulative, plot_pvalue_histogram,
    rolling_windows, sample_windows, sequence_ks_pvalues,
    acf, compare_acf, max_drawdown, drawdown_distribution,
    average_acf, compare_acf_windows, drawdown_distribution_windows,
    plot_acf_comparison, plot_drawdown_comparison,
    plot_acf_window_comparison, plot_drawdown_window_comparison,
)


# ------------------------------------------------------------------
# Statistical tests
# ------------------------------------------------------------------

class TestKsTest:
    def test_identical_distributions_high_pvalue(self):
        np.random.seed(0)
        data = np.random.randn(500)
        result = ks_test(data, data + np.random.randn(500) * 1e-6)
        assert result['pvalue'] > 0.05

    def test_very_different_distributions_low_pvalue(self):
        np.random.seed(1)
        real = np.random.randn(500)
        synth = np.random.randn(500) + 10
        result = ks_test(real, synth)
        assert result['pvalue'] < 0.01

    def test_returns_dict_with_expected_keys(self):
        result = ks_test(np.random.randn(50), np.random.randn(50))
        assert 'statistic' in result
        assert 'pvalue' in result

    def test_statistic_in_zero_one(self):
        result = ks_test(np.random.randn(50), np.random.randn(50))
        assert 0.0 <= result['statistic'] <= 1.0


class TestKsTestBatch:
    @pytest.fixture()
    def sample_data(self):
        np.random.seed(42)
        real = np.random.randn(200)
        synth_df = pd.DataFrame({
            f'synth_{i:02d}': np.random.randn(200)
            for i in range(5)
        })
        return real, synth_df

    def test_returns_pvalue_array(self, sample_data):
        real, synth_df = sample_data
        pvals = ks_test_batch(real, synth_df)
        assert isinstance(pvals, np.ndarray)
        assert len(pvals) == 5

    def test_all_pvalues_in_zero_one(self, sample_data):
        real, synth_df = sample_data
        pvals = ks_test_batch(real, synth_df)
        assert (pvals >= 0).all()
        assert (pvals <= 1).all()


class TestPassRate:
    def test_all_pass(self):
        pvals = np.array([0.1, 0.2, 0.5, 0.8])
        assert pass_rate(pvals, alpha=0.05) == 1.0

    def test_none_pass(self):
        pvals = np.array([0.001, 0.002, 0.01])
        assert pass_rate(pvals, alpha=0.05) == 0.0

    def test_partial(self):
        pvals = np.array([0.01, 0.06, 0.10])
        assert pass_rate(pvals, alpha=0.05) == pytest.approx(2 / 3)


class TestMseSorted:
    def test_identical_arrays_zero_mse(self):
        a = np.array([3.0, 1.0, 2.0])
        assert mse_sorted(a, a) == 0.0

    def test_known_offset(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        assert mse_sorted(a, b) == pytest.approx(1.0)

    def test_order_invariant(self):
        a = np.array([3.0, 1.0, 2.0])
        b = np.array([1.0, 3.0, 2.0])
        assert mse_sorted(a, b) == pytest.approx(0.0)


class TestDescribeComparison:
    def test_output_shape(self):
        data = pd.DataFrame({'real': np.random.randn(100)})
        for i in range(3):
            data[f'synth_{i:02d}'] = np.random.randn(100)
        result = describe_comparison(data, n_show=4)
        assert result.shape[1] == 4


# ------------------------------------------------------------------
# Plotting tests — verify they run without error and produce figures
# ------------------------------------------------------------------

class TestPlots:
    @pytest.fixture(autouse=True)
    def close_figures(self):
        yield
        plt.close('all')

    @pytest.fixture()
    def sample_df(self):
        np.random.seed(0)
        n = 50
        data = pd.DataFrame({'real': np.random.randn(n) * 0.01})
        for i in range(3):
            data[f'synth_{i:02d}'] = np.random.randn(n) * 0.01
        return data

    def test_plot_real_vs_synth(self, sample_df):
        plot_real_vs_synth(sample_df, n_synth=2)
        assert plt.gcf().number > 0

    def test_plot_histograms(self):
        plot_histograms(np.random.randn(50), np.random.randn(50))
        assert plt.gcf().number > 0

    def test_plot_sorted(self):
        plot_sorted(np.random.randn(50), np.random.randn(50))
        assert plt.gcf().number > 0

    def test_plot_cumulative(self, sample_df):
        plot_cumulative(sample_df, n_synth=3)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        assert labels == ['synthetic', 'synth mean', 'real']

    def test_plot_pvalue_histogram(self):
        plot_pvalue_histogram(np.random.uniform(size=20))
        assert plt.gcf().number > 0


# ------------------------------------------------------------------
# Temporal evaluation tests
# ------------------------------------------------------------------

class TestAcf:
    def test_lag_zero_is_one(self):
        x = np.random.randn(100)
        a = acf(x, max_lag=10)
        assert a[0] == pytest.approx(1.0)

    def test_output_length(self):
        x = np.random.randn(100)
        a = acf(x, max_lag=20)
        assert len(a) == 21

    def test_white_noise_near_zero(self):
        np.random.seed(42)
        x = np.random.randn(2000)
        a = acf(x, max_lag=10)
        # lags >= 1 should be close to zero for white noise
        assert np.abs(a[1:]).max() < 0.1

    def test_zero_array(self):
        a = acf(np.zeros(50), max_lag=5)
        np.testing.assert_array_equal(a, np.zeros(6))


class TestCompareAcf:
    def test_identical_series_zero_diff(self):
        x = np.random.randn(200)
        comp = compare_acf(x, x)
        assert comp['acf_diff'] == pytest.approx(0.0)
        assert comp['acf2_diff'] == pytest.approx(0.0)

    def test_returns_all_keys(self):
        comp = compare_acf(np.random.randn(50), np.random.randn(50))
        for key in ['acf_diff', 'acf2_diff', 'real_acf', 'synth_acf',
                    'real_acf2', 'synth_acf2']:
            assert key in comp


class TestWindowMetrics:
    def test_rolling_windows_shape(self):
        x = np.arange(10)
        windows = rolling_windows(x, seq_len=4)
        assert windows.shape == (7, 4)

    def test_sample_windows_shape(self):
        x = np.arange(20)
        windows = sample_windows(x, seq_len=5, n_windows=3,
                                 rng_seed=1)
        assert windows.shape == (3, 5)

    def test_sequence_ks_pvalues_length(self):
        real = np.random.randn(4, 10)
        synth = np.random.randn(4, 10)
        pvals = sequence_ks_pvalues(real, synth)
        assert len(pvals) == 4

    def test_average_acf_shape(self):
        windows = np.random.randn(5, 20)
        assert len(average_acf(windows, max_lag=4)) == 5

    def test_compare_acf_windows_keys(self):
        real = np.random.randn(5, 20)
        synth = np.random.randn(5, 20)
        comp = compare_acf_windows(real, synth, max_lag=4)
        assert 'acf_diff' in comp

    def test_drawdown_distribution_windows_length(self):
        windows = np.random.randn(5, 20) * 0.01
        dds = drawdown_distribution_windows(windows)
        assert len(dds) == 5


class TestMaxDrawdown:
    def test_no_drawdown(self):
        # monotonically increasing
        rets = np.array([0.01] * 10)
        assert max_drawdown(rets) == pytest.approx(0.0, abs=1e-10)

    def test_known_drawdown(self):
        # up 10%, down 20%, up 10%
        rets = np.log(np.array([1.1, 0.8, 1.1]))
        dd = max_drawdown(rets)
        assert dd < 0  # negative by definition

    def test_constant_zero_returns(self):
        rets = np.zeros(50)
        assert max_drawdown(rets) == pytest.approx(0.0, abs=1e-10)


class TestDrawdownDistribution:
    def test_output_length(self):
        rets = np.random.randn(500) * 0.01
        dds = drawdown_distribution(rets, n_bootstrap=20, rng_seed=0)
        assert len(dds) == 20

    def test_all_negative_or_zero(self):
        rets = np.random.randn(500) * 0.01
        dds = drawdown_distribution(rets, n_bootstrap=50, rng_seed=0)
        assert (dds <= 0).all()


class TestTemporalPlots:
    @pytest.fixture(autouse=True)
    def close_figures(self):
        yield
        plt.close('all')

    def test_plot_acf_comparison(self):
        plot_acf_comparison(np.random.randn(200), np.random.randn(200))
        assert plt.gcf().number > 0

    def test_plot_drawdown_comparison(self):
        rets = np.random.randn(500) * 0.01
        plot_drawdown_comparison(rets, rets * 1.5)
        assert plt.gcf().number > 0

    def test_plot_acf_window_comparison(self):
        real = np.random.randn(5, 50)
        synth = np.random.randn(5, 50)
        plot_acf_window_comparison(real, synth, max_lag=5)
        assert plt.gcf().number > 0

    def test_plot_drawdown_window_comparison(self):
        real = np.random.randn(5, 50) * 0.01
        synth = np.random.randn(5, 50) * 0.01
        plot_drawdown_window_comparison(real, synth)
        assert plt.gcf().number > 0
