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
