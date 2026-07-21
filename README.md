# SyntheticTS — Realistic Synthetic Financial Time Series

<img src="https://hilpisch.com/tpq_logo.png" alt="The Python Quants" width="35%" align="right" border="0">

**syntheticts** generates realistic synthetic financial time series using
generative adversarial networks. The library ships three progressive
approaches under one package (`synthts`), each with its own models,
training loop, dedicated notebook, and test suite.

## Philosophy

Real financial data is scarce, proprietary, and non-stationary. Synthetic
data that faithfully reproduces the statistical fingerprint of real
returns — marginal distributions, autocorrelation structure, volatility
clustering, and drawdown behaviour — enables rapid experimentation,
backtesting, and pedagogical exploration without exposing sensitive market
data.

The project pursues this goal in four cumulative steps:

1. **Baseline** — a vanilla GAN learns the marginal distribution of
   i.i.d. log-returns. It establishes the training pipeline and
   evaluation framework but cannot reproduce temporal dynamics.

2. **Stabilised training** — replacing the discriminator with a
   Wasserstein critic plus gradient penalty (WGAN-GP) eliminates mode
   collapse, yields a smooth loss curve, and produces higher-quality
   marginal samples.

3. **Sequential generation** — an LSTM-based generator produces return
   *sequences* rather than scalar draws, capturing autocorrelation and
   volatility clustering — two core stylised facts that i.i.d. models
   miss.

4. **TimeGAN** — a learned latent embedding, recovery map, supervisor,
   and adversarial joint training produce paths that better reproduce
   volatility clustering, drawdown dispersion, and tail risk than the
   sequential GAN baseline.

## Quick Start

```python
from synthts.data import load_eod, compute_log_returns, ReturnScaler
from synthts.vanilla import Generator, Discriminator, train_gan, generate_samples

# Load and prepare GLD log-returns
prices = load_eod(symbol='GLD', n_obs=2 * 252)
rets = compute_log_returns(prices)
scaler = ReturnScaler()
rets_ = scaler.fit_transform(rets)

# Train a vanilla GAN
g = Generator(latent_dim=1, hidden_units=24, output_dim=1)
d = Discriminator(input_dim=1, hidden_units=24)
train_gan(g, d, rets_.reshape(-1, 1), epochs=5001, batch_size=32, seed=100)

# Generate synthetic returns
samples = generate_samples(g, len(rets_), latent_dim=1)
synth = scaler.inverse_transform(samples)
```

### WGAN-GP

```python
from synthts.wgangp import WGANGenerator, Critic, train_wgangp, generate_samples

g = WGANGenerator(latent_dim=2, hidden_units=32, output_dim=1)
c = Critic(input_dim=1, hidden_units=32)
train_wgangp(g, c, rets_.reshape(-1, 1),
             epochs=10001, batch_size=64, seed=100)
synth = generate_samples(g, len(rets_), latent_dim=2)
```

### Sequential GAN

```python
from synthts.sequential import (SeqGenerator, SeqDiscriminator,
    train_seq_gan, generate_sequences)

g = SeqGenerator(latent_dim=4, hidden_units=64, output_dim=1,
                 seq_len=63, num_layers=2)
d = SeqDiscriminator(input_dim=1, hidden_units=64, num_layers=2)
train_seq_gan(g, d, rets_.reshape(-1, 1),
              seq_len=63, epochs=5001, batch_size=32, seed=100)
seqs = generate_sequences(g, 25, latent_dim=4)  # (25, 63, 1)
```

### TimeGAN

```python
from synthts.timegan import (
    TimeGANEmbedder, TimeGANRecovery, TimeGANGenerator,
    TimeGANSupervisor, TimeGANDiscriminator,
    train_timegan, generate_timegan_sequences,
)

e = TimeGANEmbedder(input_dim=1, hidden_units=128, latent_dim=24)
r = TimeGANRecovery(latent_dim=24, hidden_units=128, output_dim=1)
g = TimeGANGenerator(noise_dim=32, hidden_units=128, latent_dim=24)
s = TimeGANSupervisor(latent_dim=24, hidden_units=128)
d = TimeGANDiscriminator(latent_dim=24, hidden_units=128)

train_timegan(e, r, g, s, d, rets_.reshape(-1, 1),
              seq_len=63, batch_size=128,
              embedding_epochs=500, supervisor_epochs=500,
              joint_epochs=2000, noise_dim=32, seed=100)
seqs = generate_timegan_sequences(g, s, r, n_sequences=100,
                                   seq_len=63, noise_dim=32, seed=100)
```

### Evaluation

```python
from synthts.evaluate import (ks_test, pass_rate, compare_acf,
    plot_histograms, plot_acf_comparison, plot_drawdown_comparison)

# Marginal distribution
result = ks_test(rets, synth)
print(f"KS p-value: {result['pvalue']:.4f}")

# Temporal structure
comp = compare_acf(rets, synth, max_lag=20)
print(f"ACF diff: {comp['acf_diff']:.4f}")
print(f"Squared-return ACF diff: {comp['acf2_diff']:.4f}")

# Visualisation
plot_histograms(rets, synth)
plot_acf_comparison(rets, synth)
plot_drawdown_comparison(rets, synth)
```

## Data

The canonical data source is:

```
https://hilpisch.com/eod_data.csv
```

End-of-day prices for multiple instruments. Log-returns are computed and
standardised before training. Data is downloaded at runtime; no CSV is
committed to the repository.

## Project Structure

```
synthetic/
├── .gitignore
├── OUTLINE.md                    # roadmap and milestone tracking
├── README.md                     # this file
├── synthts/
│   ├── __init__.py
│   ├── data.py                   # data loading, returns, scaling
│   ├── evaluate.py               # KS test, ACF, drawdown, plots
│   ├── vanilla.py                # baseline GAN
│   ├── wgangp.py                 # WGAN-GP
│   ├── sequential.py             # LSTM-based sequential GAN
│   └── timegan.py                # TimeGAN model + training
├── notebooks/
│   ├── 01_synthetic_gan.ipynb    # baseline notebook
│   ├── 02_wgan_gp.ipynb          # WGAN-GP notebook
│   ├── 03_sequential.ipynb       # sequential generation notebook
│   └── 04_timegan.ipynb          # TimeGAN notebook
├── sweeps/
│   ├── config.yml                # sequential sweep config
│   ├── timegan_config.yml        # TimeGAN sweep config
│   ├── count.py                  # combination counter
│   ├── sweep_sequential.py
│   ├── sweep_timegan.py
│   └── results/                  # ignored sweep result directories
├── tests/
│   ├── test_data.py
│   ├── test_evaluate.py
│   ├── test_vanilla_models.py
│   ├── test_vanilla_train.py
│   ├── test_wgangp.py
│   ├── test_sequential.py
│   ├── test_timegan.py
│   └── smoke_test.py
├── data/
│   └── .gitkeep
├── requirements.txt              # runtime dependencies
├── requirements-dev.txt          # test dependencies (pytest)
└── _ignore/                      # reference notebook (not tracked)
```

## Package Reference

### `synthts.data`

| Symbol | Description |
|--------|-------------|
| `load_eod` | Load EOD price data for a single symbol from CSV URL |
| `compute_log_returns` | Compute log-returns from a price series |
| `ReturnScaler` | Standardise/invert log-returns via `StandardScaler` |

### `synthts.vanilla`

| Symbol | Description |
|--------|-------------|
| `Generator` | Dense generator (noise → synthetic data) |
| `Discriminator` | Dense discriminator (data → probability, sigmoid) |
| `GAN` | Composite module holding generator and discriminator |
| `train_gan` | Vanilla GAN training loop (BCE loss) |
| `generate_samples` | Produce synthetic samples from a trained generator |

### `synthts.wgangp`

| Symbol | Description |
|--------|-------------|
| `WGANGenerator` | Dense generator (same architecture as vanilla) |
| `Critic` | Wasserstein critic (no sigmoid, LeakyReLU) |
| `train_wgangp` | WGAN-GP training loop with gradient penalty |
| `generate_samples` | Produce synthetic samples from a trained generator |

### `synthts.sequential`

| Symbol | Description |
|--------|-------------|
| `SeqGenerator` | LSTM generator (latent vector → return sequence) |
| `SeqDiscriminator` | LSTM discriminator (sequence → probability) |
| `train_seq_gan` | Sequential GAN training loop (overlapping windows) |
| `generate_sequences` | Produce synthetic return sequences |

### `synthts.timegan`

| Symbol | Description |
|--------|-------------|
| `TimeGANEmbedder` | Map observed sequences into latent temporal states |
| `TimeGANRecovery` | Map latent states back to observations |
| `TimeGANGenerator` | Generate latent trajectories from per-step noise |
| `TimeGANSupervisor` | Learn one-step latent dynamics |
| `TimeGANDiscriminator` | Latent-space discriminator returning logits |
| `TimeGANObsDiscriminator` | Optional observation-space CNN critic |
| `train_timegan` | Staged TimeGAN training (embed, supervise, joint) |
| `generate_timegan_sequences` | Produce synthetic return sequences |
| `compute_financial_losses` | Dict of individually weighted financial losses |

### `synthts.evaluate`

| Symbol | Description |
|--------|-------------|
| `ks_test` | Two-sample Kolmogorov-Smirnov test |
| `ks_test_batch` | KS test across multiple synthetic columns |
| `pass_rate` | Fraction of p-values above significance level |
| `describe_comparison` | Descriptive statistics for real vs synthetic |
| `mse_sorted` | MSE between sorted real and synthetic data |
| `acf` | Autocorrelation function |
| `compare_acf` | Compare ACF and squared-return ACF (real vs synthetic) |
| `max_drawdown` | Maximum drawdown from a return array |
| `drawdown_distribution` | Bootstrap distribution of max drawdown |
| `plot_histograms` | Histogram overlay (real vs synthetic) |
| `plot_sorted` | Sorted-value comparison (QQ-style) |
| `plot_cumulative` | Cumulative return paths (single synth legend) |
| `plot_pvalue_histogram` | KS p-value histogram with significance line |
| `plot_acf_comparison` | Side-by-side ACF and squared-return ACF |
| `plot_drawdown_comparison` | Max-drawdown bootstrap histogram overlay |

## Default Settings

| Parameter | Vanilla | WGAN-GP | Sequential | TimeGAN |
|-----------|---------|---------|------------|---------|
| Hidden units | 32 / 24 | 32 | 64 | 128 |
| Latent dim | 1 | 2 | 4 | 24 |
| Noise dim | — | — | 2 | 32 |
| Sequence length | — | — | 63 (3-month) | 63 (3-month) |
| Batch size | 32 | 64 | 32 | 128 |
| Epochs / joint | 5001 | 10001 | 5001 | 2000 (after 500 + 500) |
| LR (G) | 0.0001 | 0.0001 | 0.0001 | 0.0005 |
| LR (D/C) | 0.001 | 0.001 | 0.001 | 0.0005 (joint) |
| n_critic | — | 1 (SN+BCE) | — | — |
| λ_GP | — | 10.0 (GP mode) | — | — |
| Critic norm | — | Spectral norm | Spectral norm (CNN) | Spectral norm (obs-D) |
| LSTM / GRU layers | — | — | 2 | 1 (GRU) |
| Activation (G) | ReLU | ReLU | — | tanh or layernorm |
| Activation (D/C) | ReLU / Sigmoid | LeakyReLU / Sigmoid | Sigmoid (last) | — |
| Optimiser betas | default | (0.5, 0.9) | default | default |
| LR schedule | none | none | none | cosine (joint) |
| Seed | 100 | 100 | 100 | 100 |

## Testing

```bash
pytest tests/ -v
```

The test suite covers model shapes, gradient flow, training weight updates,
round-trip scaling, statistical tests, all plotting functions, and
end-to-end smoke tests. 131 tests currently pass.

## Requirements

- Python >= 3.10
- `torch` >= 2.0
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`
- `tqdm`, `pyyaml` (sweep CLI)

Install pinned minimum versions with:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # adds pytest, pytest-cov
```

## Author

Dr. Yves J. Hilpisch — [https://tpq.io](https://tpq.io) |
[@dyjh](https://twitter.com/dyjh) |
[team@tpq.io](mailto:team@tpq.io)

Research, structuring, drafting, and visualizations were assisted by
GPT 5.x as a co-writing tool under human direction.

## Disclaimer

This project is provided for educational and informational purposes only
and does not constitute investment advice, an offer, or a solicitation to
buy or sell any security or financial instrument. The analysis relies on
data and assumptions that may be incomplete, simplified, or subject to
revision, and past performance or historical relationships are not reliable
indicators of future results. Any investment decision should be based on
the reader's own judgment and, where appropriate, on independent advice
from a qualified financial professional who is familiar with the investor's
individual objectives, constraints, and risk tolerance.

&copy; Dr. Yves J. Hilpisch | The Python Quants GmbH |
[https://tpq.io](https://tpq.io)
