# synthts — Synthetic Financial Time Series

<img src="https://hilpisch.com/tpq_logo.png" alt="The Python Quants" width="35%" align="right" border="0">

**synthts** provides generative adversarial networks (GANs) and diagnostics for
synthetic financial returns. The library ships four progressive
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

2. **Stabilised training** — a spectral-normalised critic with binary
   cross-entropy is the low-dimensional default, while a multi-critic mode
   uses Wasserstein loss with gradient penalty (WGAN-GP). Both configurations
   are designed to reduce training instability and mode collapse.

3. **Sequential generation** — a long short-term memory (LSTM) generator produces return
   *sequences* rather than scalar draws, targeting autocorrelation and
   volatility clustering — two core stylised facts that i.i.d. models
   miss.

4. **TimeGAN** — a learned latent embedding, recovery map, supervisor,
   and adversarial joint training target volatility clustering, drawdown
   dispersion, and tail risk beyond the sequential GAN baseline.

## Installation

From this checkout, install the package before running examples:

```bash
python -m pip install .
python -m pip install '.[dev]'  # notebooks, sweeps, and validation tools
```

The distribution and import package are both named `synthts`. Version
`0.1.0` is being prepared; PyPI publication is pending. Python 3.10–3.12
form the configured CI matrix. Dependency lower bounds are compatibility
constraints, not a frozen research environment.

## Google Colab

Open a notebook and select **Runtime → Run all** on a standard CPU runtime.
The first code cell installs `synthts` and missing dependencies, using public
GitHub `main` before PyPI publication. Existing installations are reused;
local checkout execution prefers the local package.

| Notebook | Open in Colab |
|---|---|
| Vanilla GAN | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yhilpisch/synthts/blob/main/notebooks/01_synthetic_gan.ipynb) |
| Spectral-normalised GAN/WGAN-GP | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yhilpisch/synthts/blob/main/notebooks/02_wgan_gp.ipynb) |
| Sequential GAN | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yhilpisch/synthts/blob/main/notebooks/03_sequential.ipynb) |
| TimeGAN | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yhilpisch/synthts/blob/main/notebooks/04_timegan.ipynb) |

These links require the notebooks/package on public `main`. No Google Drive
mount, API credentials, or uploaded files are required. Financial examples
download public end-of-day prices; frozen local input and an explicit YAML
configuration remain optional. The existing training budgets are retained;
training can take several minutes on CPU. The small networks use one CPU
thread for predictable runtimes. All four notebooks have passed full local
CPU execution with public data and unchanged training budgets; hosted Colab
runs remain to be confirmed.

Run `python scripts/validate_colab.py` for notebook/setup checks. Add
`--execute` to run every notebook unchanged with plotting in fresh kernels,
using the current interpreter and a temporary installation of this checkout.
Add `--report /tmp/synthts-colab-report.json` to record timings, versions,
notebook hashes, and failures. Full execution retains every training step;
it leaves source notebooks and the current environment unchanged.

## Offline example

This small synthetic-input run checks the API without a market-data download.
It is illustrative and makes no claim about financial realism.

```python
import numpy as np
import torch
from synthts.data import ReturnScaler
from synthts.vanilla import (
    Generator, Discriminator, train_gan, generate_samples,
)

returns = np.random.default_rng(100).normal(0, 0.01, 256)
scaler = ReturnScaler()
scaled = scaler.fit_transform(returns)
torch.manual_seed(100)
g = Generator(hidden_units=8)
d = Discriminator(hidden_units=8)
history = train_gan(g, d, scaled, epochs=10, batch_size=16, seed=100)
synthetic = scaler.inverse_transform(generate_samples(g, 100))
assert synthetic.shape == (100,)
assert np.isfinite(history['g_loss']).all()
```

## Quick Start

```python
import os
import torch
from synthts.data import EOD_URL, load_eod, compute_log_returns, ReturnScaler
from synthts.vanilla import (
    Generator, Discriminator, train_gan, generate_samples,
)

# Load and prepare GLD log-returns
source = os.environ.get('SYNTHTS_DATA', EOD_URL)
prices = load_eod(url=source, symbol='GLD', n_obs=2 * 252)
rets = compute_log_returns(prices)
scaler = ReturnScaler()
rets_ = scaler.fit_transform(rets)

# Train a vanilla GAN
torch.manual_seed(100)
g = Generator(latent_dim=1, hidden_units=24, output_dim=1)
d = Discriminator(input_dim=1, hidden_units=24)
train_gan(g, d, rets_.reshape(-1, 1), epochs=5001, batch_size=32, seed=100)

# Generate synthetic returns
samples = generate_samples(g, len(rets_), latent_dim=1)
synth = scaler.inverse_transform(samples)
```

### WGAN-GP

The variants below reuse the standardised returns from the quick start.

```python
from synthts.wgangp import WGANGenerator, Critic, train_wgangp, generate_samples

torch.manual_seed(100)
g = WGANGenerator(latent_dim=2, hidden_units=32, output_dim=1)
c = Critic(input_dim=1, hidden_units=32)
train_wgangp(g, c, rets_.reshape(-1, 1),
             epochs=10001, batch_size=64, seed=100)
synth = scaler.inverse_transform(
    generate_samples(g, len(rets_), latent_dim=2))
```

### Sequential GAN

```python
from synthts.sequential import (SeqGenerator, SeqDiscriminator,
    train_seq_gan, generate_sequences)

torch.manual_seed(100)
g = SeqGenerator(latent_dim=4, hidden_units=64, output_dim=1,
                 seq_len=63, num_layers=2)
d = SeqDiscriminator(input_dim=1, hidden_units=64)
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

torch.manual_seed(100)
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

The default example data source is:

```
https://hilpisch.com/eod_data.csv
```

End-of-day prices for multiple instruments. Log-returns are computed and
standardised before training. Examples download data unless a local CSV
is supplied. Sweeps retain local snapshots and provenance; no market-data CSV is committed to Git.

## Project Structure

```
synthetic/
├── .gitignore
├── README.md                     # usage and API reference
├── pyproject.toml                # packaging and dependency metadata
├── RELEASE.md                    # release validation and publishing
├── CHANGELOG.md                  # version changes
├── synthts/
│   ├── __init__.py
│   ├── artifacts.py              # portable inference bundles
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
│   ├── common.py                 # snapshots, splits, and provenance
│   ├── count.py                  # combination counter
│   ├── sweep_sequential.py
│   ├── sweep_timegan.py
│   └── results/                  # ignored sweep result directories
├── scripts/                      # distribution inspection
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
| `load_eod` | Load one price column from a CSV URL/local path |
| `split_returns` | Split returns chronologically before scaling/windowing |
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
| `Critic` | Spectral-normalised critic returning logits/scores |
| `train_wgangp` | BCE when n_critic=1; Wasserstein/GP when n_critic>1 |
| `generate_samples` | Produce synthetic samples from a trained generator |

### `synthts.sequential`

| Symbol | Description |
|--------|-------------|
| `SeqGenerator` | LSTM generator (latent vector → return sequence) |
| `SeqDiscriminator` | Convolutional neural network (CNN) discriminator |
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

### `synthts.artifacts`

| Symbol | Description |
|--------|-------------|
| `save_bundle` | Save named components, preprocessing, and provenance |
| `load_bundle` | Restore supported components with restricted Torch loading |

### `synthts.evaluate`

| Symbol | Description |
|--------|-------------|
| `ks_test` | Two-sample Kolmogorov-Smirnov test |
| `ks_test_batch` | KS test across multiple synthetic columns |
| `pass_rate` | Fraction of p-values above significance level |
| `describe_comparison` | Descriptive statistics for real vs synthetic |
| `mse_sorted` | MSE between sorted real and synthetic data |
| `acf` | Autocorrelation function (ACF) |
| `compare_acf` | Compare ACF and squared-return ACF (real vs synthetic) |
| `max_drawdown` | Maximum drawdown from a return array |
| `drawdown_distribution` | Bootstrap distribution of max drawdown |
| `plot_histograms` | Histogram overlay (real vs synthetic) |
| `plot_sorted` | Sorted-value comparison (QQ-style) |
| `plot_cumulative` | Cumulative return paths (single synth legend) |
| `plot_pvalue_histogram` | KS p-value histogram with significance line |
| `plot_acf_comparison` | Side-by-side ACF and squared-return ACF |
| `plot_drawdown_comparison` | Max-drawdown bootstrap histogram overlay |

## Training and evaluation contracts

Every `epoch` is one sampled minibatch update, not a complete pass through
the data. `train_wgangp` uses spectral-normalised binary cross-entropy for
`n_critic=1`; `n_critic>1` selects Wasserstein loss with gradient penalty.
Both modes are intentional. Wasserstein history contains `critic_loss`,
`wasserstein_distance`, and `gradient_penalty`; BCE history contains
`d_loss_real` and `d_loss_fake`. Both contain `g_loss`.

Seed Torch before model construction as well as passing a training seed.
Use `1 <= stat_max_lag < seq_len` for temporal losses. Invalid data/settings
raise `ValueError`; non-finite training objectives/gradients raise
`FloatingPointError`. For TimeGAN, `moment_weight` multiplies
`mean_weight * mean_loss + std_weight * std_loss`.

The quick-start examples demonstrate calibration on one sample. The
sequential/TimeGAN notebooks and sweeps split returns chronologically
before windowing, fit scaling on training data, use validation data for
selection, and reserve the final test period. Do not repeatedly tune on
test diagnostics. KS pass rates are descriptive: non-rejection does not
establish equivalence, and dependent/overlapping windows invalidate the
usual independent-sample p-value interpretation. Maximum drawdown uses
log-returns, includes initial wealth 1, and is negative or zero.

`ReturnScaler.transform` applies training-only preprocessing to future
returns. Inversion returns a vector for one feature and preserves columns
for multiple features. `compute_log_returns` omits both returns adjacent
to a missing price rather than bridging the gap. `load_eod` drops gaps in
the selected price column before computing returns; choose your missing-data
policy explicitly when supplying a price series directly. ACF routines
require `max_lag < len(series)` and return zeros for constant series.

## Reproducible research runs

Install the development extra and run the sweep modules from the checkout:

```bash
python -m sweeps.count --config sweeps/config.yml
python -m sweeps.sweep_sequential --data-path /path/to/frozen.csv
python -m sweeps.sweep_timegan --data-path /path/to/frozen.csv
```

Set grids/budgets and split fractions in the YAML configuration before
launching a full run. Sequential/TimeGAN CLIs retain the exact input CSV,
its metadata/checksum, effective settings, validation metrics, and an
inference bundle for every trial under a unique ignored run directory.
All configurations share the run seed; repeat whole grids under predefined
seeds when comparing models. The final test partition is unused by sweeps.
Source rights and adjusted-price definitions must be confirmed before
redistributing a real dataset. No canonical real-data benchmark is claimed
for this release.

Selected settings are exported only when explicitly enabled in the config
with `export_on_complete: true` and an `export_best` destination. Existing
`*_best.yml`/`timegan_ref.yml` files contain historical settings/metrics;
new runs do not overwrite or load them automatically. To use a selected
configuration in a notebook, set `SYNTHTS_CONFIG` to its explicit path.
Set `SYNTHTS_DATA` to a frozen local CSV for notebook/README examples.

Inference bundles contain component constructors/weights, fitted scaling,
and provenance. Use `save_bundle`/`load_bundle` from `synthts.artifacts`.
They restore evaluation-mode components and preprocessing; they do not
resume optimiser state. Store model binaries outside Git.

## Testing

```bash
pytest tests/ -v
```

The test suite covers model shapes, gradient flow, training weight updates,
round-trip scaling, statistical tests, all plotting functions, and
end-to-end offline examples, numerical regressions, configuration round
trips, serial/multiprocessing sweeps, and portable inference bundles.

## Requirements

- Python >= 3.10
- `torch` >= 2.0
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`
- `tqdm`, `pyyaml` (sweep CLI)

Install dependency ranges with:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # development/notebook/sweep tools
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for changes/support and
[RELEASE.md](RELEASE.md) for candidate validation and publication.

## Method references

The dense models follow the adversarial framework of
[Goodfellow et al. (2014)](https://papers.nips.cc/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html).
The Wasserstein/gradient-penalty mode draws on
[Gulrajani et al. (2017)](https://arxiv.org/abs/1704.00028).
TimeGAN uses the embedding/supervision approach of
[Yoon, Jarrett, and van der Schaar (2019)](https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html).
The financial losses, observation critic, and spectral-normalised BCE
configuration are project-specific variants; the examples do not claim to
reproduce the papers' architectures or reported benchmarks exactly.

## Author

Dr. Yves J. Hilpisch — [https://tpq.io](https://tpq.io) |
[@dyjh](https://twitter.com/dyjh) |
[team@tpq.io](mailto:team@tpq.io)

Research, structuring, drafting, and visualizations were assisted by different
LLMs as co-writing tools under human direction.

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
