# Synthetic Financial Time Series

## Motivation

Generating realistic synthetic financial time series enables rapid
experimentation, backtesting, and pedagogical exploration without
exposing proprietary or sensitive market data. The core question:

> Can a generative model learn the statistical fingerprint of real
> financial returns and produce synthetic samples that are
> indistinguishable from the originals under standard hypothesis tests?

## Approach

The project provides a family of generative models under one library
(`synthts`), each implemented as a self-contained feature module with
its own model definitions, training loop, and dedicated notebook. The
**vanilla GAN** serves as the baseline; WGAN-GP and sequential models
are added as progressively more capable alternatives.

## Data

- Source: `https://hilpisch.com/eod_data.csv`
- End-of-day prices for multiple instruments
- Log-returns are computed and standardised before training
- Data is downloaded at runtime; no CSV is committed

## Project Structure

```
synthetic/
├── .gitignore
├── OUTLINE.md                    # this file
├── synthts/
│   ├── __init__.py
│   ├── data.py                   # data loading, returns, scaling
│   ├── evaluate.py               # KS test, statistics, plots
│   ├── vanilla.py                # baseline GAN (Generator, Discriminator, train)
│   ├── wgangp.py                 # WGAN-GP models and training loop
│   └── sequential.py             # sequential/Temporal GAN models and training
├── notebooks/
│   ├── 01_vanilla_gan.ipynb      # baseline GAN notebook
│   ├── 02_wgan_gp.ipynb          # WGAN-GP notebook
│   └── 03_sequential.ipynb       # sequential generation notebook
├── tests/
│   ├── test_data.py
│   ├── test_evaluate.py
│   ├── test_vanilla.py
│   ├── test_wgangp.py
│   ├── test_sequential.py
│   └── smoke_test.py
├── data/
│   └── .gitkeep
└── _ignore/                      # reference notebook (not tracked)
```

## Roadmap and Improvement Ideas

### 1. Training stability — WGAN-GP

Vanilla GANs with binary cross-entropy are prone to mode collapse and
oscillating losses. Replacing the discriminator with a Wasserstein
critic plus gradient penalty (Arjovsky et al., 2017; Gulrajani et al.,
2017) yields:

- A smooth, monotonically decreasing training-loss curve that
  correlates with sample quality.
- Largely eliminates mode collapse.
- Removes the need for delicate discriminator/generator balance.

### 2. Temporal structure — Sequential generation

The baseline GAN generates i.i.d. scalar returns and cannot reproduce
volatility clustering, autocorrelation, or mean-reversion — three core
stylised facts of financial returns. A sequential generator (LSTM or
1-D CNN) that emits return *sequences* from a single noise vector
addresses this. The TimeGAN architecture (Yoon et al., 2019) adds a
supervised embedding loss alongside the adversarial loss to preserve
stepwise dynamics.

### 3. Conditional generation

A conditional GAN takes side information (target volatility, regime
label, macro factor) as additional input to both generator and
discriminator. This allows drawing synthetic paths "at 20% vol" or
"during a crisis regime" rather than sampling unconditionally.

### 4. Multivariate series

A generator that emits *K*-dimensional vectors per time step preserves
cross-sectional correlation (equity/rates/commodity co-movement),
which matters for portfolio and risk applications.

### 5. Richer statistical evaluation

KS test checks only the marginal distribution. A proper benchmark
should also compare:

- Autocorrelation function (ACF) of returns and squared returns
  (volatility clustering).
- Drawdown depth and duration distributions.
- Tail indices and extreme-value behaviour.
- Maximum Mean Discrepancy (MMD) — kernel-based, sensitive to higher
  moments.
- Anderson-Darling — more sensitive to tail differences than KS.

### 6. Alternative generative families

- **Variational Autoencoders (VAEs)** — easier to train, smooth
  latent space, but blurrier samples.
- **Normalising flows** — exact likelihood, invertible, analytically
  tractable density; good for risk applications needing density
  estimates.
- **Diffusion models** — state-of-the-art sample quality in many
  domains; slower to sample but excellent distribution coverage.

### 7. Practical and engineering

- Learning-rate scheduling and early stopping on evaluation metrics
  (KS p-value or MMD).
- Model checkpointing with reproducibility config (data hash, seed,
  hyperparameters).
- GPU-aware batch sizing.

## Milestones

### M1 — Project scaffold and baseline (complete)

- [x] `.gitignore`
- [x] Directory layout, `OUTLINE.md`
- [x] `synthts/data.py`, `synthts/evaluate.py`
- [x] Vanilla GAN (`synthts/vanilla.py`)
- [x] `notebooks/01_vanilla_gan.ipynb`
- [x] Unit tests and smoke test (49 tests passing)

### M2 — WGAN-GP (complete)

- [x] `synthts/wgangp.py` — Critic (no sigmoid), Generator,
      gradient-penalty computation, WGAN-GP training loop
- [x] `notebooks/02_wgan_gp.ipynb` — WGAN-GP on GLD returns
- [x] `tests/test_wgangp.py`

### M3 — Sequential generation (complete)

- [x] `synthts/sequential.py` — LSTM-based sequential generator
      and discriminator, sequence-level training loop
- [x] `notebooks/03_sequential.ipynb` — sequential GAN on GLD return
      windows, ACF and volatility-clustering evaluation
- [x] `tests/test_sequential.py`
- [x] Temporal evaluation metrics in `synthts/evaluate.py`
      (ACF, squared-return ACF, max drawdown, drawdown bootstrap)
- [x] `tests/test_evaluate.py` extended with temporal tests

### M4 — Extensions (future)

- [ ] Conditional generation (volatility regime conditioning)
- [ ] Multivariate series (correlated instruments)
- [ ] Alternative families (VAE, normalising flows, diffusion)
- [ ] Richer evaluation suite (MMD, Anderson-Darling, tail analysis)
