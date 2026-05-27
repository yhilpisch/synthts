# Synthetic Financial Time Series

## Motivation

Generating realistic synthetic financial time series enables rapid
experimentation, backtesting, and pedagogical exploration without
exposing proprietary or sensitive market data. The core question:

> Can a generative model learn the statistical fingerprint of real
> financial returns and produce synthetic samples that are
> indistinguishable from the originals under standard hypothesis tests?

## Approach

A **Generative Adversarial Network (GAN)** is trained on real log-return
data. The generator maps random noise to synthetic returns; the
discriminator learns to separate real from generated samples. Training
continues until the generator produces data that consistently passes the
Kolmogorov-Smirnov (KS) test against the real distribution.

The reference implementation (TensorFlow/Keras) is migrated to
**PyTorch**, with the training loop, model definitions, and evaluation
procedures refactored into reusable Python modules.

## Data

- Source: `https://hilpisch.com/eod_data.csv`
- End-of-day prices for multiple instruments
- Log-returns are computed and standardised before training
- Data is downloaded at runtime; no CSV is committed

## Project Structure

```
synthetic/
├── .gitignore
├── OUTLINE.md                # this file
├── code/
│   ├── __init__.py
│   ├── data.py               # data loading, return computation, scaling
│   ├── models.py             # Generator, Discriminator, GAN module
│   ├── train.py              # training loop and utilities
│   └── evaluate.py           # KS test, statistics, visualisation
├── notebooks/
│   └── 01_synthetic_gan.ipynb  # migrated & PyTorch-based notebook
├── data/
│   └── .gitkeep
└── _ignore/                  # reference notebook (not tracked)
```

## Milestones

### M1 — Project scaffold (current)

- [x] `.gitignore`
- [x] Directory layout (`code/`, `notebooks/`, `data/`)
- [x] `OUTLINE.md`

### M2 — PyTorch migration and modularisation

- [ ] `code/data.py` — data loading, log-return computation,
      `StandardScaler` wrapping
- [ ] `code/models.py` — `Generator`, `Discriminator`, `GAN` PyTorch
      modules
- [ ] `code/train.py` — GAN training loop with configurable
      hyperparameters, progress logging
- [ ] `code/evaluate.py` — KS test, descriptive statistics, comparison
      plots
- [ ] `notebooks/01_synthetic_gan.ipynb` — full PyTorch notebook
      covering the simple (cubic function) and financial examples,
      importing from `code/`

### M3 — Extensions (future)

- [ ] Conditional generation (e.g. by volatility regime)
- [ ] Multivariate synthetic series (correlated instruments)
- [ ] Wasserstein GAN with gradient penalty (WGAN-GP)
- [ ] Temporal GAN (generating return sequences, not i.i.d. samples)
- [ ] Additional statistical tests (Anderson-Darling, maximum mean
      discrepancy)
