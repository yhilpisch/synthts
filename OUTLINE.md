# Synthetic Financial Time Series

## Motivation

Generating realistic synthetic financial time series enables rapid
experimentation, backtesting, stress testing, and pedagogical exploration
without exposing proprietary or sensitive market data. The core question
remains:

> Can a generative model learn the statistical fingerprint of real
> financial returns and produce synthetic samples that are hard to
> distinguish from observed return windows under distributional,
> path-level, and temporal diagnostics?

The project now treats this question as a sequence-modelling problem, not
only as a marginal distribution problem. A synthetic return path is useful
only if it reproduces both one-period behaviour and multi-period path
features such as volatility clustering, drawdown depth, cumulative-return
fan shape, and tail risk.

## Current State

The project provides a family of generative models under one library
(`synthts`), each implemented as a self-contained module with model
definitions, training utilities, tests, and a dedicated notebook. The
implemented model stack is now:

1. **Vanilla GAN** for scalar return samples.
2. **WGAN-GP** for more stable scalar return generation.
3. **Sequential GAN** with an LSTM generator, per-step noise, a CNN
   discriminator, and auxiliary statistic matching.
4. **TimeGAN** with embedder, recovery, generator, supervisor, and
   discriminator networks plus staged training.

The sequential and TimeGAN notebooks use window-based diagnostics. Real and
synthetic samples are compared over identical horizons, avoiding the earlier
problem of comparing a full historical path against short independent
synthetic windows.

## Data

- Source: `https://hilpisch.com/eod_data.csv`
- End-of-day prices for multiple instruments
- Log-returns are computed and standardised before training
- Data is downloaded at runtime; no CSV is committed
- Current notebooks focus on GLD return windows

## Project Structure

```text
synthetic/
├── .gitignore
├── OUTLINE.md
├── README.md
├── synthts/
│   ├── __init__.py
│   ├── data.py                 # data loading, returns, scaling
│   ├── evaluate.py             # KS tests, ACF, drawdown, plots
│   ├── vanilla.py              # baseline scalar GAN
│   ├── wgangp.py               # scalar WGAN-GP
│   ├── sequential.py           # LSTM/CNN sequential GAN baseline
│   └── timegan.py              # true TimeGAN-style sequence model
├── notebooks/
│   ├── 01_synthetic_gan.ipynb
│   ├── 02_wgan_gp.ipynb
│   ├── 03_sequential.ipynb
│   └── 04_timegan.ipynb
├── sweeps/
│   ├── config.yml              # sequential sweep config
│   ├── timegan_config.yml      # TimeGAN sweep config
│   ├── count.py                # combination counter
│   ├── sweep_sequential.py
│   ├── sweep_timegan.py
│   ├── sequential_best.yml     # exported current leader, if present
│   ├── timegan_best.yml        # exported current leader, if present
│   └── results/                # ignored sweep result directories
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
└── _ignore/
```

## Implemented Evaluation Contract

The evaluation utilities now distinguish between marginal and path-level
quality:

- Flattened window KS statistic for the marginal return distribution.
- Per-window KS pass rate for equal-horizon real and synthetic windows.
- Return ACF and squared-return ACF, excluding the trivial lag 0 value in
  visualisations and diff scores.
- Cumulative-return fan plots with all paths starting at zero.
- Window-level maximum drawdown distributions.
- Randomised sweep ordering so early trials cover a broader part of the
  parameter space.
- Continuous export of the current sweep leader, not only the final winner.

The current TimeGAN best configuration can produce reasonable KS pass rates,
but visual diagnostics still show that the synthetic path fan is too narrow
and drawdown dispersion is weaker than observed windows. This is the main
open quality issue.

## Sweep Infrastructure

Both sequential and TimeGAN sweeps now follow the same operating pattern:

- Expand all parameter combinations.
- Randomise the task order with `task_order: random`.
- Respect `max_trials` after randomisation.
- Run serially or in parallel through `n_jobs`.
- Use `torch_threads: 1` by default to avoid Apple Silicon oversubscription.
- Write timestamped result directories under `sweeps/results/`.
- Continuously update run-local `best_config.json` after every completed
  trial.
- Continuously export the current notebook-ready leader to
  `sweeps/sequential_best.yml` or `sweeps/timegan_best.yml`.

This means a long-running sweep can be interrupted and the best available
configuration up to that point is still usable by the corresponding notebook.

## Milestones

### M1 — Project scaffold and baseline (complete)

- [x] `.gitignore`
- [x] Directory layout and `OUTLINE.md`
- [x] `synthts/data.py`, `synthts/evaluate.py`
- [x] Vanilla GAN (`synthts/vanilla.py`)
- [x] `notebooks/01_synthetic_gan.ipynb`
- [x] Baseline tests and smoke test

### M2 — WGAN-GP (complete)

- [x] `synthts/wgangp.py` — critic, generator, gradient penalty,
      WGAN-GP training loop
- [x] `notebooks/02_wgan_gp.ipynb`
- [x] `tests/test_wgangp.py`

### M3 — Sequential generation (complete)

- [x] `synthts/sequential.py` — LSTM generator with per-step noise,
      CNN discriminator, statistic matching loss
- [x] `notebooks/03_sequential.ipynb`
- [x] Temporal evaluation utilities in `synthts/evaluate.py`
- [x] Sequential sweep infrastructure and exported best config
- [x] `tests/test_sequential.py`

### M4 — TimeGAN (implemented, quality refinement active)

- [x] `synthts/timegan.py` — embedder, recovery, generator,
      supervisor, discriminator
- [x] Staged TimeGAN training: embedding pretraining, supervisor
      pretraining, joint adversarial training
- [x] `notebooks/04_timegan.ipynb`
- [x] TimeGAN sweep infrastructure and exported best config
- [x] `tests/test_timegan.py`
- [ ] Improved path realism, especially cumulative-return fan width,
      drawdown dispersion, and volatility clustering

### M5 — Extensions (future)

- [ ] Conditional generation through volatility or regime labels
- [ ] Multivariate sequence generation for correlated instruments
- [ ] Richer tail diagnostics: Anderson-Darling, extreme quantiles,
      expected shortfall, tail index estimates
- [ ] Alternative sequence models: temporal convolutional GAN,
      diffusion model, normalising flow, or volatility-conditioned VAE

## Appendix A — Current TimeGAN Limitations

The current TimeGAN implementation is structurally closer to the TimeGAN
paper than the sequential GAN baseline: it has a learned latent embedding,
recovery network, supervisor, generator, and discriminator. However, the
latest visual diagnostics show that a good scalar score is not yet enough for
convincing synthetic financial paths. The main limitations are the following.

### A.1 Path dispersion is too narrow

The synthetic cumulative-return paths tend to cluster more tightly than real
windows. This means the model is learning a central return dynamic but not the
full variability of multi-period outcomes. KS pass rates can look strong even
when the cumulative path fan is too compressed, because KS is dominated by the
one-period marginal distribution.

Recommended improvement: add direct losses for terminal cumulative-return
quantiles and path-level volatility. The model should explicitly match the
distribution of `sum(returns)` over each window, not only the distribution of
individual returns.

### A.2 Drawdown tails are under-represented

The synthetic drawdown histogram is often centred near the real one but lacks
the left-tail mass of deeper real drawdowns. This indicates insufficient
extreme path risk. Matching one-period lower quantiles is not enough; a deep
drawdown is a serial event created by several adverse returns and volatility
states.

Recommended improvement: include a differentiable soft drawdown loss. A hard
maximum drawdown is not smooth, but a soft approximation can be built from
cumulative returns and a temperature-controlled soft maximum. The loss should
match several quantiles of the window drawdown distribution.

### A.3 Volatility clustering remains weak

The squared-return ACF is a central diagnostic for financial time series. The
current statistic loss includes squared-return ACF, but it is not weighted
separately and it is averaged together with other losses. This can allow the
model to improve marginal fit while leaving volatility persistence weak.

Recommended improvement: split the statistic loss into separate components:

- marginal moment loss,
- central quantile loss,
- tail quantile loss,
- return ACF loss,
- absolute-return ACF loss,
- squared-return ACF loss,
- drawdown quantile loss,
- terminal cumulative-return quantile loss.

Each component should have its own weight and each component should be logged
separately in training history and sweep outputs.

### A.4 Latent bottleneck and activation constraints

The current embedder and generator map latent states through `tanh`. This can
stabilise training, but it can also compress latent variation and make tails
hard to reproduce. If the recovery network receives overly compressed latent
states, it can produce paths that are visually too smooth or too narrow.

Recommended improvement: test unbounded latent projections with `LayerNorm`
instead of `tanh`. This keeps latent states numerically controlled without
forcing them into a narrow interval. The sweep should compare bounded and
unbounded latent variants.

### A.5 The discriminator only sees latent states

The current discriminator distinguishes latent sequences. This follows the
TimeGAN spirit, but financial path realism is ultimately judged in observation
space. A discriminator that sees only latent states may not penalise unrealistic
recovered return paths strongly enough.

Recommended improvement: add a second observation-space critic. A compact
1-D CNN or temporal convolutional network can classify recovered synthetic
return windows against real return windows. The generator/recovery path should
then receive both latent adversarial feedback and observation-space feedback.

### A.6 Final epoch is not necessarily the best epoch

Adversarial training often improves and then degrades. The current sweep
tracks the best configuration across trials, but within each trial the final
joint epoch may not be the best epoch for diagnostics.

Recommended improvement: add checkpoint scoring inside `train_timegan`. Every
`250` or `500` joint epochs, generate a validation batch and score it with the
same path-level objective used by the sweep. Restore the best checkpoint at
the end of training and export the best epoch in the config.

## Appendix B — Concrete TimeGAN Improvement Plan

### B.1 Add validation-window checkpoint scoring

Create a validation set of real windows sampled with a fixed seed. During
joint training, evaluate generated windows at regular intervals:

- flattened KS statistic,
- window KS pass rate,
- return ACF diff,
- squared-return ACF diff,
- drawdown mean and drawdown quantile errors,
- terminal cumulative-return quantile error.

The training function should keep the best model state according to the same
composite score used by the sweep. The exported config should include:

- selected epoch,
- validation score,
- score components,
- generation seed,
- validation window seed.

This avoids selecting a strong hyperparameter configuration whose final epoch
has already drifted away from a better earlier checkpoint.

### B.2 Add observation-space adversarial feedback

Extend `synthts/timegan.py` with an observation discriminator, for example:

- input shape `(batch, seq_len, input_dim)`,
- two or three `Conv1d` layers with spectral normalisation,
- LeakyReLU activations,
- pooled sequence output,
- logits rather than sigmoid probabilities.

The joint generator loss would include:

```text
generator loss = latent adversarial loss
               + observation adversarial loss
               + supervised latent loss
               + moment loss
               + financial statistic loss
```

This should directly penalise unrealistic cumulative-return shape and too
narrow synthetic path dispersion.

### B.3 Make financial losses modular

Replace the single `_stat_loss` with a dictionary-returning function such as:

```text
financial_losses(real, fake) -> {
    mean,
    std,
    central_quantile,
    tail_quantile,
    acf,
    abs_acf,
    squared_acf,
    terminal_quantile,
    drawdown_quantile,
}
```

The total statistic loss should be a weighted sum. This makes the sweep more
informative because a poor visual result can be linked to a specific failed
component instead of one opaque statistic.

### B.4 Improve the sweep objective

The current objective rewards KS pass rate and penalises moment, tail, ACF,
and drawdown errors. For TimeGAN, the objective should emphasise path shape
more heavily:

```text
score = marginal errors
      + 2 * tail quantile error
      + 2 * squared-return ACF error
      + 3 * drawdown quantile error
      + 3 * terminal cumulative-return quantile error
      - KS pass rate
```

This will intentionally prefer models that look more convincing in the path
figures, even if their marginal KS statistic is marginally worse.

### B.5 Expand the next sweep around path realism

The next TimeGAN sweep should focus on path dispersion rather than broad
coverage alone. Suggested ranges:

```yaml
hidden_units_values: [96, 128]
latent_dim_values: [24, 32, 48]
noise_dim_values: [16, 32]
lr_values: [0.0002, 0.0003, 0.0005]
supervisor_weight_values: [25.0, 50.0, 100.0]
moment_weight_values: [25.0, 50.0]
stat_weight_values: [10.0, 25.0, 50.0]
stat_max_lag_values: [10, 20]
joint_epochs_values: [1500, 3000]
```

Use randomized ordering and a moderate `max_trials` first, then run longer
only around the strongest region.

### B.6 Add regime or volatility conditioning

Unconditional generation from one GLD series is intrinsically difficult
because the model must learn calm and stressed windows from limited data.
A cleaner approach is to condition on a regime or realised volatility bucket.
The generator and discriminators would receive a condition vector such as:

- low/medium/high realised volatility,
- positive/negative trend regime,
- drawdown regime,
- crisis/normal regime.

The notebook can then compare real and synthetic windows within the same
condition. This is likely to improve both drawdown realism and cumulative
path dispersion.

### B.7 Consider a volatility-residual decomposition

Another robust approach is to separate volatility modelling from residual
shape:

1. Estimate a rolling or exponentially weighted volatility process.
2. Standardise returns by that volatility.
3. Train TimeGAN on standardised residual windows and volatility windows.
4. Recompose returns from generated residuals and generated volatility.

This explicitly teaches the model that large moves arrive through volatility
states, not only through isolated tail draws. It is often more natural for
financial data than asking one unconditional GAN to learn all regimes at once.

## Appendix C — Immediate Next Implementation Steps

The next development wave should be ordered as follows:

1. Add checkpoint scoring and best-epoch restoration to `train_timegan`.
2. Add observation-space discriminator and adversarial loss.
3. Split financial statistic loss into separately weighted components.
4. Update `sweeps/timegan_config.yml` for path-realism weights and larger
   latent/noise dimensions.
5. Update `sweeps/sweep_timegan.py` to record every score component.
6. Keep `04_timegan.ipynb` unchanged structurally, but display the selected
   epoch and score components from `sweeps/timegan_best.yml` when present.

This sequence keeps the current notebook and sweep workflow intact while
making the TimeGAN training objective much closer to the diagnostics that
matter visually.
