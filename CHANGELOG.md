# Changelog

## 0.1.0 — Unreleased

The first packaged release provides four trainable generator families and
shared statistical diagnostics for synthetic financial returns.

- Correct dense-model loss histories, initial-wealth drawdown, and TimeGAN
  moment weights; reject invalid horizons and non-finite training.
- Preserve multivariate feature columns during inverse scaling and support
  training-only transformations for future observations.
- Separate chronological training, validation, and test windows in research
  workflows; save input snapshots/provenance and portable inference bundles.
- Align notebook configuration, distinguish BCE/Wasserstein objectives, and
  provide offline example and distribution checks.

Historical sweep metrics have not been reproduced under the corrected
protocol. This release makes no comparative model-performance claim.
