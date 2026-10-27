# Publication Readiness Review

Reviewed 2026-10-27 against local commit `264ce11`. Scope: inventory,
source inspection, executable quality checks, and the GitHub/PyPI release
path. Implementation is in progress; only unresolved findings remain below.
The numerical correction stage now has regression coverage.
Remove resolved items after their acceptance checks pass.

**Assessment:** a working research codebase, but not yet an installable
distribution. Packaging and reproducibility work remains before release.
Resolve the release blockers before publishing a first package version.

## Current inventory

The repository contains 35 tracked files. The following components form the
maintained package and its supporting examples.

| Component | Contents and publication implications |
|---|---|
| Import package | `synthts`: seven modules, including `__init__.py`; explicit top-level exports; no package version. |
| Data utilities | CSV price loading, log-return calculation, and standardisation/inversion. |
| Generators | Dense vanilla GAN; spectral-normalised BCE/Wasserstein-GP alternatives; LSTM sequential GAN with CNN discriminator; extended TimeGAN with modular financial losses and optional observation discriminator. |
| Evaluation | Marginal statistics, KS diagnostics, return/squared-return ACFs, window comparisons, drawdown statistics, and Matplotlib plots. |
| Examples | Four notebooks. The first three have no stored outputs; TimeGAN has 16 output objects, including six embedded PNGs. |
| Sweep utilities | Two training CLIs, a combination counter, two grid configurations, and three selected/reference configurations. Sequential grid: 24 trials; TimeGAN grid: 32 combinations capped at 24. |
| Tests | Eight test-bearing files plus `tests/__init__.py`; 139 collected tests. |
| Automation | One GitHub Actions workflow testing checkout imports on Linux/Python 3.10–3.12. No distribution build or release workflow. |
| Documentation/dependencies | README, review ledger, runtime and development requirements. No licence, changelog, release instructions, or build metadata. |
| Local research material | Ignored `_ignore/` contains two reference files. Ignored `sweeps/results/` contains 24 run directories/62 files. Preserve these; they are not present in a fresh clone. |
| Git state | Clean at audit start; local `main`, no configured remote, no tags. `git fsck --no-dangling` completed without findings. |

## Release blockers

These items prevent a reliable first release or undermine its documented
behaviour. Each requires a focused regression or release check.

### 1. Establish distribution identity, licensing, and packaging

There is no `pyproject.toml`, `setup.py`, or `setup.cfg`.
`pip wheel --no-deps --no-build-isolation .` fails because the directory is
not installable. The README uses `SyntheticTS`/`syntheticts`, whereas Python
imports use `synthts`; the distribution name has not been established in
metadata. PyPI name availability and account ownership were not checked.

Choose the distribution name, first version, licence, and GitHub destination.
Add a build backend and metadata covering Python support, dependencies,
README, licence, author, and project URLs. Explicitly select package contents
so flat-layout discovery cannot accidentally include tests, sweep utilities,
or local research material. A copyright notice/disclaimer is not a licence.
Keep runtime requirements and package metadata consistent; consider moving
sweep-only dependencies into an optional extra.

Acceptance: build both wheel and source distribution, run `twine check`,
inspect archive contents, rebuild from the source distribution, and test
imports/examples from an installed wheel outside the checkout. Do not
require a directory reorganisation solely to add packaging.

### 6. Align TimeGAN sweep configuration and notebook execution

`04_timegan.ipynb` reads `timegan_best.yml` but ignores the selected
`latent_activation`, scheduler, checkpoint interval, modular loss weights,
and observation discriminator. The selected configuration specifies
`layernorm`; the notebook constructs the default `tanh` models and omits
the observation discriminator entirely. Its outputs therefore cannot be
treated as a reproduction of the selected sweep trial.

The sweep itself exposes `obs_disc_channels`/`obs_disc_layers` in defaults
and YAML but hardcodes 32/two layers in `run_trial`; the notebook export
also omits that architecture. Make the configuration contract complete
across construction, training, export, and notebook loading. Test a
non-default round trip and refresh stored outputs only after alignment.

### 7. Repair public examples and objective descriptions

The README sequential example raises `TypeError`: `SeqDiscriminator` has
no `num_layers` argument. Its reference table also describes that CNN as
an LSTM discriminator. The tree refers to nonexistent `OUTLINE.md`.
The settings table differs from the configurations actually loaded by
the notebooks; clarify whether it describes an explicit example or API
defaults. Requirements are lower bounds/ranges, not pinned environments.

`02_wgan_gp.ipynb` advertises Wasserstein gradient-penalty training but
explicitly uses `n_critic=1`, which selects spectral-normalised BCE.
README/reference/docstring descriptions need to distinguish these modes
consistently. Preserve the intentional dual-mode implementation unless an
API change is explicitly chosen. Explain that an `epoch` here is one
sampled minibatch update, not a full pass through the data.

Execute every advertised example at a reduced training budget against the
installed package. The first two notebooks and README seed the training
loop after model construction; seed before construction as well when
claiming reproducibility. Add a clear installation path and a short offline
example before asking users to run thousands of training steps.

## Publication evidence and reproducibility

The original review's long-run validation and data-provenance concerns
remain relevant, with the following concrete requirements.

### 8. Separate calibration, model selection, and held-out evaluation

Both sweep CLIs fit the scaler and train on the complete return series,
then sample evaluation windows from that same series. TimeGAN's internal
checkpoint `val_x` is also sampled from training windows. These are
in-sample calibration scores, despite the validation terminology.

For generalisation claims, split chronologically before scaling/windowing,
fit preprocessing on training data, keep windows within split boundaries,
and reserve independent validation/test periods. Compare fixed budgets and
multiple seeds using predefined metrics and simple resampling baselines.
Sweep seeds currently vary with configuration/trial ID, which confounds
configuration effects with seed effects. Retain a final test evaluation
outside hyperparameter selection.

KS non-rejection is not evidence of equivalent distributions or realistic
paths. Serial dependence and overlapping sampled windows compromise the
usual independent-sample p-value interpretation. Label current pass rates
as descriptive diagnostics; use effect sizes and dependence-aware
uncertainty for inferential claims. The sequential notebook's statements
about stable training/realistic statistics need supporting evidence or
more qualified wording.

### 9. Freeze data and retain complete experiment contracts

Examples/sweeps download `https://hilpisch.com/eod_data.csv` without a
retrieval date or checksum. Record source, date range, symbol, price
definition/adjustments, retrieval date, checksum, transformations, schema,
and redistribution terms. Provide an offline input path. If bundling a
snapshot, use CSV plus a same-stem `.meta.json`; the current `*.csv` ignore
rule will need a narrowly scoped exception.

The three selected/reference YAML files point to ignored result directories
that exist locally but are absent from a fresh clone. Treat their recorded
metrics as historical observations, not newly verified benchmarks. Preserve
the runs and identify which, if any, should back publication claims.
The sweep CLIs write metrics/configuration but do not persist trained
weights, fitted preprocessing, input checksums, or environment versions.
TimeGAN checkpointing is only in-memory weight selection, not resumable
training or a portable inference artefact.

For any promoted model/result, retain architecture, weights, preprocessing,
data identity, seeds, training settings, dependency versions, evaluation
protocol, and metrics together. Store model binaries through the asset or
release pipeline, not Git. Verify save/load inference equivalence.

## Additional quality work

These items should be addressed alongside release preparation, with scope
kept proportional to the public API.

### 10. Define boundary behaviour for data/evaluation helpers

`acf` returns different lengths for constant versus varying short input
under the same `max_lag`; comparing arrays of lengths three/five at the
default lag raises a broadcasting error. Define supported lag/length
rules consistently across metrics and plots. Specify empty/non-finite
input handling and equal-length requirements for `mse_sorted`.

`ReturnScaler` accepts multi-column data but flattens inverse output;
either document/enforce a univariate contract or preserve feature shape.
Define missing-price adjacency semantics in `compute_log_returns` and
positive `n_obs` handling in `load_eod`. `ks_test_batch` accepts an unused
`alpha` parameter. Resolve these contracts before stabilising the API.

### 11. Strengthen automation around actual release artefacts

Keep the fast offline unit suite, but add the regressions above, sweep
configuration/export checks, notebook/example smoke execution, and installed
wheel/source-distribution checks. The present CI does not cover those paths
or verify minimum dependency versions. Declare supported Python versions
and test that matrix; Python >=3.10 alone does not establish compatibility.
Test supported dependency combinations in clean environments; current
requirements are not a reproducibility lock.

Default Ruff checking reports 24 findings: six sweep import-order findings
from checkout path injection, 15 semicolon findings, and three unused
imports. Adopt a small explicit lint configuration and remove or justify
findings; adding installable packaging can remove the path-injection need.
Preserve the project's intentional annotation/default spacing convention.

Sweep settings silently ignore a missing configuration file, permit empty
grids/non-finite rankings, and export selected settings into tracked YAML
during execution. Validate configuration paths/ranges and finite scores,
and make promotion of selected settings deliberate. Cover one serial and
one multiprocessing CLI path with small deterministic local inputs.

### 12. Complete the GitHub-to-PyPI release path

After the correctness and packaging gates pass, configure the intended
GitHub remote, version/tag policy, concise changelog, support/contribution
instructions, and release procedure. Use a TestPyPI rehearsal to verify
metadata and installation before a production upload. A tag-triggered
workflow should build/check distributions and publish those same tested
artefacts; PyPI Trusted Publishing avoids storing a long-lived upload token.
Repository/account setup and publication remain future work in this audit.

The release checklist should follow the official
[PyPA packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
and [PyPI Trusted Publishing documentation](https://docs.pypi.org/trusted-publishers/).

## Validation and remaining limits

Numerical correction stage: **154 tests passed** on the workspace Python
3.12 interpreter. New regressions cover both dense training objectives,
initial-loss/extreme-path drawdown, effective mean/std weights, invalid
horizons/settings, and non-finite loss/gradient rejection. Existing TimeGAN
checkpoint tests now require finite scores and the minimum-score epoch.
Historical metrics and stored notebook outputs have not been recomputed.

Full training, held-out benchmarks, installed-distribution tests, clean
environments, hosted CI, dataset rights, account setup, and publication
remain unvalidated. The original audit's inventory describes its starting
checkout; new modules/tests are being added in the correction stages.
