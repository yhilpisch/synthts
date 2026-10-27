# Publication Readiness Review

Reviewed 2026-10-27 against local commit `264ce11`. Scope: inventory,
source inspection, executable quality checks, and the GitHub/PyPI release
path. This is an audit; the implementation findings below remain open.
Remove resolved items after their acceptance checks pass.

**Assessment:** a working research codebase, but not yet an installable
distribution. Passing tests currently conceal several correctness defects.
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

### 2. Repair loss histories in both dense training loops

`train_gan` and both branches of `train_wgangp` initialise history lists but
never append losses. Two-step runs return three empty lists. Current tests
check that a dictionary exists and weights change, so they miss the defect.

Return one finite observation per training step for each documented series.
For Wasserstein mode, define history keys that accurately describe critic
loss/distance rather than inventing BCE-style values. Test lengths, finite
values, and both training objectives.

### 3. Correct maximum drawdown before using it to rank models

`evaluate.max_drawdown` starts the running peak after the first return,
omitting initial wealth. It returns `0.0` for log-returns `[-0.10]` and
`[-0.10, 0.05]`; both have maximum drawdown
`exp(-0.10) - 1 = -0.09516258196404043` from initial wealth 1.
The defect propagates into window diagnostics and sweep scores.

Include initial wealth in the peak calculation; specify log-return units
and the negative drawdown convention. Test initial losses, recovery paths,
and exact known values. Preserve historical results as historical; corrected
metrics need a new, explicitly identified evaluation.

### 4. Make TimeGAN mean/std weights effective

`train_timegan` accepts `mean_weight` and `std_weight`, and sweep exports
record them, but the objective uses the unweighted sum of mean/std losses.
Identically initialised two-step runs with both weights set to 0 versus 999
produce identical loss histories and all model tensors.

Define how component weights combine with `moment_weight`, implement that
contract, and test that changing each weight changes its contribution to
the objective/gradient. Keep default behaviour deliberate and documented.

### 5. Reject invalid horizons and prevent silent non-finite training

The Torch ACF helpers in `sequential.py` and `timegan.py` average empty
slices when a lag reaches the sequence length. A TimeGAN run with
`seq_len=10` and default `stat_max_lag=10` returns `g_loss=[nan]`.
Several existing TimeGAN tests use this combination and assert only keys
or weight changes; checkpoint tests explicitly tolerate non-finite scores.

Validate `1 <= stat_max_lag < seq_len` when temporal losses are evaluated,
and validate sequence/model dimensions, non-empty finite input, positive
batch sizes, valid training budgets/intervals, and schedule choices.
TimeGAN supervision requires at least two time steps. Reject unsupported
values clearly rather than silently falling back or returning corrupt
models. Add finite loss, gradient, parameter, and generated-output checks
to meaningful short training tests. Disabled loss weights must not allow
`0 * nan` to contaminate the objective.

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

## Validation baseline and limits

These results describe the inspected checkout, not a released distribution.
Local validation used the workspace Python interpreter
`/Users/yves/Temp/pyai/bin/python` on macOS, Python 3.12.12, Torch 2.10.0,
NumPy 1.26.4, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.8.0, and
Matplotlib 3.10.8.

- `python -m pytest tests/ -q --cov=synthts --cov-report=term-missing`:
  **139 passed**, **98% statement coverage**. Coverage does not establish
  finite training or scientific validity; see the confirmed defects above.
- All 19 Python files parsed. All four notebooks passed `nbformat` schema
  validation and IPython-transformed code-cell syntax checks. No Python or
  notebook code lines exceed 80 characters; notebook cells have no leading
  or trailing empty lines.
- Both sweep counters ran. One tiny in-process trial per sweep completed
  with finite numeric result fields on seeded synthetic input; these were
  interface checks, not financial benchmarks. No full sweep was launched.
- Targeted probes reproduced empty histories, incorrect initial-loss
  drawdown, the README constructor error, short-series ACF failure,
  non-finite short-horizon TimeGAN loss, and ineffective mean/std weights.
- Wheel creation failed as expected from missing packaging metadata.
  Default Ruff check failed with the 24 findings above.
- `pip check` found two unrelated conflicts in the shared environment:
  `aider-chat` requires markdown-it-py 4.0.0/Pillow 12.1.1; installed versions
  are 4.2.0/12.3.0. Use a clean release environment rather than changing
  unrelated workspace packages.
- A limited literal scan of tracked files found no private-key header or
  AWS/GitHub/PyPI token prefixes checked. This was not a full history or
  credential audit.

Full notebook training, historical-result reproduction, remote dataset
availability/rights, clean-environment compatibility, hosted CI, GitHub/PyPI
account state, and TestPyPI installation were not validated. No source,
notebook, configuration, reference, or historical-result files were changed
by this review, and nothing was pushed or published.
