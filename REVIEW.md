# Remaining Publication Work

Updated 2026-10-28. The distribution/import name is **synthts**; candidate
version **0.1.0** is prepared locally. Only unresolved work is listed below.

The maintained inventory comprises ten package modules, four notebooks,
research sweep/configuration utilities, regression/example/CLI tests,
standard packaging metadata, archive inspection scripts, contribution/release
instructions, and CI/Trusted Publishing workflows. Historical selected YAML
and ignored research runs are preserved. New sweeps retain exact input CSVs,
metadata/checksums, validation results, and all trial inference bundles.

## Publication decisions and external setup

The remote is configured as `git@github.com:yhilpisch/synthts.git`, with
Source/Issues package metadata. Licence selection remains pending: add the
chosen `LICENSE` and SPDX licence metadata before publication. Confirm
PyPI/TestPyPI name ownership, configure Trusted Publishers/environments,
and rehearse the release from the same commit/tag intended for production.
No tag, push, or upload has been performed. The publication gate still
fails while licence metadata is absent. Public raw URLs for package metadata
and the first notebook returned HTTP 404 on 2026-10-28. Make the reviewed
notebooks/package available on public `main` before sharing the Colab links.
See `RELEASE.md` for the release procedure.

## Canonical data and empirical evidence

Select the canonical real-data snapshot/run and establish source institution,
price definitions/adjustments, and redistribution terms. The snapshot pipeline
records provenance but cannot infer those rights or certify historical inputs.
Existing selected/reference metrics remain historical in-sample observations.

Before comparative financial claims, run predeclared fixed-budget/multi-seed
experiments with training-only scaling, held-out validation/test periods,
resampling baselines, and dependence-aware uncertainty. The code now supports
these splits and retains portable inference contracts. Default notebook
training budgets have passed functional execution checks; historical-result
reproduction and controlled benchmarks remain outstanding. Notebook outputs
remain cleared. Use an explicitly identified run for published empirical
results; the package documentation makes no benchmark superiority claim.

## Validation still to run externally

Hosted Google Colab Run all, GitHub Actions, the Python 3.10/3.11 jobs,
declared runtime-minimum combinations, TestPyPI installation, and production
publication have not run in this session. The workflows are configured; YAML parsing is not evidence
that hosted jobs have passed. GPU behaviour has not been validated.

## Local validation baseline

- Workspace interpreter: `/Users/yves/Temp/pyai/bin/python`, Python 3.12.12.
  **190 tests passed**; Ruff, formal notebook checks, and source/diff checks
  passed. This includes all four notebooks and every README Python
  example on deterministic local inputs with reduced training budgets.
- All four unchanged notebooks passed full execution in fresh CPU kernels
  with live public data, a temporary local installation, and the workspace
  interpreter. No training budgets were reduced: vanilla 27.4 s, WGAN 28.0 s,
  sequential 278.6 s, TimeGAN 231.2 s. These timings describe this local run,
  not Colab performance. Notebook source hashes were unchanged. The local
  report is `reports/colab_full_2026-10-28.json` (ignored), identifying commit
  `1678925` and dependency versions. No outputs were saved into notebooks.
- Bootstrap tests cover public GitHub installation, local checkout detection,
  missing helpers, existing installations, and propagated install failures.
  The public GitHub installation branch is tested with a simulated installer;
  live installation requires the public files noted above.
- Serial TimeGAN and two-worker sequential sweep CLI checks pass. Exact input
  snapshot checksums, chronological partitions, selected-setting exports,
  and inference save/load equivalence are checked.
- Wheel/source distribution build, `twine check`, and archive-content
  inspection pass. The wheel contains only `synthts` and its metadata;
  source archives exclude local runs, datasets, and notebook checkpoints.
- The preceding packaging sweep installed the candidate wheel and then-current
  resolved dependencies in a fresh temporary environment. The offline README example and **180 installed
  package/notebook tests passed outside the checkout**; repository-only
  sweep tests run separately in the checkout. `pip check` found no conflicts.
  This environment resolved NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1,
  scikit-learn 1.9.1, Matplotlib 3.11.2, and Torch 2.14.1. The shared `pyai`
  environment was not changed.

These checks establish API/integration behaviour, not convergence, market
realism, or validated financial performance.
