# Remaining Publication Work

Updated 2026-10-27. The distribution/import name is **synthts**; candidate
version **0.1.0** is prepared locally. Only unresolved work is listed below.

The maintained inventory comprises ten package modules, four notebooks,
research sweep/configuration utilities, regression/example/CLI tests,
standard packaging metadata, archive inspection scripts, contribution/release
instructions, and CI/Trusted Publishing workflows. Historical selected YAML
and ignored research runs are preserved. New sweeps retain exact input CSVs,
metadata/checksums, validation results, and all trial inference bundles.

## Publication decisions and external setup

The licence and GitHub owner/repository are not yet specified. Add the
chosen `LICENSE`, SPDX licence metadata, and Source/Issues URLs before
publication. Confirm PyPI/TestPyPI name ownership, configure the remote and
Trusted Publishers/environments, and rehearse the release from the same
commit/tag intended for production. No remote, tag, push, or upload has been
performed. The archive publication gate deliberately fails while licence or
repository metadata is absent. See `RELEASE.md` for the concrete procedure.

## Canonical data and empirical evidence

Select the canonical real-data snapshot/run and establish source institution,
price definitions/adjustments, and redistribution terms. The snapshot pipeline
records provenance but cannot infer those rights or certify historical inputs.
Existing selected/reference metrics remain historical in-sample observations.

Before comparative financial claims, run predeclared fixed-budget/multi-seed
experiments with training-only scaling, held-out validation/test periods,
resampling baselines, and dependence-aware uncertainty. The code now supports
these splits and retains portable inference contracts. Full training and
historical-result reproduction have not been performed. Cleared notebook
outputs must be refreshed from an explicitly identified run if published
empirical results are desired. The package documentation makes no benchmark
superiority claim.

## Validation still to run externally

Hosted GitHub Actions, the Python 3.10/3.11 jobs, declared runtime-minimum
combinations, TestPyPI installation, and production publication have not run
in this session. The workflows are configured; YAML parsing is not evidence
that hosted jobs have passed. GPU behaviour has not been validated.

## Local validation baseline

- Workspace interpreter: `/Users/yves/Temp/pyai/bin/python`, Python 3.12.12.
  **185 tests passed**, with **98% statement coverage**; Ruff and source/diff
  checks passed. This includes all four notebooks and every README Python
  example on deterministic local inputs with reduced training budgets.
- Serial TimeGAN and two-worker sequential sweep CLI checks pass. Exact input
  snapshot checksums, chronological partitions, selected-setting exports,
  and inference save/load equivalence are checked.
- Wheel/source distribution build, `twine check`, and archive-content
  inspection pass. The wheel contains only `synthts` and its metadata;
  source archives exclude local runs, datasets, and notebook checkpoints.
- A fresh temporary environment installed the candidate wheel and current
  resolved dependencies. The offline README example and **180 installed
  package/notebook tests passed outside the checkout**; repository-only
  sweep tests run separately in the checkout. `pip check` found no conflicts.
  This environment resolved NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1,
  scikit-learn 1.9.1, Matplotlib 3.11.2, and Torch 2.14.1. The shared `pyai`
  environment was not changed.

These checks establish API/integration behaviour, not convergence, market
realism, or validated financial performance.
