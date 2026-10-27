# Release procedure

Version `0.1.0` is prepared locally and has not been published. The package
and distribution are both `synthts`; the version is defined once in
`synthts/_version.py`.

## Publication prerequisites

Choose and add the licence before distribution. Configure the intended
GitHub repository and add its Source/Issues URLs to `pyproject.toml`.
Confirm ownership/availability of `synthts` on PyPI and TestPyPI. Configure
Trusted Publishers for `.github/workflows/release.yml` with the corresponding
`pypi`/`testpypi` GitHub environments. These account settings are external to
the source tree. Licence selection and remote creation are not inferred.

The official references are the
[PyPA packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
and [PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Validate a candidate

Use a fresh checkout and virtual environment. Install CPU Torch where
appropriate for the platform; no GPU is needed for validation.

```bash
python -m pip install '.[dev]'
python -m ruff check synthts sweeps tests scripts
python -m pytest
python -m build
python -m twine check dist/*
python scripts/check_distribution.py dist
python scripts/check_distribution.py dist --publication
```

The publication check additionally requires licence and repository metadata.
Inspect the source archive and wheel, rebuild a wheel from the source archive,
and install the candidate wheel in a separate environment. Run the offline
README example and core/notebook tests outside the checkout and verify that `synthts.__file__` points
to that environment's installed package. CI automates these checks; repository-only sweep CLI tests run in the
checkout jobs.

Update the changelog and version before creating the corresponding `vX.Y.Z`
tag. Local commits follow the house policy; tagging/pushing/uploading is a
separate release action. Do not republish the same version with new contents.

## Rehearse and publish

Dispatch `release.yml` for `testpypi` first. It validates the candidate and
uploads the same tested distributions using Trusted Publishing. Install the
result from TestPyPI in a clean environment, with runtime dependencies
installed from the normal index, and run the offline example. Check the
rendered README and distribution metadata on TestPyPI.

For production, use the reviewed version tag and dispatch the workflow for
`pypi` on that tag; tag pushes also run the production workflow. Production
publication requires the tag to match the packaged version. A TestPyPI
rehearsal of that same commit must be completed before the production action.

## Research evidence

Package publication does not imply validated comparative performance.
Identify a frozen dataset and its redistribution terms before publishing
financial benchmark claims. Retain fixed budgets, predeclared seeds,
training-only preprocessing, held-out selection/evaluation, baseline models,
and complete inference contracts. Keep historical metrics labelled with
their original protocol; do not overwrite them with corrected results.
