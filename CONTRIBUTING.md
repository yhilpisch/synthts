# Contributing

Install the development dependencies in a virtual environment, then run the
checks from the checkout root:

```bash
python -m pip install '.[dev]'
python -m ruff check synthts sweeps tests scripts
python -m pytest
python -m build
python -m twine check dist/*
python scripts/check_distribution.py dist
```

The test suite uses small deterministic synthetic inputs. Notebook tests
execute all code with reduced training budgets and local prices. They do
not establish long-run convergence or financial realism.

Keep changes focused and include regressions for changed numerical behaviour.
Use at most 80 characters per Python line and two spaces before inline
comments. Annotated defaults follow the intentional `value: int=5` style.
Use concise comments and preserve the existing model families and public API
unless a change is explicitly agreed.

Record source, retrieval date, checksum, preprocessing, chronological split,
model settings, seeds, versions, and evaluation protocol for real-data
experiments. Preserve old runs; corrected metrics belong in a new run.
Do not commit datasets without redistribution rights, secrets, model binaries,
or generated figures. Use ignored run directories or release assets instead.

Report reproducible problems with the package version, environment, minimal
example, and expected/observed behaviour. Include synthetic or redistributable
inputs; never include account credentials or proprietary data. Use the
[issue tracker](https://github.com/yhilpisch/synthts/issues), or contact
team@tpq.io for support.
