"""Validate the Colab notebooks or run them unchanged in fresh CPU kernels.

Run: python scripts/validate_colab.py [--execute] [--report PATH]
Execution installs this checkout into a temporary directory, leaving the
current environment and source notebooks unchanged. Install .[dev] first.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = (
    '01_synthetic_gan.ipynb', '02_wgan_gp.ipynb',
    '03_sequential.ipynb', '04_timegan.ipynb',
)
SOURCE = 'synthts @ git+https://github.com/yhilpisch/synthts.git@main'
COLAB = 'https://colab.research.google.com/github/yhilpisch/synthts/'


def validate(path):
    """Validate schema, bootstrap, badge, syntax, and clean cell sources."""
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager

    raw = json.loads(path.read_text())
    nbformat.validate(raw)
    identifiers = [c['id'] for c in raw['cells'] if 'id' in c]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('duplicate cell IDs')
    notebook = nbformat.reads(json.dumps(raw), as_version=4)
    code = [c for c in notebook.cells if c.cell_type == 'code']
    if not code:
        raise ValueError('no code cells')
    setup = ast.parse(code[0].source)
    constants = {n.value for n in ast.walk(setup)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    if SOURCE not in constants or 'pyyaml>=6.0' not in constants:
        raise ValueError('first code cell must install synthts/dependencies')
    badge = f'{COLAB}blob/main/notebooks/{path.name}'
    if not any(badge in c.source for c in notebook.cells
               if c.cell_type == 'markdown'):
        raise ValueError('missing or incorrect Colab link')
    transformer = TransformerManager()
    for number, cell in enumerate(notebook.cells, 1):
        if cell.source != cell.source.strip('\n'):
            raise ValueError(f'cell {number}: leading/trailing empty lines')
        if cell.cell_type != 'code':
            continue
        if cell.outputs or cell.execution_count is not None:
            raise ValueError(f'cell {number}: stored execution/output')
        if set(cell.metadata.get('tags', [])) & {
                'skip-execution', 'raises-exception'}:
            raise ValueError(f'cell {number}: execution bypass tag')
        for line in cell.source.splitlines():
            if len(line) > 80 or '\t' in line or line != line.rstrip():
                raise ValueError(f'cell {number}: width/whitespace')
        compile(transformer.transform_cell(cell.source), str(path), 'exec')
    return notebook


def execute(notebook, runtime, timeout):
    """Run every cell with plotting using this interpreter and package."""
    from nbclient import NotebookClient

    with tempfile.TemporaryDirectory(prefix='synthts-kernel-') as directory:
        env = os.environ | {
            'MPLBACKEND': 'Agg', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'MPLCONFIGDIR': str(Path(directory) / 'matplotlib'),
            'IPYTHONDIR': str(Path(directory) / 'ipython'),
            'PYTHONPATH': os.pathsep.join(filter(None, (
                str(runtime), os.environ.get('PYTHONPATH')))),
        }
        client = NotebookClient(
            notebook, timeout=timeout, kernel_name='python3',
            allow_errors=False, force_raise_errors=True,
            record_timing=False, resources={'metadata': {'path': directory}})
        client.km = client.create_kernel_manager()
        spec = client.km.kernel_spec
        if spec is None:
            raise RuntimeError('Python kernel specification unavailable')
        spec.argv = [sys.executable, '-m', 'ipykernel_launcher',
                     '-f', '{connection_file}']
        client.execute(env=env)
        for cell in notebook.cells:
            if cell.cell_type == 'code' and cell.source.strip():
                if cell.execution_count is None:
                    raise RuntimeError('a code cell did not execute')
                if any(o.output_type == 'error' for o in cell.outputs):
                    raise RuntimeError('a code cell returned an error')


def main():
    """Check all four notebooks and optionally record full-run evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--timeout', type=int, default=1800,
                        help='Seconds per cell (default 1800)')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    paths = sorted((ROOT / 'notebooks').glob('*.ipynb'))
    if {p.name for p in paths} != set(EXPECTED):
        raise ValueError('unexpected notebook inventory')
    results, notebooks, setups = [], {}, set()
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for path in paths:
        start = time.perf_counter()
        result = {'notebook': path.name, 'phase': 'formal', 'error': None}
        try:
            notebooks[path.name] = validate(path)
            setups.add(next(c.source for c in notebooks[path.name].cells
                            if c.cell_type == 'code'))
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        result['seconds'] = time.perf_counter() - start
        results.append(result)
        print(f"{'FAIL' if result['error'] else 'PASS'} formal {path.name}",
              flush=True)
    if len(setups) > 1:
        results.append({'phase': 'formal', 'notebook': 'setup cells',
                        'error': 'setup cells differ across notebooks'})
    if args.execute and not any(r['error'] for r in results):
        with tempfile.TemporaryDirectory(prefix='synthts-colab-') as directory:
            runtime = Path(directory) / 'site'
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps',
                '--no-build-isolation', '--target', str(runtime), str(ROOT),
            ], check=True)
            for name, notebook in notebooks.items():
                result = {'notebook': name, 'phase': 'execute', 'error': None}
                start = time.perf_counter()
                print(f'RUN  {name}', flush=True)
                try:
                    execute(notebook, runtime, args.timeout)
                except Exception as exc:
                    result['error'] = f'{type(exc).__name__}: {exc}'
                result['seconds'] = time.perf_counter() - start
                results.append(result)
                status = 'FAIL' if result['error'] else 'PASS'
                print(f"{status} execute {name} ({result['seconds']:.1f}s)",
                      flush=True)
    success = not any(r['error'] for r in results)
    if args.report:
        versions = {name: importlib.metadata.version(name) for name in (
            'numpy', 'pandas', 'torch', 'matplotlib', 'nbformat', 'nbclient')}
        report = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'success': success, 'execute': args.execute,
            'python': sys.version, 'executable': sys.executable,
            'versions': versions, 'results': results,
            'git_commit': subprocess.check_output(
                ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                text=True).strip(),
            'notebook_sha256': hashes,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    for result in results:
        if result['error']:
            print(result['error'], file=sys.stderr)
    return int(not success)


if __name__ == '__main__':
    raise SystemExit(main())
