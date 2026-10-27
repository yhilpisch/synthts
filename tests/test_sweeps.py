"""Offline sweep configuration and process-level checks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from sweeps.common import prepare_data
from sweeps import sweep_sequential, sweep_timegan
from synthts.artifacts import load_bundle


@pytest.fixture()
def data_path(tmp_path):
    rng = np.random.default_rng(8)
    prices = 100 * np.exp(rng.normal(0, .01, 200).cumsum())
    path = tmp_path / 'illustrative.csv'
    pd.DataFrame({'GLD': prices},
                 index=pd.date_range('2020-01-01', periods=200)).to_csv(path)
    return path


def test_chronological_split_scaler_and_checksum(data_path):
    settings = sweep_sequential.DEFAULT_CONFIG | {'data_path': str(data_path),
                                                   'seq_len': 12}
    train, val, test, scaled, scaler, meta = prepare_data(settings)
    assert len(train) + len(val) + len(test) == 199
    assert meta['return_split_offsets'] == [0, 121, 160, 199]
    assert len(meta['sha256']) == 64
    assert scaler.scaler.mean_[0] == pytest.approx(train.mean())
    assert abs(scaled.mean()) < 1e-6
    assert not np.isclose(train.mean(), val.mean())


@pytest.mark.parametrize('module', [sweep_sequential, sweep_timegan])
def test_invalid_configuration_paths_and_grids_fail(module):
    with pytest.raises(FileNotFoundError):
        module.load_settings(argparse.Namespace(config='missing-audit.yml'))
    key = 'hu_gen_values' if module is sweep_sequential else 'hidden_units_values'
    with pytest.raises(ValueError, match='empty'):
        module.build_configs(module.DEFAULT_CONFIG | {key: []})


@pytest.mark.parametrize('family, jobs', [('sequential', 2), ('timegan', 1)])
def test_offline_sweep_cli(tmp_path, data_path, family, jobs):
    module = sweep_sequential if family == 'sequential' else sweep_timegan
    settings = dict(module.DEFAULT_CONFIG)
    settings.update(data_path=str(data_path), seq_len=12, n_jobs=jobs,
                    torch_threads=1, n_eval=4, max_trials=None,
                    out_dir=str(tmp_path / 'runs'), export_on_complete=True,
                    export_best=str(tmp_path / 'selected.yml'))
    for key, value in list(settings.items()):
        if key.endswith('_values'):
            settings[key] = value[:1]
    if family == 'sequential':
        settings.update(hu_gen_values=[4, 6], hu_disc_values=[4],
                        epochs_values=[2], stat_max_lag=3)
    else:
        settings.update(hidden_units_values=[4], latent_dim_values=[3],
                        noise_dim_values=[2], embedding_epochs=1,
                        supervisor_epochs=1, joint_epochs_values=[2],
                        stat_max_lag_values=[3], checkpoint_interval=1,
                        obs_disc_channels=4, obs_disc_layers=1)
    settings['batch_size_values'] = [4]
    config = tmp_path / 'config.yml'
    config.write_text(yaml.safe_dump(settings))
    env = os.environ | {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                        'MPLBACKEND': 'Agg'}
    result = subprocess.run(
        [sys.executable, '-m', f'sweeps.sweep_{family}',
         '--config', str(config)], cwd=Path(__file__).resolve().parents[1],
        env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    run, = (tmp_path / 'runs').iterdir()
    rows = pd.read_csv(run / 'results.csv')
    assert len(rows) == (2 if family == 'sequential' else 1)
    assert np.isfinite(rows['score']).all()
    assert rows['seed'].nunique() == 1
    meta = json.loads((run / 'data.meta.json').read_text())
    assert meta['protocol'].endswith('test unused in sweep')
    assert hashlib.sha256((run / 'data.csv').read_bytes()).hexdigest() == (
        meta['sha256'])
    for trial in rows['trial']:
        bundle = load_bundle(run / f'trial_{trial:04d}.pt')
        assert bundle['metadata']['sha256'] == meta['sha256']
    selected = yaml.safe_load((tmp_path / 'selected.yml').read_text())
    if family == 'timegan':
        assert selected['model']['obs_disc_channels'] == 4
        assert selected['model']['obs_disc_layers'] == 1
        assert selected['model']['latent_activation'] == settings[
            'latent_activation']
