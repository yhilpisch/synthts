"""Data provenance and validation shared by research sweeps."""

import hashlib
import importlib.metadata
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from synthts._validation import integer
from synthts.data import (
    EOD_URL, ReturnScaler, compute_log_returns, load_eod, split_returns,
)


def load_config(path):
    """Require a YAML mapping rather than silently ignoring path errors."""
    import yaml

    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError('configuration must be a YAML mapping')
    return config


def validate_settings(settings):
    """Check runtime settings before creating files or launching workers."""
    for key in ('n_jobs', 'torch_threads', 'n_eval', 'seq_len'):
        integer(key, settings[key])
    if settings['seq_len'] <= 10:
        raise ValueError('sweep evaluation requires seq_len > 10')
    if settings.get('max_trials') is not None:
        integer('max_trials', settings['max_trials'])
    for key in ('validation_fraction', 'test_fraction'):
        value = settings[key]
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError(f'{key} must lie between zero and one')
    if settings['validation_fraction'] + settings['test_fraction'] >= 1:
        raise ValueError('validation/test fractions must sum to less than 1')


def prepare_data(settings, snapshot_dir=None):
    """Hash exact input bytes and split returns before scaling/windowing."""
    source = settings.get('data_path') or EOD_URL
    if str(source).startswith(('https://', 'http://')):
        with urlopen(source, timeout=60) as response:
            content = response.read()
    else:
        content = Path(source).read_bytes()
    prices = load_eod(url=io.BytesIO(content), symbol=settings['symbol'])
    returns = compute_log_returns(prices)
    train, validation, test = split_returns(
        returns, settings['validation_fraction'], settings['test_fraction'],
        settings['seq_len'])
    val_start, test_start = len(train), len(train) + len(validation)
    scaler = ReturnScaler()
    scaled = scaler.fit_transform(train)
    metadata = {
        'source': str(source), 'symbol': settings['symbol'],
        'source_institution': settings.get('source_institution'),
        'schema': {'index': 'date',
                   'selected_price_column': settings['symbol']},
        'retrieved_at': datetime.now(timezone.utc).isoformat(),
        'sha256': hashlib.sha256(content).hexdigest(),
        'price_start': str(prices.index[0]),
        'price_end': str(prices.index[-1]),
        'transformation': 'adjacent log price differences; training-only scale',
        'return_split_offsets': [0, val_start, test_start, len(returns)],
        'protocol': 'chronological train/validation/test; test unused in sweep',
        'versions': {name: importlib.metadata.version(name) for name in (
            'numpy', 'pandas', 'scipy', 'scikit-learn', 'torch')},
        'redistribution_note': 'input not redistributed; confirm source rights',
    }
    if snapshot_dir is not None:
        snapshot = Path(snapshot_dir)
        (snapshot / 'data.csv').write_bytes(content)
        write_metadata(snapshot / 'data.meta.json', metadata)
    return train, validation, test, scaled, scaler, metadata


def finite_row(row):
    """Reject non-finite results before selecting or exporting a trial."""
    for key, value in row.items():
        if (isinstance(value, (int, float, np.number))
                and not np.isfinite(value)):
            raise FloatingPointError(f'non-finite trial metric: {key}')
    return row


def write_metadata(path, metadata):
    """Write strict JSON provenance."""
    Path(path).write_text(json.dumps(metadata, indent=2, allow_nan=False)
                          + '\n', encoding='utf-8')
