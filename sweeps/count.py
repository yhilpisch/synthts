"""Count sweep combinations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from sweeps.common import load_config as read_config
from synthts._validation import integer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = {
    'max_trials': None,
    'grid_keys': None,
}


def parse_values(value) -> list:
    """Parse list-like config values."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [v.strip() for v in str(value).split(',') if v.strip()]


def load_config(path: str) -> dict:
    """Load config file with defaults."""
    config = DEFAULT_CONFIG.copy()
    config_path = PROJECT_ROOT / path
    config.update(read_config(config_path))
    return config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='sweeps/config.yml')
    return parser.parse_args()


def main() -> None:
    """Print sweep combination counts."""
    args = parse_args()
    config = load_config(args.config)
    grid_keys = config.get('grid_keys')
    if not grid_keys:
        grid_keys = sorted(k for k in config if k.endswith('_values'))
    counts = {key: len(parse_values(config[key])) for key in grid_keys}
    if not counts or any(value == 0 for value in counts.values()):
        raise ValueError('sweep grid must not be empty')
    total = math.prod(counts.values())
    max_trials = config['max_trials']
    if max_trials is not None:
        integer('max_trials', max_trials)
    effective = min(total, max_trials) if max_trials is not None else total

    print(f'config: {PROJECT_ROOT / args.config}')
    for key in grid_keys:
        print(f'{key}: {counts[key]}')
    print(f'total combinations: {total}')
    print(f'effective trials: {effective}')


if __name__ == '__main__':
    main()
