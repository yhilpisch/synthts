"""Count sequential GAN sweep combinations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = {
    'max_trials': None,
    'hu_gen_values': [64, 96],
    'hu_disc_values': [64],
    'batch_size_values': [128, 256],
    'lr_g_values': [0.0002, 0.0003],
    'stat_weight_values': [3.0, 5.0, 7.5],
    'epochs_values': [750, 1500],
}

GRID_KEYS = [
    'hu_gen_values',
    'hu_disc_values',
    'batch_size_values',
    'lr_g_values',
    'stat_weight_values',
    'epochs_values',
]


def parse_values(value) -> list:
    """Parse list-like config values."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [v.strip() for v in str(value).split(',') if v.strip()]


def load_config(path: str) -> dict:
    """Load config file with defaults."""
    config = DEFAULT_CONFIG.copy()
    config_path = PROJECT_ROOT / path
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text()) or {}
        config.update(loaded)
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
    counts = {key: len(parse_values(config[key])) for key in GRID_KEYS}
    total = math.prod(counts.values())
    effective = min(total, config['max_trials']) if config['max_trials'] else total

    print(f'config: {PROJECT_ROOT / args.config}')
    for key in GRID_KEYS:
        print(f'{key}: {counts[key]}')
    print(f'total combinations: {total}')
    print(f'effective trials: {effective}')


if __name__ == '__main__':
    main()
