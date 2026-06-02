"""CLI parameter sweep for the sequential GAN."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from synthts.data import compute_log_returns, load_eod, ReturnScaler
from synthts.evaluate import (
    compare_acf_windows,
    drawdown_distribution_windows,
    ks_test,
    pass_rate,
    sample_windows,
    sequence_ks_pvalues,
)
from synthts.sequential import (
    SeqDiscriminator,
    SeqGenerator,
    generate_sequences,
    train_seq_gan,
)


DEFAULT_CONFIG = {
    'symbol': 'GLD',
    'n_jobs': 3,
    'torch_threads': 1,
    'n_eval': 100,
    'max_trials': None,
    'seed': 100,
    'seq_len': 63,
    'latent_dim': 8,
    'noise_dim': 4,
    'lr_d': 0.0001,
    'n_gen_steps': 2,
    'stat_max_lag': 5,
    'hu_gen_values': [64, 96],
    'hu_disc_values': [64],
    'batch_size_values': [128, 256],
    'lr_g_values': [0.0002, 0.0003],
    'stat_weight_values': [3.0, 5.0, 7.5],
    'epochs_values': [750, 1500],
    'out_dir': 'sweeps/results',
    'export_best': 'sweeps/sequential_best.yml',
    'export_on_complete': True,
    'task_order': 'random',
    'task_order_seed': None,
}


@dataclass(frozen=True)
class SweepConfig:
    trial: int
    seed: int
    seq_len: int
    latent_dim: int
    noise_dim: int
    hu_gen: int
    hu_disc: int
    epochs: int
    batch_size: int
    lr_d: float
    lr_g: float
    n_gen_steps: int
    stat_weight: float
    stat_max_lag: int


def parse_values(value, kind=float) -> list:
    """Parse list-like config values."""
    if isinstance(value, (list, tuple)):
        return [kind(v) for v in value]
    return [kind(v.strip()) for v in str(value).split(',') if v.strip()]


def build_configs(settings: dict) -> list[SweepConfig]:
    """Create the sweep grid."""
    task_order = settings.get('task_order', 'random')
    if task_order not in {'grid', 'random'}:
        raise ValueError('task_order must be grid or random')
    grid = itertools.product(
        parse_values(settings['hu_gen_values'], int),
        parse_values(settings['hu_disc_values'], int),
        parse_values(settings['batch_size_values'], int),
        parse_values(settings['lr_g_values'], float),
        parse_values(settings['stat_weight_values'], float),
        parse_values(settings['epochs_values'], int),
    )
    configs = []
    for trial, values in enumerate(grid, start=1):
        hu_gen, hu_disc, batch_size, lr_g, stat_weight, epochs = values
        trial_config = SweepConfig(
            trial=trial,
            seed=settings['seed'] + trial - 1,
            seq_len=settings['seq_len'],
            latent_dim=settings['latent_dim'],
            noise_dim=settings['noise_dim'],
            hu_gen=hu_gen,
            hu_disc=hu_disc,
            epochs=epochs,
            batch_size=batch_size,
            lr_d=settings['lr_d'],
            lr_g=lr_g,
            n_gen_steps=settings['n_gen_steps'],
            stat_weight=stat_weight,
            stat_max_lag=settings['stat_max_lag'],
        )
        configs.append(trial_config)
    if task_order == 'random':
        order_seed = settings.get('task_order_seed')
        if order_seed is None:
            order_seed = settings['seed']
        rng = np.random.default_rng(int(order_seed))
        rng.shuffle(configs)
    if settings['max_trials']:
        configs = configs[:settings['max_trials']]
    return configs


def score_metrics(row: dict) -> float:
    """Composite score; lower is better."""
    return (
        row['mean_err_scaled']
        + row['std_err_scaled']
        + row['tail_err_scaled']
        + row['acf_diff']
        + row['acf2_diff']
        + 10.0 * row['drawdown_mean_err']
        - row['ks_pass_rate']
    )


def json_ready(row: dict) -> dict:
    """Convert NumPy scalar values to plain Python values."""
    clean = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean


def run_trial(payload: tuple) -> dict:
    """Train one configuration and return metrics."""
    config, rets, rets_scaled, mean, scale, n_eval, threads = payload
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)

    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    generator = SeqGenerator(
        latent_dim=config.latent_dim,
        noise_dim=config.noise_dim,
        hidden_units=config.hu_gen,
        output_dim=1,
        seq_len=config.seq_len,
        num_layers=1,
    )
    discriminator = SeqDiscriminator(input_dim=1,
                                     hidden_units=config.hu_disc)

    start = time.time()
    with redirect_stdout(StringIO()):
        history = train_seq_gan(
            generator,
            discriminator,
            rets_scaled,
            seq_len=config.seq_len,
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr_d=config.lr_d,
            lr_g=config.lr_g,
            latent_dim=config.latent_dim,
            label_smooth=0.1,
            n_gen_steps=config.n_gen_steps,
            stat_weight=config.stat_weight,
            stat_max_lag=config.stat_max_lag,
            seed=config.seed,
            log_interval=config.epochs + 1,
        )
    seconds = time.time() - start

    seqs = generate_sequences(
        generator,
        n_eval,
        latent_dim=config.latent_dim,
        seed=config.seed + 10_000,
    )
    synth_windows = seqs[:, :, 0] * scale + mean
    real_windows = sample_windows(
        rets,
        config.seq_len,
        n_eval,
        rng_seed=config.seed + 20_000,
    )
    synth_flat = synth_windows.reshape(-1)
    real_flat = real_windows.reshape(-1)
    real_std = real_flat.std()

    ks_flat = ks_test(real_flat, synth_flat)
    pvals = sequence_ks_pvalues(real_windows, synth_windows)
    comp = compare_acf_windows(real_windows, synth_windows,
                               max_lag=10)
    dd_real = drawdown_distribution_windows(real_windows)
    dd_synth = drawdown_distribution_windows(synth_windows)
    q = np.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    q_real = np.quantile(real_flat, q)
    q_synth = np.quantile(synth_flat, q)

    row = asdict(config)
    row.update({
        'seconds': seconds,
        'final_g_loss': history['g_loss'][-1],
        'final_d_loss_real': history['d_loss_real'][-1],
        'final_d_loss_fake': history['d_loss_fake'][-1],
        'real_mean': real_flat.mean(),
        'synth_mean': synth_flat.mean(),
        'real_std': real_std,
        'synth_std': synth_flat.std(),
        'mean_err_scaled': abs(real_flat.mean()
                               - synth_flat.mean()) / real_std,
        'std_err_scaled': abs(real_std - synth_flat.std()) / real_std,
        'tail_err_scaled': np.abs(q_real - q_synth).mean() / real_std,
        'ks_stat': ks_flat['statistic'],
        'ks_pvalue': ks_flat['pvalue'],
        'ks_pass_rate': pass_rate(pvals),
        'ks_median_pvalue': np.median(pvals),
        'acf_diff': comp['acf_diff'],
        'acf2_diff': comp['acf2_diff'],
        'real_drawdown_mean': dd_real.mean(),
        'synth_drawdown_mean': dd_synth.mean(),
        'drawdown_mean_err': abs(dd_real.mean() - dd_synth.mean()),
    })
    row['score'] = score_metrics(row)
    return json_ready(row)


def write_row(path: Path, row: dict) -> None:
    """Append a result row to CSV."""
    write_header = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_notebook_config(best: dict, settings: dict,
                          result_dir: Path) -> dict:
    """Build a notebook-ready config from the winning trial."""
    return {
        'source': {
            'result_dir': str(result_dir.relative_to(PROJECT_ROOT)),
            'trial': int(best['trial']),
            'score': float(best['score']),
        },
        'data': {
            'symbol': settings['symbol'],
        },
        'model': {
            'seq_len': int(best['seq_len']),
            'latent_dim': int(best['latent_dim']),
            'noise_dim': int(best['noise_dim']),
            'hu_gen': int(best['hu_gen']),
            'hu_disc': int(best['hu_disc']),
        },
        'training': {
            'epochs': int(best['epochs']),
            'batch_size': int(best['batch_size']),
            'lr_d': float(best['lr_d']),
            'lr_g': float(best['lr_g']),
            'label_smooth': 0.1,
            'n_gen_steps': int(best['n_gen_steps']),
            'stat_weight': float(best['stat_weight']),
            'stat_max_lag': int(best['stat_max_lag']),
            'seed': int(best['seed']),
            'log_interval': 500,
        },
        'generation': {
            'n_sequences': int(settings['n_eval']),
            'seed': int(best['seed']) + 10_000,
            'window_seed': int(best['seed']) + 20_000,
        },
        'metrics': {
            'score': float(best['score']),
            'ks_stat': float(best['ks_stat']),
            'ks_pass_rate': float(best['ks_pass_rate']),
            'acf_diff': float(best['acf_diff']),
            'acf2_diff': float(best['acf2_diff']),
            'real_mean': float(best['real_mean']),
            'synth_mean': float(best['synth_mean']),
            'real_std': float(best['real_std']),
            'synth_std': float(best['synth_std']),
        },
    }


def export_best_config(best: dict, settings: dict,
                       result_dir: Path) -> Path | None:
    """Export the winning trial as notebook-ready YAML."""
    if not settings.get('export_on_complete', True):
        return None
    export_path_raw = settings.get('export_best')
    if not export_path_raw:
        return None
    export_path = PROJECT_ROOT / export_path_raw
    export_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_config = build_notebook_config(best, settings, result_dir)
    export_path.write_text(
        yaml.safe_dump(notebook_config, sort_keys=False),
        encoding='utf-8',
    )
    return export_path


def load_settings(args: argparse.Namespace) -> dict:
    """Merge defaults, YAML config, and CLI overrides."""
    settings = DEFAULT_CONFIG.copy()
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text()) or {}
        settings.update(loaded)
    overrides = vars(args).copy()
    overrides.pop('config')
    settings.update(overrides)
    return settings


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='sweeps/config.yml')
    parser.add_argument('--symbol', default=argparse.SUPPRESS)
    parser.add_argument('--n-jobs', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--torch-threads', type=int,
                        default=argparse.SUPPRESS)
    parser.add_argument('--n-eval', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--max-trials', type=int,
                        default=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--seq-len', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--latent-dim', type=int,
                        default=argparse.SUPPRESS)
    parser.add_argument('--noise-dim', type=int,
                        default=argparse.SUPPRESS)
    parser.add_argument('--lr-d', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--n-gen-steps', type=int,
                        default=argparse.SUPPRESS)
    parser.add_argument('--stat-max-lag', type=int,
                        default=argparse.SUPPRESS)
    parser.add_argument('--hu-gen-values', default=argparse.SUPPRESS)
    parser.add_argument('--hu-disc-values', default=argparse.SUPPRESS)
    parser.add_argument('--batch-size-values', default=argparse.SUPPRESS)
    parser.add_argument('--lr-g-values', default=argparse.SUPPRESS)
    parser.add_argument('--stat-weight-values', default=argparse.SUPPRESS)
    parser.add_argument('--epochs-values', default=argparse.SUPPRESS)
    parser.add_argument('--out-dir', default=argparse.SUPPRESS)
    parser.add_argument('--export-best', default=argparse.SUPPRESS)
    parser.add_argument('--task-order', default=argparse.SUPPRESS)
    parser.add_argument('--task-order-seed', type=int,
                        default=argparse.SUPPRESS)
    parser.add_argument('--export-on-complete', action='store_true',
                        default=argparse.SUPPRESS)
    parser.add_argument('--no-export-on-complete',
                        dest='export_on_complete',
                        action='store_false',
                        default=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    """Run the sweep."""
    args = parse_args()
    settings = load_settings(args)
    run_id = datetime.now().strftime('sequential_%Y%m%d_%H%M%S')
    out_dir = PROJECT_ROOT / settings['out_dir'] / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'results.csv'
    best_path = out_dir / 'best_config.json'
    settings_path = out_dir / 'effective_config.yml'
    settings_path.write_text(yaml.safe_dump(settings, sort_keys=True))

    prices = load_eod(symbol=settings['symbol'])
    rets = compute_log_returns(prices)
    scaler = ReturnScaler()
    rets_scaled = scaler.fit_transform(rets)
    mean = float(scaler.scaler.mean_[0])
    scale = float(scaler.scaler.scale_[0])

    configs = build_configs(settings)
    payloads = [
        (config, rets, rets_scaled, mean, scale,
         settings['n_eval'], settings['torch_threads'])
        for config in configs
    ]
    print(f'Writing results to {out_dir}')
    print(f'Running {len(payloads)} trials with n_jobs={settings["n_jobs"]}')

    rows = []
    best = None
    if settings['n_jobs'] == 1:
        progress = tqdm(payloads, total=len(payloads), desc='trials')
        for payload in progress:
            row = run_trial(payload)
            rows.append(row)
            write_row(csv_path, row)
            if best is None or row['score'] < best['score']:
                best = row
            best_path.write_text(json.dumps(best, indent=2))
            export_best_config(best, settings, out_dir)
            progress.set_postfix(
                trial=f'{row["trial"]:03d}',
                score=f'{row["score"]:.4f}',
                best=f'{best["score"]:.4f}',
            )
    else:
        with ProcessPoolExecutor(max_workers=settings['n_jobs']) as pool:
            futures = [pool.submit(run_trial, p) for p in payloads]
            progress = tqdm(
                as_completed(futures),
                total=len(futures),
                desc='trials',
            )
            for future in progress:
                row = future.result()
                rows.append(row)
                write_row(csv_path, row)
                if best is None or row['score'] < best['score']:
                    best = row
                best_path.write_text(json.dumps(best, indent=2))
                export_best_config(best, settings, out_dir)
                progress.set_postfix(
                    trial=f'{row["trial"]:03d}',
                    score=f'{row["score"]:.4f}',
                    best=f'{best["score"]:.4f}',
                )

    export_path = export_best_config(best, settings, out_dir)
    print(f'Best trial: {best["trial"]:03d}')
    print(f'Best score: {best["score"]:.4f}')
    print(f'Best config: {best_path}')
    if export_path is not None:
        print(f'Notebook config: {export_path}')


if __name__ == '__main__':
    main()
