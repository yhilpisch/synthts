"""Execute public examples offline with short training budgets."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from synthts import vanilla, wgangp, sequential, timegan


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def offline_examples(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    path = tmp_path / 'illustrative.csv'
    prices = 100 * np.exp(rng.normal(0, .01, 800).cumsum())
    pd.DataFrame({'GLD': prices, 'SPY': prices},
                 index=pd.date_range('2020-01-01', periods=800)).to_csv(path)
    monkeypatch.setenv('SYNTHTS_DATA', str(path))
    config = {
        'model': {'hidden_units': 8, 'latent_dim': 4, 'noise_dim': 3,
                  'latent_activation': 'layernorm', 'obs_disc_channels': 4,
                  'obs_disc_layers': 1, 'hu_gen': 8, 'hu_disc': 8},
        'training': {'mean_weight': 2., 'std_weight': 3.,
                     'lr_schedule': 'cosine', 'checkpoint_interval': 1},
    }
    config_path = tmp_path / 'selected.yml'
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv('SYNTHTS_CONFIG', str(config_path))
    torch.set_num_threads(1)

    def shortened(original):
        def run(*args, **kwargs):
            if original is timegan_train:
                kwargs.update(embedding_epochs=1, supervisor_epochs=1,
                              joint_epochs=2, batch_size=4)
            else:
                kwargs.update(epochs=2, batch_size=4)
            result = original(*args, **kwargs)
            assert all(np.isfinite(v).all() for v in result.values())
            return result
        return run

    timegan_train = timegan.train_timegan
    for module, name in [(vanilla, 'train_gan'), (wgangp, 'train_wgangp'),
                         (sequential, 'train_seq_gan'),
                         (timegan, 'train_timegan')]:
        monkeypatch.setattr(module, name, shortened(getattr(module, name)))
    yield path
    plt.close('all')


@pytest.mark.parametrize('path', sorted((ROOT / 'notebooks').glob('*.ipynb')))
def test_notebook_executes_offline(path, offline_examples):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {'__name__': '__main__'}
    for cell in notebook.cells:
        assert not cell.source.startswith('\n')
        assert not cell.source.endswith('\n')
        if cell.cell_type != 'code':
            continue
        assert all(len(line) <= 80 for line in cell.source.splitlines())
        source = cell.source.removeprefix('%%time\n')
        exec(compile(source, str(path), 'exec'), namespace)
    if path.name.startswith('04'):
        assert namespace['embedder'].latent_activation == 'layernorm'
        assert namespace['obs_discriminator'].config['hidden_channels'] == 4
        assert namespace['history']['best_epoch'] > 0
        assert namespace['LOSS_WEIGHTS']['mean_weight'] == 2


def test_readme_python_examples_execute(offline_examples):
    namespace = {'__name__': '__main__'}
    text = (ROOT / 'README.md').read_text()
    for source in re.findall(r'```python\n(.*?)```', text, flags=re.S):
        assert all(len(line) <= 80 for line in source.splitlines())
        exec(compile(source, 'README.md', 'exec'), namespace)
