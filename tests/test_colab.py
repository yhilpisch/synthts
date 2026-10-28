"""Exercise the missing-package and existing-install Colab setup paths."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'synthts @ git+https://github.com/yhilpisch/synthts.git@main'


def bootstrap():
    path = ROOT / 'notebooks/01_synthetic_gan.ipynb'
    notebook = json.loads(path.read_text())
    return next(''.join(c['source']) for c in notebook['cells']
                if c['cell_type'] == 'code')


def test_missing_package_uses_public_https_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(importlib.util, 'find_spec',
                        lambda name: None if name in {'synthts', 'yaml'} else 1)
    calls = []
    monkeypatch.setattr(subprocess, 'check_call', calls.append)
    exec(bootstrap(), {})
    assert calls == [[sys.executable, '-m', 'pip', 'install', '--quiet',
                      SOURCE, 'pyyaml>=6.0']]


def test_local_checkout_is_preferred(tmp_path, monkeypatch):
    (tmp_path / 'pyproject.toml').write_text('[project]\nname="synthts"\n')
    (tmp_path / 'synthts').mkdir()
    (tmp_path / 'synthts/__init__.py').touch()
    (tmp_path / 'notebooks').mkdir()
    monkeypatch.chdir(tmp_path / 'notebooks')
    monkeypatch.setattr(importlib.util, 'find_spec',
                        lambda name: None if name == 'synthts' else 1)
    calls = []
    monkeypatch.setattr(subprocess, 'check_call', calls.append)
    exec(bootstrap(), {})
    assert calls[0][-1] == str(tmp_path.resolve())
    assert SOURCE not in calls[0]


def test_existing_install_only_adds_missing_helpers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(importlib.util, 'find_spec',
                        lambda name: None if name == 'yaml' else 1)
    calls = []
    monkeypatch.setattr(subprocess, 'check_call', calls.append)
    exec(bootstrap(), {})
    assert calls == [[sys.executable, '-m', 'pip', 'install', '--quiet',
                      'pyyaml>=6.0']]


def test_complete_install_does_not_run_pip(monkeypatch):
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: 1)
    calls = []
    monkeypatch.setattr(subprocess, 'check_call', calls.append)
    exec(bootstrap(), {})
    assert not calls


def test_installation_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None)

    def fail(command):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, 'check_call', fail)
    with pytest.raises(subprocess.CalledProcessError):
        exec(bootstrap(), {})
