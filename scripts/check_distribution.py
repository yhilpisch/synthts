"""Inspect built archives and enforce publication metadata when requested."""

import argparse
from email.parser import Parser
from pathlib import Path
import tarfile
import zipfile


def check(directory, publication=False):
    """Check one matching source/wheel candidate without installing it."""
    directory = Path(directory)
    wheel, = directory.glob('*.whl')
    sdist, = directory.glob('*.tar.gz')
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_path, = [n for n in names if n.endswith('.dist-info/METADATA')]
        metadata = Parser().parsestr(archive.read(metadata_path).decode())
        assert metadata['Name'] == 'synthts'
        assert 'synthts/artifacts.py' in names
        assert 'synthts/_version.py' in names
        assert all(n.startswith(('synthts/', 'synthts-')) for n in names)
        assert not any(n.endswith(('.png', '.pt', '.csv')) for n in names)
        if publication:
            assert metadata['License-Expression'], 'licence is not selected'
            assert any(n.endswith('/LICENSE') for n in names), (
                'missing LICENSE')
            urls = metadata.get_all('Project-URL', [])
            assert any(u.startswith('Source, https://github.com/')
                       for u in urls)
            assert any(u.startswith('Issues, https://github.com/')
                       for u in urls)
    with tarfile.open(sdist) as archive:
        files = {m.name.split('/', 1)[-1]: m for m in archive.getmembers()
                 if m.isfile()}
        assert 'pyproject.toml' in files
        assert 'tests/test_examples.py' in files
        assert 'notebooks/04_timegan.ipynb' in files
        assert not any('.ipynb_checkpoints/' in n for n in files)
        assert not any(n.startswith(('_ignore/', 'sweeps/results/', 'data/'))
                       for n in files)
        source_metadata = Parser().parsestr(
            archive.extractfile(files['PKG-INFO']).read().decode())
        assert source_metadata['Version'] == metadata['Version']
        assert source_metadata['Name'] == metadata['Name']
    print(f"Validated {metadata['Name']} {metadata['Version']} archives")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--publication', action='store_true')
    args = parser.parse_args()
    check(args.directory, args.publication)
