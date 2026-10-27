"""Execute the offline README example against an installed wheel."""

import importlib.metadata
from pathlib import Path
import re
import sys

import synthts


if __name__ == '__main__':
    origin = Path(synthts.__file__).resolve()
    assert origin.is_relative_to(Path(sys.prefix).resolve()), origin
    assert synthts.__version__ == importlib.metadata.version('synthts')
    readme = Path(sys.argv[1]).read_text()
    example = re.search(r'```python\n(.*?)```', readme, flags=re.S).group(1)
    exec(compile(example, 'README.md offline example', 'exec'))
    print(f'Installed-package smoke passed: {origin}')
