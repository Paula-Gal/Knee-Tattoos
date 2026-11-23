import importlib

import pytest

MODULES = ['io', 'detect', 'lift', 'table', 'evaluate', 'run']


@pytest.mark.parametrize('name', MODULES)
def test_module_imports(name):
    importlib.import_module(f'marker_tracking.{name}')
