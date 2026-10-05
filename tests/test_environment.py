import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    ["pandas", "numpy", "scipy", "requests", "streamlit", "plotly"],
)
def test_required_dependencies_are_importable(module_name):
    assert importlib.import_module(module_name) is not None