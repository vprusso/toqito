"""Test the public API of the channel_props package."""

import importlib

import pytest

from toqito import channel_props


@pytest.mark.parametrize("name", channel_props.__all__)
def test_public_function_not_shadowed_by_submodule(name):
    """Importing a submodule directly must not replace the function of the same name."""
    importlib.import_module(f"toqito.channel_props.{name}")
    func = getattr(channel_props, name)
    assert callable(func)
    assert func.__name__ == name
