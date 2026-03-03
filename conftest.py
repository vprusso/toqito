"""Root conftest for toqito test suite."""

import pytest


def pytest_addoption(parser):
    """Add --runslow command-line option."""
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    """Register the slow marker."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --runslow is given."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
