"""Every subpackage must be importable as the first toqito import of a fresh interpreter."""

import subprocess
import sys

import pytest

import toqito


@pytest.mark.parametrize("subpackage", toqito.__all__)
def test_subpackage_imports_first(subpackage):
    """Importing a subpackage before any other must not trip a circular import between the package inits."""
    result = subprocess.run(
        [sys.executable, "-c", f"import toqito.{subpackage}"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
