"""Tests for nonnegative matrix check function."""

import numpy as np
import pytest

from toqito.matrix_props import is_nonnegative


def test_identity():
    """Check an identity matrix is nonnegative as expected."""
    assert is_nonnegative(np.identity(3))
