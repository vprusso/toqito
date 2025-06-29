"""Tests for nonnegative matrix check function."""

import numpy as np

from toqito.matrix_props import is_positive


def test_matrices():
    """Check an identity matrix is not positive as expected."""
    assert not is_positive(np.identity(3))

    assert is_positive(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
