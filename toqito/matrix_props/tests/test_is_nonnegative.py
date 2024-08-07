"""Tests for nonnegative and doubly nonnegative matrix check function."""

import numpy as np
import pytest

from toqito.matrix_props import is_nonnegative


def test_identity():
    """Check an identity matrix is nonnegative as expected."""
    assert is_nonnegative(np.identity(3))
    assert is_nonnegative(np.identity(3), "nonnegative")
    assert is_nonnegative(np.identity(3), "doubly")


@pytest.mark.parametrize("bad_type", [("l"), ("r"), (1), (), ("d")])
def test_true(bad_type):
    """Check if error raised correctly for invalid nonnegative matrix type."""
    with pytest.raises(TypeError):
        is_nonnegative(np.identity(3), bad_type)
