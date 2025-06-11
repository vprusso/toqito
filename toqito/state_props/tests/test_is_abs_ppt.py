"""Test is_abs_ppt."""

import numpy as np
import pytest

from toqito.rand import random_psd_operator
from toqito.state_props import is_abs_ppt


def test_is_hermitian():
    """Test that a skew-symmetric matrix returns False."""
    mat = np.triu(np.random.rand(100, 100))
    mat += -mat.T  # Make it skew-symmetric
    assert not is_abs_ppt(mat)


def test_is_positive_semidefinite():
    """Test that a non-PSD matrix returns False."""
    mat = -random_psd_operator(100)  # Make a negative semidefinite matrix
    assert not is_abs_ppt(mat)


def test_in_separable_ball():
    """Test that matrix in separable ball returns True."""
    mat = np.identity(4) @ np.diag(np.array([1, 1, 1, 0])) / 3 @ np.identity(4).conj().T
    assert is_abs_ppt(mat)


def test_not_absolutely_ppt():
    """Test that a random PSD matrix is not PPT. Passes with high probability."""
    mat = random_psd_operator(40)
    assert not is_abs_ppt(mat)


@pytest.mark.parametrize(
    "mat, dim, error_msg",
    [
        # Invalid subsystem dimension
        (np.identity(4), 3, "dim must divide the dimensions of the matrix"),
        # Invalid non-square matrix
        (np.arange(12).reshape((3, 4)), None, "Matrix must be square"),
    ],
)
def test_invalid(mat, dim, error_msg):
    """Test error-checking."""
    with pytest.raises(ValueError, match=error_msg):
        is_abs_ppt(mat, dim)
