"""Test is_abs_ppt."""

import numpy as np
import pytest

from toqito.rand import random_psd_operator
from toqito.state_props import is_abs_ppt


@pytest.mark.parametrize(
    "mat",
    [
        # Matrix satisfying Theorem 7.2 of :cite:`Jivulescu_2015_Reduction` is absolutely PPT
        np.identity(4) @ np.diag(np.array([1, 1, 1, 0])) / 3 @ np.identity(4).conj().T,
        # Matrix in separable ball is absolutely PPT
        np.diag([0.7, 0.7, 0.2, 0.2]) / 1.8,
        # Absolutely PPT 2 * 2 matrix which satisfies neither of the previous properties
        np.diag([0.42775974, 0.38590341, 0.11395246, 0.07238439]),
    ],
)
def test_absolutely_ppt(mat):
    """Test absolutely PPT matrices."""
    assert is_abs_ppt(mat)


def skew_symmetric(mat):
    """Make a matrix skew-symmetric."""
    return mat - mat.T


@pytest.mark.parametrize(
    "mat",
    [
        # Random PSD operator is not absolutely PPT with high probability
        random_psd_operator(100),
        # Negative semidefinite matrix is not absolutely PPT
        -random_psd_operator(100),
        # Skew-symmetric matrix is not absolutely PPT
        skew_symmetric(np.random.rand(100, 100)),
    ],
)
def test_not_absolutely_ppt(mat):
    """Test not absolutely PPT matrices."""
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
