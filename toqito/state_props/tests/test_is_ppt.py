"""Test is_ppt."""

import numpy as np
import pytest

from toqito.state_props import is_ppt
from toqito.states import bell, horodecki


@pytest.mark.parametrize(
    "mat, sys, dim, tol, expected_result",
    [
        (np.identity(9), 2, None, None, True),
        (np.identity(9), 2, [3], None, True),
        (np.identity(9), 2, [3], 1e-10, True),
        (bell(2) @ bell(2).conj().T, 2, None, None, False),
        (horodecki(0.5, [3, 3]), 2, None, None, True),
    ],
)
def test_is_ppt(mat, sys, dim, tol, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_equal(is_ppt(mat=mat, sys=sys, dim=dim, tol=tol), expected_result)


def test_is_ppt_non_square_matrix():
    """Test is_ppt raises ValueError for non-square input."""
    mat = np.array([[1, 0, 0],
                    [0, 1, 0]])

    with pytest.raises(ValueError):
        is_ppt(mat=mat, sys=2)


def test_is_ppt_one_dimensional_input():
    """Test is_ppt raises ValueError for 1D input."""
    mat = np.array([1, 2, 3, 4])

    with pytest.raises(ValueError):
        is_ppt(mat=mat, sys=2)


def test_is_ppt_invalid_dim():
    """Test is_ppt raises ValueError for invalid dim."""
    mat = np.eye(4)

    with pytest.raises(ValueError):
        is_ppt(mat=mat, sys=2, dim=[2, 3])


def test_is_ppt_non_numeric_input():
    """Test is_ppt raises TypeError for non-numeric input."""
    with pytest.raises((TypeError, ValueError)):
        is_ppt(mat="not a matrix", sys=2)
