"""Test cases for the `is_abs_ppt` function."""

import numpy as np
import pytest

from toqito.state_props.is_abs_ppt import is_abs_ppt
from toqito.states import bell, max_mixed


@pytest.mark.parametrize(
    "rho, dim, expected_result",
    [
        (max_mixed(4), [2, 2], 1),
        (bell(0) @ bell(0).conj().T, [2, 2], 0),
        (np.eye(9) / 9, [3, 3], 1),
        (np.random.rand(36, 36), [6, 6], -1),
        (np.array([[0.7, 0.3], [0.3, 0.3]]), [2, 1], 1),
        (0.99 * max_mixed(4) + 0.01 * np.eye(4), [2, 2], 1),
        (np.eye(49) / 49, [7, 7], -1),
    ],
)
def test_is_abs_ppt(rho, dim, expected_result):
    """Test function works as expected for practical computational cases."""
    assert is_abs_ppt(rho, dim) == expected_result


def test_non_square_matrix_raises():
    """Test that a non-square matrix raises a ValueError."""
    non_square = np.random.rand(3, 4)
    with pytest.raises(ValueError, match="must be square"):
        is_abs_ppt(non_square, [2, 2])


def test_zero_trace_matrix_raises():
    """Test that a matrix with zero trace raises a ValueError."""
    zero_trace = np.eye(4) - np.eye(4)
    with pytest.raises(ValueError, match="zero trace"):
        is_abs_ppt(zero_trace, [2, 2])


def test_dimension_mismatch_raises():
    """Test that a dimension mismatch raises a ValueError."""
    rho = np.eye(4) / 4
    with pytest.raises(ValueError, match="Dimension mismatch"):
        is_abs_ppt(rho, [3, 2])
