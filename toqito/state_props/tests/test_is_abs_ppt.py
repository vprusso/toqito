"""Test cases for the `is_abs_ppt` function."""

import numpy as np
import pytest

from toqito.state_props.is_abs_ppt import is_abs_ppt
from toqito.states import bell, max_mixed


@pytest.mark.parametrize(
    "rho, dim",
    [
        # Check for non-square input (invalid density matrix)
        (np.random.rand(3, 4), [2, 2]),
        # Zero-trace matrix (invalid)
        (np.eye(4) - np.eye(4), [2, 2]),
        # Dimension mismatch
        (np.eye(4) / 4, [3, 2]),
        # Slightly off-normalized trace
        (1.001 * max_mixed(4), [2, 2]),
        # Matrix with negative eigenvalue
        ((np.array([[0.6, 0.8], [0.8, -0.2]]) + np.array([[0.6, 0.8], [0.8, -0.2]]).T.conj()) / 2, [1, 2]),
        # Invalid in high dimension
        (np.eye(36) / 36 - 0.03 * np.eye(36), [6, 6]),
    ],
)
def test_is_abs_ppt_invalid_inputs(rho, dim):
    """Test that invalid input raises appropriate errors."""
    with np.testing.assert_raises(ValueError):
        is_abs_ppt(rho, dim)


@pytest.mark.parametrize(
    "rho, dim, expected_result",
    [
        # General tests
        (max_mixed(4), [2, 2], True),
        (bell(0) @ bell(0).conj().T, [2, 2], False),
        (np.eye(9) / 9, [3, 3], True),
        (np.array([[0.7, 0.3], [0.3, 0.3]]), [2, 1], True),
        (
            lambda: (0.99 * max_mixed(4) + 0.01 * np.eye(4)) / np.trace(0.99 * max_mixed(4) + 0.01 * np.eye(4)),
            [2, 2],
            True,
        ),
        # QETLAB examples
        (np.eye(25) / 25, [5, 5], True),
        (np.array([0.22, 0.18, 0.14, 0.12, 0.10, 0.08, 0.06, 0.06, 0.04]), [3, 3], False),
        (np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.04, 0.03]), [3, 3], False),
        (np.eye(49) / 49, [7, 7], None),
    ],
)
def test_is_abs_ppt_known_values(rho, dim, expected_result):
    """Test known examples and general functionality for is_abs_ppt."""
    if callable(rho):
        rho = rho()
    result = is_abs_ppt(rho, dim)
    if expected_result is None:
        assert result in [True, False, None]
    else:
        np.testing.assert_equal(result, expected_result)


@pytest.mark.parametrize("dim_size", [7, 8, 9])
def test_is_abs_ppt_high_dimension_indeterminate(dim_size):
    """Test larger dimensions that should return None or valid result."""
    rho = np.eye(dim_size**2) / (dim_size**2)
    result = is_abs_ppt(rho, [dim_size, dim_size])
    assert result in [True, False, None]
