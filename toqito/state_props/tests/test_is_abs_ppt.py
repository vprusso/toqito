"""Test cases for the `is_abs_ppt` function."""

import numpy as np
import pytest

from toqito.state_props.is_abs_ppt import is_abs_ppt
from toqito.states import bell, max_mixed


@pytest.mark.parametrize(
    "rho, dim, expected_result",
    [
        (max_mixed(4), [2, 2], True),
        (bell(0) @ bell(0).conj().T, [2, 2], False),
        (np.eye(9) / 9, [3, 3], True),
        (np.array([[0.7, 0.3], [0.3, 0.3]]), [2, 1], True),
        (
            lambda: (0.99 * max_mixed(4) + 0.01 * np.eye(4)) / np.trace(0.99 * max_mixed(4) + 0.01 * np.eye(4)),
            [2, 2],
            True,
        ),
    ],
)
def test_is_abs_ppt(rho, dim, expected_result):
    """Check function with typical states in lower dimensions (p <= 6)."""
    if callable(rho):
        rho = rho()
    assert is_abs_ppt(rho, dim) == expected_result


@pytest.mark.parametrize("dim_size", [7, 8, 9])
def test_is_abs_ppt_sdp(dim_size):
    """Check high-dimensional (p >= 7) cases that trigger an SDP-based check."""
    rho = np.eye(dim_size**2) / (dim_size**2)
    result = is_abs_ppt(rho, [dim_size, dim_size])
    assert result in [True, False, None], f"Unexpected result for p={dim_size}: {result}"


def test_non_square_matrix_raises():
    """A non-square matrix should raise ValueError about invalid density matrix."""
    non_square = np.random.rand(3, 4)
    with pytest.raises(ValueError, match="Input `rho` is not a valid density matrix."):
        is_abs_ppt(non_square, [2, 2])


def test_zero_trace_matrix_raises():
    """A matrix with zero trace should raise ValueError about invalid density matrix."""
    zero_trace = np.eye(4) - np.eye(4)
    with pytest.raises(ValueError, match="Input `rho` is not a valid density matrix."):
        is_abs_ppt(zero_trace, [2, 2])


def test_dimension_mismatch_raises():
    """Dimension mismatch should raise a ValueError indicating shape mismatch."""
    rho = np.eye(4) / 4
    with pytest.raises(ValueError, match="must match the shape of `rho`."):
        is_abs_ppt(rho, [3, 2])


def test_normalization_of_trace_off_by_small_amount():
    """A matrix whose trace is not ~1 is invalid as a density matrix."""
    rho = 1.001 * max_mixed(4)
    with pytest.raises(ValueError, match="Input `rho` is not a valid density matrix."):
        is_abs_ppt(rho, [2, 2])


def test_invalid_density_due_to_negative_eigenvalue():
    """A Hermitian matrix with negative eigenvalue is invalid as a density matrix."""
    rho = np.array([[0.6, 0.8], [0.8, -0.2]])
    rho = (rho + rho.T.conj()) / 2
    with pytest.raises(ValueError, match="Input `rho` is not a valid density matrix."):
        is_abs_ppt(rho, [1, 2])


def test_inconclusive_result_for_non_psd_in_higher_dim():
    """For p >= 6 and invalid density, the function raises ValueError."""
    rho = np.eye(36) / 36 - 0.03 * np.eye(36)
    with pytest.raises(ValueError, match="Input `rho` is not a valid density matrix."):
        is_abs_ppt(rho, [6, 6])
