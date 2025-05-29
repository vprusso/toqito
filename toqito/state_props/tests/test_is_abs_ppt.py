"""Unit tests for the is_abs_ppt function in toqito.state_props."""

import re

import numpy as np
import pytest

from toqito.state_props import is_abs_ppt
from toqito.states import bell, max_mixed


@pytest.mark.parametrize("dims, n, expected_result", [([2, 2], 4, True), ([2, 3], 6, True), ([3, 3], 9, True)])
def test_maximally_mixed_states(dims, n, expected_result):
    """Test that maximally mixed states are absolutely PPT for supported dimensions."""
    rho = max_mixed(n)
    assert is_abs_ppt(rho, dims) == expected_result


@pytest.mark.parametrize(
    "matrix, dims, expected_result",
    [
        # 2x2: Bell state (not absolutely PPT)
        (bell(2) @ bell(2).conj().T, [2, 2], False),
        # 2x3: Diagonal state with sorted eigenvalues (not absolutely PPT)
        (np.diag([0.4, 0.2, 0.15, 0.1, 0.1, 0.05]), [2, 3], False),
        # 2x3: Diagonal state with unsatisfying spectrum (not absolutely PPT)
        (np.diag([0.7, 0.1, 0.1, 0.05, 0.03, 0.02]), [2, 3], False),
        # 3x3: Diagonal state with sorted eigenvalues (absolutely PPT)
        (np.diag([1 / 9] * 9), [3, 3], True),
        # 3x3: Diagonal state with unsatisfying spectrum (not absolutely PPT)
        (np.diag([0.5, 0.2, 0.1, 0.07, 0.05, 0.03, 0.03, 0.01, 0.01]), [3, 3], False),
        # 2x2: Separable pure product state (not absolutely PPT)
        (np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), [2, 2], False),
        # 2x3: Separable pure product state (not absolutely PPT)
        (np.eye(6)[[0], :].T @ np.eye(6)[[0], :], [2, 3], False),
        # 2x2: Permuted maximally mixed state (should be absolutely PPT)
        (np.eye(4)[[2, 0, 3, 1], :][:, [2, 0, 3, 1]] * 0.25, [2, 2], True),
        # 3x3: Block-diagonal state with two large and one small block (not absolutely PPT)
        (np.diag([0.45, 0.45, 0.05, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005]), [3, 3], False),
        # 3x3: State with one eigenvalue close to 1 (not absolutely PPT)
        (np.diag([0.97, 0.01, 0.01, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001]), [3, 3], False),
    ],
)
def test_known_absppt_and_non_absppt(matrix, dims, expected_result):
    """Test known absolutely PPT and non-absolutely PPT states for 2x2, 2x3, 3x3, and edge cases."""
    # Normalize if not already
    matrix = matrix / np.trace(matrix)
    assert is_abs_ppt(matrix, dims) == expected_result


@pytest.mark.parametrize(
    "matrix, error_msg",
    [
        (np.ones((3, 4)), "Input matrix must be square."),
        (np.array([[0, 1], [0, 0]]), "Input matrix must be Hermitian."),
        (np.eye(4) * 2, "Input matrix must have trace 1"),
        (np.array([[1, 0], [0, -1]]), "Input matrix must be positive semidefinite."),
    ],
)
def test_invalid_input(matrix, error_msg):
    """Test that invalid input raises appropriate ValueError with correct message."""
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        is_abs_ppt(matrix)


def test_large_dim_returns_none():
    """Test that unsupported large dimensions return None."""
    rho = max_mixed(16)
    assert is_abs_ppt(rho, [4, 4]) is None


def test_abs_ppt_constraints_not_implemented():
    """Test NotImplementedError is raised for dimensions > 3x3 in _abs_ppt_constraints."""
    from toqito.state_props.is_abs_ppt import _abs_ppt_constraints
    eigvals = np.ones(16)
    with pytest.raises(NotImplementedError,
                       match="Absolutely PPT constraints for dimensions > 3x3 are not implemented."):
        _abs_ppt_constraints(eigvals, [4, 4])


def test_dimension_mismatch():
    """Test ValueError is raised when provided dimensions do not match matrix size."""
    rho = max_mixed(4)
    with pytest.raises(ValueError, match="Dimensions 2 x 3 do not match matrix size 4."):
        is_abs_ppt(rho, [2, 3])


def test_dim_as_int():
    """Test that dim as int is handled correctly (should be [2,2] for 4x4)."""
    rho = max_mixed(4)
    assert is_abs_ppt(rho, 2) is True


def test_numerical_tolerance_edge_case():
    """Test that a non-PSD input matrix raises ValueError."""
    mat = np.eye(4) / 4
    mat[0, 0] -= 0.3  # Make the smallest eigenvalue negative
    mat /= np.trace(mat)  # Ensure trace is 1
    with pytest.raises(ValueError, match="Input matrix must be positive semidefinite."):
        is_abs_ppt(mat)


def test_complex_non_hermitian():
    """Test that a non-Hermitian complex matrix raises the correct error."""
    mat = np.array([[0, 1j], [0, 0]])
    with pytest.raises(ValueError, match="Input matrix must be Hermitian."):
        is_abs_ppt(mat)
