"""Tests for matsumoto_fidelity."""

import cvxpy
import numpy as np
import pytest

from toqito.state_metrics import matsumoto_fidelity
from toqito.states import basis

rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])

e_0, e_1 = basis(2, 0), basis(2, 1)
rho1 = 3 / 4 * e_0 @ e_0.conj().T + 1 / 4 * e_1 @ e_1.conj().T
sigma1 = 2 / 3 * e_0 @ e_0.conj().T + 1 / 3 * e_1 @ e_1.conj().T

rho2 = 3 / 4 * e_0 @ e_0.conj().T + 1 / 4 * e_1 @ e_1.conj().T
sigma2 = 1 / 8 * e_0 @ e_0.conj().T + 7 / 8 * e_1 @ e_1.conj().T

rho5 = cvxpy.bmat([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])


@pytest.mark.parametrize(
    "input1, input2, expected",
    [
        # when inputs are the same
        (rho, rho, 1),
        # inputs are non-identical
        (rho1, sigma1, 0.996),
        (rho2, sigma2, 0.774),
        # when inputs are the same for cvxpy variable
        (rho5, rho5, 1),
    ],
)
def test_matsumoto_fidelity(input1, input2, expected):
    """Test function works as expected for valid inputs."""
    calculated_result = matsumoto_fidelity(input1, input2)
    assert abs(calculated_result - expected) <= 1e-03


def test_matsumoto_fidelity_identical_singular_states():
    """Test identical singular pure states have Matsumoto fidelity one."""
    rho = e_0 @ e_0.conj().T

    np.testing.assert_allclose(matsumoto_fidelity(rho, rho), 1)


def test_matsumoto_fidelity_orthogonal_singular_states():
    """Test orthogonal singular pure states have Matsumoto fidelity zero."""
    rho = e_0 @ e_0.conj().T
    sigma = e_1 @ e_1.conj().T

    np.testing.assert_allclose(matsumoto_fidelity(rho, sigma), 0, atol=1e-4)


def test_matsumoto_fidelity_noncommuting_singular_supports():
    """Two singular pure states with distinct, non-orthogonal supports (|0> and |+>).

    Their supports intersect only at the origin, so the matrix geometric mean is zero and the
    Matsumoto fidelity is zero. The closed-form pseudoinverse expression instead returns
    ``1/sqrt(2)`` here, so this exercises the exact SDP fallback for both-singular inputs.
    """
    plus = (e_0 + e_1) / np.sqrt(2)
    rho = e_0 @ e_0.conj().T
    sigma = plus @ plus.conj().T

    np.testing.assert_allclose(matsumoto_fidelity(rho, sigma), 0, atol=1e-3)


def test_matsumoto_fidelity_rank_deficient_matches_limit():
    """A rank-two pair in a three-dimensional space matches the eps-regularized limit value 1/2."""
    v = np.array([[0], [1], [1]]) / np.sqrt(2)
    rho = (basis(3, 0) @ basis(3, 0).conj().T + basis(3, 1) @ basis(3, 1).conj().T) / 2
    sigma = (basis(3, 0) @ basis(3, 0).conj().T + v @ v.conj().T) / 2

    np.testing.assert_allclose(matsumoto_fidelity(rho, sigma), 0.5, atol=1e-3)


rho3 = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
sigma3 = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])

rho4 = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
sigma4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@pytest.mark.parametrize(
    "input1, input2, expected_msg",
    [
        # non square dims
        (rho3, sigma3, "Matsumoto fidelity is only defined for density operators."),
        # invalid dims
        (rho4, sigma4, "InvalidDim: `rho` and `sigma` must be matrices of the same size."),
    ],
)
def test_matsumoto_fidelity_invalid_input(input1, input2, expected_msg):
    """Test function raises an error for invalid inputs."""
    with pytest.raises(ValueError, match=expected_msg):
        matsumoto_fidelity(input1, input2)
