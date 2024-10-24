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
    """Test functions works as expected for valid inputs."""
    calculated_result = matsumoto_fidelity(input1, input2)
    assert abs(calculated_result - expected) <= 1e-03


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
