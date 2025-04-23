"""Tests for operator_sinkhorn."""

import warnings

import numpy as np
import pytest

from toqito.channels.partial_trace import partial_trace
from toqito.matrices.operator_sinkhorn import operator_sinkhorn
from toqito.rand import random_density_matrix
from toqito.states import bell


def test_operator_sinkhorn_unitary_invariance():
    """Test invariance of operator Sinkhorn on swapping subsystems."""
    rho = random_density_matrix(4)

    U = np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))  # Swap subsystem.
    rho_new = U @ rho @ U.conj().T
    sigma_new, local_ops_new = operator_sinkhorn(rho_new)
    sigma_old, local_ops_old = operator_sinkhorn(rho)
    np.testing.assert_allclose(sigma_new, U @ sigma_old @ U.conj().T)


def test_operator_sinkhorn_trace_property():
    """Test the trace property of operator Sinkhorn when a valid inputs are passed."""
    rho = random_density_matrix(9)  # 9-dimensional density matrix.

    # The dimension `3` divides evenly into `9`, so no error should occur.
    sigma, local_ops = operator_sinkhorn(rho, dim=[3])

    # Check that sigma is a valid density matrix with trace equal to 1.
    np.testing.assert_almost_equal(np.trace(sigma), 1)


# Test partial traces for bipartite and tripartite systems.
@pytest.mark.parametrize(
    "test_input, expected",
    [
        # Random bipartite system with (3x3 qubit, 9 dimensional)
        ([random_density_matrix(9), [3, 3]], [np.eye(3) * (1 / 3), 0]),
        # Random tripartite system with (2x2x2 qubit, 8 dimensional).
        ([random_density_matrix(8), [2, 2, 2]], [np.eye(2) * (1 / 2), [0, 2]]),
    ],
)
def test_operator_sinkhorn_partial_trace(test_input, expected):
    """Test partial traces for bipartite and tripartite systems."""
    rho1, dim1 = test_input
    sigma, local_ops = operator_sinkhorn(rho=rho1, dim=dim1)

    # Expected partial trace should be proportional to identity matrix.
    expected_identity, target_subsys = expected

    # Partial trace on the first subsystem.
    pt = partial_trace(sigma, target_subsys, dim=dim1)
    pt_rounded = np.around(pt, decimals=2)

    # Check that partial trace matches the expected identity.
    np.testing.assert_allclose(pt_rounded, expected_identity, rtol=1e-2)


# Test error handling in different scenarios.
@pytest.mark.parametrize(
    "test_input,expected_msg",
    [
        # Singular matrix.
        (
            [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), [2, 2]],
            (
                "The operator Sinkhorn iteration does not converge for rho. "
                "This is often the case if rho is not of full rank."
            ),
        ),
        # Invalid dim array with size 1.
        (
            [random_density_matrix(8), [3]],
            (
                "If dim is of size 1, rho must be square and dim[0] must evenly divide rho.shape[0]; "
                "please provide the dim array containing the dimensions of the subsystems."
            ),
        ),
        # Invalid dim array - when product of the dim array != density matrix dims.
        (
            [random_density_matrix(8), [4, 3, 2]],  # Since 4*3*2 != 8.
            "Product of dimensions [4, 3, 2] does not match rho dimension 8.",
        ),
        # Non square matrix input.
        ([np.random.rand(4, 5), [2, 2]], "Input 'rho' must be a square matrix."),
    ],
)
def test_operator_sinkhorn_errors(test_input, expected_msg):
    """Test error handling in different scenarios."""
    # Unpack test inputs.
    rho, dim1 = test_input

    try:
        operator_sinkhorn(rho, dim=dim1)
    except ValueError as e:
        assert str(e) == expected_msg


def test_operator_sinkhorn_max_iterations():
    """Test operator sinkhorn on insufficient iteration limit."""
    rho_random = random_density_matrix(4, seed=42)
    num_iters = 20
    try:
        operator_sinkhorn(rho=rho_random, dim=[2, 2], max_iterations=num_iters)
    except RuntimeError as e:
        expected_msg = f"operator_sinkhorn did not converge within {num_iters} iterations."
        assert str(e) == expected_msg


def test_operator_sinkhorn_near_zero_trace():
    """Test if operator sinkhorn raises warnings on near zero output trace."""
    rho_random = np.array(
        [
            [np.finfo(float).eps, 0, 0, 0],
            [0, np.finfo(float).eps, 0, 0],
            [0, 0, np.finfo(float).eps, 0],
            [0, 0, 0, np.finfo(float).eps],
        ]
    )
    expected_msg = "Final trace is near zero, but initial trace was not. Result may be unreliable."

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            operator_sinkhorn(rho=rho_random, dim=[2, 2])
        except RuntimeWarning as warn_string:
            assert str(warn_string) == expected_msg


# Test outputs on maximally mixed and entangled states.
@pytest.mark.parametrize(
    "test_input",
    [
        # Maximally mixed state - should be invariant.
        ([np.eye(9), [3]]),
        # Maximally entangled state - should be invariant.
        ([(bell(0)) @ ((bell(0)).conj().T), [2]]),
    ],
)
def test_operator_sinkhorn_mix_ent(test_input):
    """Test outputs on maximally mixed and entangled states."""
    rho1, dim1 = test_input

    sigma, local_ops = operator_sinkhorn(rho=rho1, dim=dim1)

    # Check that rho is invariant after sinkhorn operation.
    np.testing.assert_allclose(sigma, rho1)
