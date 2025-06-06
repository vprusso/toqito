"""Tests for operator_sinkhorn."""

import warnings

import numpy as np
import pytest

from toqito.matrices import operator_sinkhorn
from toqito.matrix_ops import partial_trace
from toqito.rand import random_density_matrix
from toqito.states import bell


def test_operator_sinkhorn_unitary_invariance():
    """Test invariance of operator Sinkhorn on swapping subsystems."""
    rho = random_density_matrix(4)

    # Swap subsystems.
    U = np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))
    rho_new = U @ rho @ U.conj().T
    sigma_new, local_ops_new = operator_sinkhorn(rho_new)
    sigma_old, local_ops_old = operator_sinkhorn(rho)
    assert pytest.approx(U @ sigma_old @ U.conj().T) == sigma_new


def test_operator_sinkhorn_trace_property():
    """Test the trace property of operator Sinkhorn when a valid inputs are passed."""
    rho = random_density_matrix(9)

    # The dimension `3` divides evenly into `9`, so no error should occur.
    sigma, local_ops = operator_sinkhorn(rho, dim=[3])

    # Check that sigma is a valid density matrix with trace equal to 1.
    assert pytest.approx(np.trace(sigma), rel=1e-7) == 1


# Test partial traces for bipartite and tripartite systems.
@pytest.mark.parametrize(
    "rho, dim, expected_identity, target_subsys",
    [
        # Random bipartite system with (3x3 qubit, 9 dimensional).
        (random_density_matrix(9), [3, 3], np.eye(3) * (1 / 3), 0),
        # Random tripartite system with (2x2x2 qubit, 8 dimensional).
        (random_density_matrix(8), [2, 2, 2], np.eye(2) * (1 / 2), [0, 2]),
    ],
)
def test_operator_sinkhorn_partial_trace(rho, dim, expected_identity, target_subsys):
    """Test partial traces for bipartite and tripartite systems."""
    sigma, local_ops = operator_sinkhorn(rho=rho, dim=dim)

    # Partial trace on the first subsystem.
    pt = partial_trace(sigma, target_subsys, dim=dim)
    pt_rounded = np.around(pt, decimals=3)

    # Check that partial trace matches the expected identity.
    assert pytest.approx(pt_rounded, rel=1e-2) == expected_identity


# Test error handling in different scenarios.
@pytest.mark.parametrize(
    "rho, dim, iters, error_type, expected_msg",
    [
        # Singular matrix.
        (
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            [2, 2],
            10_000,
            ValueError,
            (
                "The operator Sinkhorn iteration does not converge for rho. "
                "This is often the case if rho is not of full rank."
            ),
        ),
        # Invalid dim array with size 1.
        (
            random_density_matrix(8),
            [3],
            10_000,
            ValueError,
            (
                "If dim is of size 1, rho must be square and dim[0] must evenly divide rho.shape[0]; "
                "please provide the dim array containing the dimensions of the subsystems."
            ),
        ),
        # Invalid dim array - when product of the dim array != density matrix dims.
        (
            random_density_matrix(8),
            [4, 3, 2],  # Since 4*3*2 != 8.
            10_000,
            ValueError,
            "Product of dimensions [4, 3, 2] does not match rho dimension 8.",
        ),
        # Non square matrix input.
        (np.random.rand(4, 5), [2, 2], 10_000, ValueError, "Input 'rho' must be a square matrix."),
        # Insufficient iteration limit.
        (
            random_density_matrix(4, seed=42),
            [2, 2],
            20,
            RuntimeError,
            "operator_sinkhorn did not converge within 20 iterations.",
        ),
    ],
)
def test_operator_sinkhorn_errors(rho, dim, iters, error_type, expected_msg):
    """Test error handling in different scenarios."""
    try:
        operator_sinkhorn(rho=rho, dim=dim, max_iterations=iters)
    except error_type as e:
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
    "rho, dim",
    [
        # Maximally mixed state - already satisfies the required condition. So the input is invariant.
        (np.eye(9), [3]),
        # Maximally entangled state - has local marginals proportional to identity. So the input is invariant.
        ((bell(0)) @ ((bell(0)).conj().T), [2]),
    ],
)
def test_operator_sinkhorn_mix_ent(rho, dim):
    """Test outputs on maximally mixed and entangled states."""
    sigma, local_ops = operator_sinkhorn(rho=rho, dim=dim)

    # Check that rho is invariant after sinkhorn operation.
    assert pytest.approx(sigma, rel=1e-7) == rho
