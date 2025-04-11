"""Tests for operator_sinkhorn."""

import numpy as np

from toqito.channels.partial_trace import partial_trace
from toqito.matrices.operator_sinkhorn import operator_sinkhorn
from toqito.rand import random_density_matrix
from toqito.states import bell


def test_operator_sinkhorn_unitary_invariance():
    """Test invariance of Operator Sinkhorn on swapping subsystems."""
    rho = random_density_matrix(4)

    U = np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))  # Swap subsystem
    rho_new = U @ rho @ U.conj().T
    sigma_new, F_new = operator_sinkhorn(rho_new)
    sigma_original, F_org = operator_sinkhorn(rho)
    np.testing.assert_allclose(sigma_new, U @ sigma_original @ U.conj().T)

def test_operator_sinkhorn_bipartite_partial_trace():
    """Test operator Sinkhorn partial trace on a bipartite system."""
    # Generate a random density matrix for a 3x3 system (9-dimensional).
    rho = random_density_matrix(9)
    sigma, F = operator_sinkhorn(rho)

    # Expected partial trace should be proportional to identity matrix.
    expected_identity = np.eye(3) * (1 / 3)

    # Partial trace on the first subsystem.
    pt = partial_trace(sigma, 0, [3, 3])
    pt_rounded = np.around(pt, decimals=2)

    # Check that partial trace matches the expected identity.
    np.testing.assert_array_almost_equal(pt_rounded, expected_identity, decimal=2)

def test_operator_sinkhorn_tripartite_partial_trace():
    """Test operator Sinkhorn partial trace on a tripartite system."""
    # Generate a random density matrix for a 2x2x2 system (8-dimensional).
    rho = random_density_matrix(8)
    sigma, _ = operator_sinkhorn(rho, [2, 2, 2])

    # Expected partial trace should be proportional to identity matrix.
    expected_identity = np.eye(2) * (1 / 2)

    # Partial trace on the first and third subsystems.
    pt = partial_trace(sigma, [0, 2], [2, 2, 2])
    pt_rounded = np.around(pt, decimals=2)

    # Check that partial trace matches the expected identity.
    np.testing.assert_array_almost_equal(pt_rounded, expected_identity, decimal=2)

def test_operator_sinkhorn_singular_matrix():
    """Test operator Sinkhorn with a singular matrix that triggers LinAlgError."""
    # Create a valid 4x4 singular matrix (non-invertible).
    rho = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])  # This matrix is singular

    try:
        operator_sinkhorn(rho, dim=[2, 2])
    except ValueError as e:
        expected_msg = (
            "The operator Sinkhorn iteration does not converge for RHO. "
            "This is often the case if RHO is not of full rank."
        )
        assert str(e) == expected_msg

def test_operator_sinkhorn_invalid_single_dim():
    """Test operator Sinkhorn when a single number is passed as `dim` and it is invalid."""
    rho = random_density_matrix(8)  # 8-dimensional density matrix

    # The dimension `3` does not divide evenly into `8`, so we expect an error
    try:
        operator_sinkhorn(rho, dim=[3])
    except ValueError as e:
        expected_msg = (
            "If dim is of size 1, rho must be square and dim[0] must evenly divide length(rho); "
            "please provide the dim array containing the dimensions of the subsystems."
        )
        assert str(e) == expected_msg

def test_operator_sinkhorn_invalid_dim_array():
    """Test operator Sinkhorn when product of the dim array does not match the density matrix dims."""
    dX = 8
    dim1 = [4, 3, 2] # 4*3*2 != 8

    rho = random_density_matrix(8)  # 8-dimensional density matrix

    # The dimension `4` does not divide evenly into `8`, so we expect an error
    try:
        operator_sinkhorn(rho, dim=dim1)
    except ValueError as e:
        expected_msg = (
            f"Product of dimensions {dim1} does not match rho dimension {dX}."
        )
        assert str(e) == expected_msg

def test_operator_sinkhorn_valid_single_dim():
    """Test operator Sinkhorn when a single valid number is passed as `dim`."""
    rho = random_density_matrix(9)  # 9-dimensional density matrix

    # The dimension `3` divides evenly into `9`, so no error should occur
    sigma, F = operator_sinkhorn(rho, dim=[3])

    # Check that sigma is a valid density matrix with trace equal to 1
    np.testing.assert_almost_equal(np.trace(sigma), 1)

def test_operator_sinkhorn_max_mixed():
    """Test operator Sinkhorn on a maximally mixed bipartite state. Should be invariant."""
    rho = np.eye(9)  # 9-dimensional density matrix

    # The dimension `3` divides evenly into `9`, so no error should occur
    sigma, F = operator_sinkhorn(rho, dim=[3])

    # Check that rho is invariant after sinkhorn operation
    np.testing.assert_almost_equal(sigma, rho)

def test_operator_sinkhorn_max_entangled():
    """Test operator Sinkhorn on a maximally entangled bipartite state. Should be invariant."""
    # function should return the inintial state since it already satisfies the trace property

    u0 = bell(0)
    rho = u0 @ u0.conj().T

    sigma, F = operator_sinkhorn(rho, dim=[2])

    # Check that rho is invariant after sinkhorn operation
    np.testing.assert_almost_equal(sigma, rho)

def test_operator_sinkhorn_non_square_rho():
    """Test operator sinkhorn on non-square input matrix."""
    # function should raise a ValueError

    rho = np.random.rand(4, 5)
    try:
        operator_sinkhorn(rho)
    except ValueError as e:
        expected_msg = (
            "Input 'rho' must be a square matrix."
        )
        assert str(e) == expected_msg

def test_operator_sinkhorn_max_iterations():
    """Test operator sinkhorn on insufficient iteration limit."""
    # function should raise a RuntimeError

    rho_random = random_density_matrix(4, seed=42)
    try:
        operator_sinkhorn(rho=rho_random, dim=[2, 2], max_iterations=20)
    except RuntimeError as e:
        expected_msg = (
            "operator_sinkhorn did not converge within 20 iterations."
        )
        assert str(e) == expected_msg
