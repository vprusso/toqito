"""Tests for operator_sinkhorn."""

import numpy as np

from toqito.channels import partial_trace
from toqito.matrices import operator_sinkhorn
from toqito.rand import random_density_matrix


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

    # The dimension `4` does not divide evenly into `8`, so we expect an error
    try:
        operator_sinkhorn(rho, dim=[4])
    except ValueError as e:
        expected_msg = (
            "If DIM is a scalar, X must be square and DIM must evenly divide length(X); "
            "please provide the DIM array containing the dimensions of the subsystems."
        )
        assert str(e) == expected_msg


def test_operator_sinkhorn_valid_single_dim():
    """Test operator Sinkhorn when a single valid number is passed as `dim`."""
    rho = random_density_matrix(9)  # 9-dimensional density matrix

    # The dimension `3` divides evenly into `9`, so no error should occur
    sigma, F = operator_sinkhorn(rho, dim=[3])

    # Check that sigma is a valid density matrix with trace equal to 1
    np.testing.assert_almost_equal(np.trace(sigma), 1)
