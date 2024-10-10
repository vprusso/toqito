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
    sigma, F = operator_sinkhorn(rho, [2, 2, 2])

    # Expected partial trace should be proportional to identity matrix.
    expected_identity = np.eye(2) * (1 / 2)

    # Partial trace on the first and third subsystems.
    pt = partial_trace(sigma, [0, 2], [2, 2, 2])
    pt_rounded = np.around(pt, decimals=2)

    # Check that partial trace matches the expected identity.
    np.testing.assert_array_almost_equal(pt_rounded, expected_identity, decimal=2)
