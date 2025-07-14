"""Test the commutant function."""

import numpy as np
import pytest

from toqito.matrix_props.commutant import commutant
from toqito.perms.vec import vec


@pytest.mark.parametrize(
    "matrices, expected_size",
    [
        # Pauli-Z: Expect diagonal matrices.
        ([np.array([[1, 0], [0, -1]])], 2),
        # Pauli-X: Expect identity and X.
        ([np.array([[0, 1], [1, 0]])], 2),
        # Pauli-Y: Expect I and Y.
        ([np.array([[0, -1j], [1j, 0]])], 2),
        # Identity matrix: Should commute with everything.
        ([np.eye(2)], 4),
    ],
)
def test_commutant_output_size(matrices, expected_size):
    """Check if the number of commutant basis elements is as expected."""
    comm_basis = commutant(matrices)
    assert len(comm_basis) == expected_size


@pytest.mark.parametrize(
    "matrices",
    [
        # Pauli-Z.
        ([np.array([[1, 0], [0, -1]])]),
        # Pauli-X.
        ([np.array([[0, 1], [1, 0]])]),
    ],
)
def test_commutation_property(matrices):
    """Ensure all computed matrices commute with the given matrices."""
    comm_basis = commutant(matrices)
    for mat in matrices:
        for B in comm_basis:
            assert np.allclose(mat @ B, B @ mat)


def test_commutant_identity_dim():
    """For the identity matrix, the commutant should be the full space."""
    identity = np.eye(2)
    comm_basis = commutant(identity)
    assert len(comm_basis) == 4


def test_commutant_identity():
    """For the identity matrix, check if the commutant contains the expected basis matrices."""
    identity = np.eye(2)
    comm_basis = commutant(identity)

    expected_commutant = [
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, 0]]),
        np.array([[0, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
    ]

    assert len(comm_basis) == len(expected_commutant)

    for expected_matrix in expected_commutant:
        assert any(np.allclose(expected_matrix, computed_matrix) for computed_matrix in comm_basis)


def test_bicommutant_m3():
    """Verify that the bicommutant spans the same set."""
    A = [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
        np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
    ]

    B = commutant(commutant(A))  # Compute bicommutant

    assert len(B) == len(A)

    for expected_matrix in A:
        assert any(
            np.allclose(expected_matrix, computed_matrix) or np.allclose(expected_matrix, -computed_matrix)
            for computed_matrix in B
        )


def test_explicit_commutant_condition():
    """Check that the computed commutant matrices satisfy the commutant equation."""
    # Pauli-Z
    matrices = [np.array([[1, 0], [0, -1]])]

    comm_basis = commutant(matrices)
    dim = matrices[0].shape[0]

    for A in matrices:
        for B in comm_basis:
            # Compute the commutant condition: (A ⊗ I - I ⊗ A^T) vec(B) = 0
            comm_matrix = np.kron(A, np.eye(dim)) - np.kron(np.eye(dim), A.T)
            residual = comm_matrix @ vec(B)
            assert np.allclose(residual, 0)
