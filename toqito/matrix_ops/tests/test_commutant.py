"""Test the commutant function."""
import numpy as np
import pytest
from toqito.matrix_ops.commutant import commutant

@pytest.mark.parametrize(
    "matrices, expected_size",
    [
        ([np.array([[1, 0], [0, -1]])], 2),  # Pauli-Z: Expect diagonal matrices
        ([np.array([[0, 1], [1, 0]])], 2),   # Pauli-X: Expect identity and X
        ([np.array([[0, -1j], [1j, 0]])], 2),  # Pauli-Y: Expect I and Y
        ([np.eye(2)], 4),  # Identity matrix: Should commute with everything
    ],
)
def test_commutant_output_size(matrices, expected_size):
    """Check if the number of commutant basis elements is as expected."""
    comm_basis = commutant(matrices)
    assert len(comm_basis) == expected_size, f"Expected {expected_size}, got {len(comm_basis)}"

@pytest.mark.parametrize(
    "matrices",
    [
        ([np.array([[1, 0], [0, -1]])]),  # Pauli-Z
        ([np.array([[0, 1], [1, 0]])]),   # Pauli-X
    ],
)
def test_commutation_property(matrices):
    """Ensure all computed matrices commute with the given matrices."""
    comm_basis = commutant(matrices)
    for mat in matrices:
        for B in comm_basis:
            assert np.allclose(mat @ B, B @ mat), f"Matrix does not commute:\n{B}"

def test_commutant_identity():
    """For the identity matrix, the commutant should be the full space."""
    identity = np.eye(2)
    comm_basis = commutant([identity])
    assert len(comm_basis) == 4, "Expected full space for identity matrix"
