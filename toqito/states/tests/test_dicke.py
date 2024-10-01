"""Test dicke."""

import numpy as np
import pytest
from toqito.states import dicke


@pytest.mark.parametrize(
    "num_qubit, num_excited, expected_state",
    [
        # Dicke state for 3 qubits and 1 excitation
        (3, 1, np.array([0, 1/np.sqrt(3), 1/np.sqrt(3), 0, 1/np.sqrt(3), 0, 0, 0])),
        # Dicke state for 4 qubits and 2 excitations (corrected expected state)
        (4, 2, np.array([0, 0, 0, 1/np.sqrt(6), 0, 1/np.sqrt(6), 1/np.sqrt(6), 0,
                         0, 1/np.sqrt(6), 1/np.sqrt(6), 0, 1/np.sqrt(6), 0, 0, 0])),
    ],
)
def test_dicke_state(num_qubit, num_excited, expected_state):
    """Test that dicke_state produces the correct vector state."""
    np.testing.assert_array_almost_equal(dicke(num_qubit, num_excited), expected_state, decimal=6)


@pytest.mark.parametrize(
    "num_qubit, num_excited, expected_dm_shape",
    [
        # Density matrix for 3 qubits and 1 excitation
        (3, 1, (8, 8)),
        # Density matrix for 4 qubits and 2 excitations
        (4, 2, (16, 16)),
    ],
)
def test_dicke_state_density_matrix(num_qubit, num_excited, expected_dm_shape):
    """Test that dicke_state returns correct density matrix dimensions."""
    dm = dicke(num_qubit, num_excited, return_dm=True)
    assert dm.shape == expected_dm_shape
    assert np.isclose(np.trace(dm), 1)  # Ensure the trace of the density matrix is 1


@pytest.mark.parametrize(
    "num_qubit, num_excited",
    [
        # Number of excitations exceeds the number of qubits
        (2, 3),
    ],
)
def test_dicke_state_invalid_input(num_qubit, num_excited):
    """Test that dicke_state raises an error for invalid input."""
    with pytest.raises(ValueError):
        dicke(num_qubit, num_excited)
