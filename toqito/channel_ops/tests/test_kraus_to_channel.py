"""Tests for kraus to channel."""

import numpy as np
import pytest

import toqito.state_ops
from toqito.channel_ops import apply_channel, kraus_to_channel

dim = 2**2
kraus_list = [np.random.randint(-1, 4, (2, dim, dim)) for _ in range(12)]

vector = np.random.randint(-3, 3, (dim, 1))
dm = toqito.matrix_ops.to_density_matrix(vector)
vec_dm = toqito.matrix_ops.vec(dm)

# Random quantum test states (density matrices)
rho_1 = np.array([[0.5, 0.5], [0.5, 0.5]])
rho_2 = np.array([[0.5, 0], [0, 0.5]])
rho_3 = np.array([[1, 0], [0, 0]])
rho_4 = np.array([[0, 0], [0, 1]])

A = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
rho_5 = A @ A.conj().T
rho_5 /= np.trace(rho_5)  # Normalize trace to 1


p_1 = 0.1  # Probability of bit flip (Bit-Flip Channel)
kraus_operators_1 = [
    (np.sqrt(1 - p_1) * np.array([[1, 0], [0, 1]]), np.sqrt(1 - p_1) * np.array([[1, 0], [0, 1]])),  # Identity
    (np.sqrt(p_1) * np.array([[0, 1], [1, 0]]), np.sqrt(p_1) * np.array([[0, 1], [1, 0]])),  # Bit-flip (X)
]

p_2 = 0.2 # Depolarizing Channel
kraus_operators_2 = [
    (np.sqrt(1 - 3 * p_2 / 4) * np.eye(2), np.sqrt(1 - 3 * p_2 / 4) * np.eye(2)),  # Identity
    (np.sqrt(p_2 / 4) * np.array([[0, 1], [1, 0]]), np.sqrt(p_2 / 4) * np.array([[0, 1], [1, 0]])),  # X
    (np.sqrt(p_2 / 4) * np.array([[0, -1j], [1j, 0]]), np.sqrt(p_2 / 4) * np.array([[0, -1j], [1j, 0]])),  # Y
    (np.sqrt(p_2 / 4) * np.array([[1, 0], [0, -1]]), np.sqrt(p_2 / 4) * np.array([[1, 0], [0, -1]])),  # Z
]

kraus_operators_3 = [
    (np.array([[1, 0], [0, 0]]), np.array([[1, 0], [0, 0]])),  # Projection onto |0⟩
    (np.array([[0, 1], [0, 0]]), np.array([[0, 1], [0, 0]])),  # Projection onto |1⟩
]

@pytest.mark.parametrize(
    "kraus_list",
    [
        (kraus_list)
    ],
)
def test_kraus_to_channel(kraus_list):
    """Test kraus_tochannel works as expected for valid inputs."""
    calculated = kraus_to_channel(kraus_list)

    value = sum(A @ dm @ B.conj().T for A, B in kraus_list)

    assert toqito.matrix_ops.unvec(calculated @ vec_dm).all() == value.all()


@pytest.mark.parametrize(
    "rho, kraus_operators",
    [
        (rho_1, kraus_operators_1), (rho_2, kraus_operators_1), (rho_3, kraus_operators_2),
        (rho_4, kraus_operators_2), (rho_5, kraus_operators_3), (rho_1, kraus_operators_3)
    ],
)
def test_kraus_to_channel_on_quantumStates(rho, kraus_operators):
    """Test kraus_to_channel works as expected for valid inputs."""
    # Generate the quantum channel using your function
    quantum_channel = kraus_to_channel(kraus_operators)

    # Apply the quantum channel using Toqito's apply_channel function
    rho_after_channel = apply_channel(rho, kraus_operators)

    # Apply the superoperator to the vectorized form of rho
    rho_vec = rho.flatten("F")  # Column-major order
    rho_after_super_op = quantum_channel @ rho_vec
    rho_after_super_op = rho_after_super_op.reshape(2, 2, order="F")  # Reshape back

    # Compare both methods
    print("Using apply_channel:\n", rho_after_channel)
    print("Using superoperator:\n", rho_after_super_op)
    print("Difference:\n", rho_after_channel - rho_after_super_op)

    # The difference should be close to zero
    assert np.allclose(rho_after_channel, rho_after_super_op), "Mismatch in quantum channel application!"
