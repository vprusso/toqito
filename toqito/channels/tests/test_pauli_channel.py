"""Test pauli channel."""

import re

import numpy as np
import pytest
from scipy.linalg import eigvalsh

from toqito.channels import partial_trace, pauli_channel


@pytest.mark.parametrize(
    "in_p",
    [
        # Negative probability.
        ([-0.1, 1.1]),
        # Incorrect length.
        ([0.2, 0.2, 0.2, 0.2, 0.2]),
        # Out of range probabilities.
        ([1.1, 0.1]),
    ],
)
def test_pauli_channel_valid_probability(in_p):
    """Test probability vector validation with various invalid inputs."""
    with pytest.raises(
        ValueError,
        match=(
            re.escape("Probabilities must be non-negative and sum to 1.")
            + "|"
            + re.escape("The length of the probability vector must be 4^q for some integer q (number of qubits).")
        ),
    ):
        pauli_channel(prob=in_p)


@pytest.mark.parametrize(
    "num_qubits,expected_dim",
    [
        # Single qubit.
        (1, 4),
        # Two qubits.
        (2, 16),
        # Three qubits.
        (3, 64),
    ],
)
def test_pauli_channel_dimensions(num_qubits, expected_dim):
    """Test Pauli channel generation for different numbers of qubits."""
    p = np.ones(expected_dim) / expected_dim
    Phi = pauli_channel(prob=p)
    assert Phi.shape == (expected_dim, expected_dim), f"Incorrect matrix dimensions for {num_qubits} qubits"


@pytest.mark.parametrize("prob", [np.array([0.1, 0.2, 0.3, 0.4]), np.random.rand(16)])
def test_pauli_channel_kraus_operators(prob):
    """Test generation of Kraus operators for different probability inputs.

    This function normalizes the probability vector, generates Kraus operators
    using the `pauli_channel` function, and verifies their properties:

    - Ensures the number of Kraus operators matches the number of probabilities.
    - Checks that the Kraus operators satisfy the completeness condition.
    """
    prob = prob / np.sum(prob)
    _, kraus_operators = pauli_channel(prob=prob, return_kraus_ops=True)

    assert len(kraus_operators) == len(prob)

    total_ops = np.zeros_like(kraus_operators[0], dtype=np.complex128)
    for K in kraus_operators:
        total_ops += K.conj().T @ K

    np.testing.assert_almost_equal(total_ops, np.eye(total_ops.shape[0]), decimal=10)


@pytest.mark.parametrize(
    "input_mat, prob",
    [
        # Identity matrix.
        (np.eye(2), np.array([0.1, 0.2, 0.3, 0.4])),
        # Pure state.
        (np.array([[1, 0], [0, 0]], dtype=complex), np.array([0.25, 0.25, 0.25, 0.25])),
    ],
)
def test_pauli_channel_input_matrix_properties(input_mat, prob):
    """Test properties of the output matrix when an input matrix is provided.

    This function ensures that the Pauli channel transformation holds:

    - The output matrix remains Hermitian.
    - The output matrix is positive semidefinite.
    - The trace of the input matrix is preserved in the output.

    """
    prob = prob / np.sum(prob)
    _, output_mat = pauli_channel(prob=prob, input_mat=input_mat)

    assert np.allclose(output_mat, output_mat.conj().T), "Output matrix is not Hermitian."

    eig_v = eigvalsh(output_mat)
    assert np.all(eig_v >= -1e-10), "Output matrix is not positive semidefinite."

    in_trace = np.trace(input_mat)
    out_trace = np.trace(output_mat)
    assert np.isclose(out_trace, in_trace), "Trace is not preserved in the output matrix."

    Phi, output_mat, kraus_ops = pauli_channel(prob=prob, input_mat=input_mat, return_kraus_ops=True)

    assert isinstance(output_mat, np.ndarray)
    assert isinstance(kraus_ops, list)
    assert len(kraus_ops) == len(prob)


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_pauli_channel_choi_matrix_properties(num_qubits):
    """Test Choi matrix properties for different numbers of qubits.

    This function verifies the following properties of Choi Matrix generated by Pauli Channel:

    - The Choi matrix has the expected dimension `(4**q, 4**q)`.
    - It is positive semidefinite.
    - Its partial trace over the output system yields the identity matrix of the input Hilbert space.
    - Its total trace equals the input dimension `(2**q)`.

    """
    Phi, _ = pauli_channel(prob=num_qubits, return_kraus_ops=True)
    Phi = np.array(Phi)
    expected_choi_dim = 4**num_qubits
    assert Phi.shape == (expected_choi_dim, expected_choi_dim), (
        f"Expected shape {(expected_choi_dim, expected_choi_dim)}, got {Phi.shape}"
    )

    eigenvalues = eigvalsh(Phi)
    assert np.all(eigenvalues >= -1e-10), f"Phi is not positive semidefinite for {num_qubits} qubits"

    input_dim = 2**num_qubits
    dims = [input_dim, input_dim]
    pt_output = partial_trace(Phi, sys=1, dim=dims)
    identity_input = np.eye(input_dim)
    assert np.allclose(pt_output, identity_input), f"Partial trace does not equal identity for {num_qubits} qubits"

    total_trace = np.trace(Phi)
    assert np.isclose(total_trace, input_dim), (
        f"Total trace ({total_trace}) does not equal input dimension ({input_dim}) for {num_qubits} qubits"
    )


@pytest.mark.parametrize(
    "prob",
    [
        # Multiple 0 entries.
        np.array([0.0, 0.5, 0.0, 0.5]),
        # Single 0 entries.
        np.array([0.25, 0.0, 0.25, 0.5]),
    ],
)
def test_pauli_channel_zero_probability(prob):
    """Test Pauli Channel when some input probabilities are zero.

    This function ensures that when certain probabilities in `p` are zero,
    their corresponding Kraus operators are effectively zero.
    """
    _, kraus_ops = pauli_channel(prob=prob, return_kraus_ops=True)

    for i, k in enumerate(kraus_ops):
        if prob[i] == 0:
            np.testing.assert_almost_equal(k, np.zeros_like(k))
