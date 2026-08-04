"""Test channel_gram_matrix."""

import numpy as np
import pytest

from toqito.channel_ops import channel_gram_matrix

PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])
PAULIS = [np.eye(2), PAULI_X, PAULI_Y, PAULI_Z]


def test_pauli_default_sigma_is_identity():
    """The Pauli unitaries are orthonormal under Tr(A^* B)/dim, so G(I/d) = I."""
    gram = channel_gram_matrix(PAULIS)
    assert np.allclose(gram, np.eye(4))


def test_matches_choi_state_gram_for_unitaries():
    """At sigma = I/d, G(sigma)_ij equals the Choi-state overlap Tr(U_j^* U_i)/d."""
    unitaries = [np.eye(2), PAULI_X, (PAULI_X + PAULI_Z) / np.sqrt(2)]
    gram = channel_gram_matrix(unitaries)
    expected = np.array([[np.trace(unitaries[j].conj().T @ unitaries[i]) / 2 for j in range(3)] for i in range(3)])
    assert np.allclose(gram, expected)


def test_hermitian_gram():
    """The weighted Gram matrix is Hermitian for a Hermitian sigma."""
    gram = channel_gram_matrix(PAULIS)
    assert np.allclose(gram, gram.conj().T)


def test_custom_sigma():
    """A custom density operator is respected."""
    sigma = np.array([[0.75, 0.0], [0.0, 0.25]])
    gram = channel_gram_matrix([np.eye(2), PAULI_Z], sigma=sigma)
    # G_00 = Tr(sigma) = 1; G_01 = Tr(Z sigma) = 0.75 - 0.25 = 0.5.
    assert np.isclose(gram[0, 0], 1.0)
    assert np.isclose(gram[0, 1], 0.5)


def test_vectorized_result_matches_reference_for_rectangular_isometries():
    """The vectorized contraction agrees with the entry-by-entry definition."""
    rng = np.random.default_rng(0)
    isometries = [np.linalg.qr(rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3)))[0] for _ in range(4)]
    sigma_factor = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    sigma = sigma_factor @ sigma_factor.conj().T
    sigma /= np.trace(sigma)
    expected = np.array(
        [[np.trace(right.conj().T @ left @ sigma) for right in isometries] for left in isometries],
        dtype=complex,
    )

    result = channel_gram_matrix(isometries, sigma)

    assert result.dtype == np.dtype(complex)
    assert np.allclose(result, expected)


def test_non_isometry_raises():
    """A non-isometry operator is rejected."""
    with pytest.raises(ValueError, match="must be an isometry"):
        channel_gram_matrix([np.eye(2), 2 * np.eye(2)])


def test_mismatched_dimensions_raise():
    """Isometries of differing shape are rejected."""
    with pytest.raises(ValueError, match="same input and output dimensions"):
        channel_gram_matrix([np.eye(2), np.eye(3)])


def test_wrong_sigma_shape_raises():
    """A sigma of the wrong shape is rejected."""
    with pytest.raises(ValueError, match="dim_in, dim_in"):
        channel_gram_matrix([np.eye(2), PAULI_X], sigma=np.eye(3))


def test_empty_raises():
    """At least one isometry is required."""
    with pytest.raises(ValueError, match="At least one isometry"):
        channel_gram_matrix([])
