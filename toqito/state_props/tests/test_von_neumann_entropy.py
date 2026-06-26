"""Tests for von_neumann_entropy."""

import numpy as np
import pytest

from toqito.state_props import von_neumann_entropy
from toqito.states import bell, max_mixed


@pytest.mark.parametrize(
    "rho, expected_result",
    [
        # Entangled state von Neumann entropy should be zero.
        (bell(0) @ bell(0).conj().T, 0),
        # Von Neumann entropy of the maximally mixed state should be one.
        (max_mixed(2, is_sparse=False), 1),
        # Rank-deficient state with zero eigenvalues: entropy of diag(1/2, 1/2, 0, 0) is one.
        (np.diag([0.5, 0.5, 0.0, 0.0]), 1),
    ],
)
def test_von_neumann_entropy(rho, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_allclose(von_neumann_entropy(rho), expected_result, atol=1e-5)


def test_von_neumann_entropy_complex_hermitian():
    """Entropy is unitarily invariant; a complex-unitary-rotated state has entropy H(eigenvalues)."""
    probs = np.array([0.7, 0.3])
    # A complex 2x2 unitary, so rho is a genuinely non-diagonal Hermitian matrix.
    v_mat = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    rho = v_mat @ np.diag(probs) @ v_mat.conj().T

    expected = float(-np.sum(probs * np.log2(probs)))
    np.testing.assert_allclose(von_neumann_entropy(rho), expected, atol=1e-10)


@pytest.mark.parametrize(
    "rho",
    [
        # Test von Neumann entropy on non-density matrix.
        (np.array([[1, 2], [3, 4]])),
    ],
)
def test_von_neumann_invalid_input(rho):
    """Test function works as expected for an invalid input."""
    with np.testing.assert_raises(ValueError):
        von_neumann_entropy(rho)
