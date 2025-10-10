"""Tests for renyi_entropy."""

import numpy as np
import pytest

from toqito.state_props import purity, renyi_entropy, von_neumann_entropy
from toqito.states import bell, max_mixed

RHO_TEST = np.array([[0.8, 0.0], [0.0, 0.2]])


@pytest.mark.parametrize(
    "rho, alpha, expected_result",
    [
        # Entangled state Rényi entropy should be zero.
        (bell(0) @ bell(0).conj().T, 3 / 2, 0),
        # Rényi entropy of the maximally mixed state should be one.
        (max_mixed(2, is_sparse=False), 5 / 2, 1),
        # For alpha=0, log of the number of outcomes
        (RHO_TEST, 0.0, 1.0),
        # For alpha=1, recovers von Neumann entropy
        (RHO_TEST, 1.0, von_neumann_entropy(RHO_TEST)),
        # For alpha=2, collision entropy
        (RHO_TEST, 2.0, -np.log2(purity(RHO_TEST))),
        # For alpha=+inf, min-entropy
        (RHO_TEST, float("inf"), -np.log2(0.8)),
    ],
)
def test_renyi_entropy(rho, alpha, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_allclose(renyi_entropy(rho, alpha), expected_result, atol=1e-5)


@pytest.mark.parametrize(
    "rho, alpha",
    [
        # Test Rényi entropy on non-density matrix.
        (np.array([[1, 2], [3, 4]]), 3 / 2),
        # Test Rényi entropy on non-positive order.
        (RHO_TEST, -1.0),
    ],
)
def test_renyi_invalid_input(rho, alpha):
    """Test function works as expected for an invalid input."""
    with np.testing.assert_raises(ValueError):
        renyi_entropy(rho, alpha)
