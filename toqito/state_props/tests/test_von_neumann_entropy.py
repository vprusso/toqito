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
    ],
)
def test_von_neumann_entropy(rho, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_allclose(von_neumann_entropy(rho), expected_result, atol=1e-5)


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
