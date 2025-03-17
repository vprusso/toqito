"""Unit tests for the common_quantum_overlap function."""

import numpy as np
import pytest

from toqito.state_props import common_quantum_overlap
from toqito.states import bell


@pytest.mark.parametrize(
    "states, expected_overlap, description",
    [
        (
            # Bell states are perfectly antidistinguishable (ω_Q = 0) as per arXiv:2401.17980v2.
            [bell(0), bell(1), bell(2), bell(3)],
            0,
            "Bell states are perfectly antidistinguishable (ω_Q = 0) as per arXiv:2401.17980v2.",
        ),
        (
            # For n maximally mixed preparations (here n=3) the overlap is maximum (ω_Q = 1).
            [np.eye(2) / 2, np.eye(2) / 2, np.eye(2) / 2],
            1,
            "Maximally mixed states have maximum overlap (ω_Q = 1) as per Corollary 1 in arXiv:2401.17980v2.",
        ),
    ],
)
def test_common_quantum_overlap_parametrized(states, expected_overlap, description):
    """Test common quantum overlap for various sets of state preparations.

    References:

        - Bell states case: arXiv:2401.17980v2.
        - Maximally mixed states: Corollary 1 in arXiv:2401.17980v2.
    """
    overlap = common_quantum_overlap(states)
    assert np.isclose(overlap, expected_overlap), description


def test_two_states_known_overlap():
    """Test with two pure states having a known inner product.

    For two pure states with inner product p, the quantum overlap is given by:
        ω_Q = 1 - √(1 - |p|²)
    """
    theta = np.pi / 4  # Define an angle so that inner_product = cos(theta)
    inner_product = np.cos(theta)

    psi_0 = np.array([1, 0])
    psi_1 = np.array([np.cos(theta), np.sin(theta)])

    states = [psi_0, psi_1]
    overlap = common_quantum_overlap(states)

    expected_value = 1 - np.sqrt(1 - inner_product**2)
    assert np.isclose(overlap, expected_value)


def test_three_states_on_great_circle():
    """Test for three pure states lying on a great circle of the Bloch sphere.

    Equally spaced states on a great circle are perfectly antidistinguishable.
    """
    theta = 2 * np.pi / 3  # Equal angular separation on the great circle
    psi_0 = np.array([1, 0])
    psi_1 = np.array([np.cos(theta), np.sin(theta)])
    psi_2 = np.array([np.cos(2 * theta), np.sin(2 * theta)])

    states = [psi_0, psi_1, psi_2]
    overlap = common_quantum_overlap(states)

    assert np.isclose(overlap, 0)

