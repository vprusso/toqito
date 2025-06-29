"""Unit tests for the common_quantum_overlap function."""

import numpy as np
import pytest

from toqito.state_props import common_quantum_overlap
from toqito.states import bell


@pytest.mark.parametrize(
    "states, expected_overlap",
    [
        (
            # Bell states: They are perfectly antidistinguishable (ω_Q = 0)
            [bell(0), bell(1), bell(2), bell(3)],
            0,
        ),
        (
            # Maximally mixed states: For n = 3, the overlap is maximum (ω_Q = 1)
            [np.eye(2) / 2, np.eye(2) / 2, np.eye(2) / 2],
            1,
        ),
        (
            # Two pure states with inner product cos(theta) (theta = π/4)
            # Expected overlap: ω_Q = 1 - √(1 - cos(π/4)²)
            [np.array([1, 0]), np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])],
            1 - np.sqrt(1 - np.cos(np.pi / 4) ** 2),
        ),
        (
            # Three pure states on a great circle of the Bloch sphere
            # Equally spaced states (angles: 0, 2π/3, 4π/3) are perfectly antidistinguishable (ω_Q = 0)
            [
                np.array([1, 0]),
                np.array([np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)]),
                np.array([np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3)]),
            ],
            0,
        ),
    ],
)
def test_common_quantum_overlap_parametrized(states, expected_overlap):
    """Test function works as expected for a valid input."""
    overlap = common_quantum_overlap(states)
    assert np.isclose(overlap, expected_overlap)
