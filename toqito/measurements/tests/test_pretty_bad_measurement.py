"""Test pretty_bad_measurement."""

import numpy as np
import pytest

from toqito.measurements import pretty_bad_measurement
from toqito.states import bell, trine


@pytest.mark.parametrize(
    "states, probs, expected_result",
    [
        # Trine states (with probabilities).
        (
            trine(),
            [1 / 3, 1 / 3, 1 / 3],
            [
                np.array([[1 / 6, 0], [0, 1 / 2]]),
                np.array([[5 / 12, -1 / (4 * np.sqrt(3))], [-1 / (4 * np.sqrt(3)), 1 / 4]]),
                np.array([[5 / 12, 1 / (4 * np.sqrt(3))], [1 / (4 * np.sqrt(3)), 1 / 4]]),
            ],
        ),
        # Trine states (without probabilities).
        (
            trine(),
            None,
            [
                np.array([[1 / 6, 0], [0, 1 / 2]]),
                np.array([[5 / 12, -1 / (4 * np.sqrt(3))], [-1 / (4 * np.sqrt(3)), 1 / 4]]),
                np.array([[5 / 12, 1 / (4 * np.sqrt(3))], [1 / (4 * np.sqrt(3)), 1 / 4]]),
            ],
        ),
        # Bell states.
        (
            [bell(0), bell(1), bell(2), bell(3)],
            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
            [
                1 / 6 * np.array([[1, 0, 0, -1], [0, 2, 0, 0], [0, 0, 2, 0], [-1, 0, 0, 1]]),
                1 / 6 * np.array([[1, 0, 0, 1], [0, 2, 0, 0], [0, 0, 2, 0], [1, 0, 0, 1]]),
                1 / 6 * np.array([[2, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0, 2]]),
                1 / 6 * np.array([[2, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 2]]),
            ],
        ),
    ],
)
def test_pretty_bad_measurement(states, probs, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_allclose(pretty_bad_measurement(states, probs), expected_result)


@pytest.mark.parametrize(
    "states, probs",
    [
        # Invalid vector of probabilities.
        (trine(), [1 / 3, 1 / 2, 1 / 2]),
    ],
)
def test_pbm_invalid_probs(states, probs):
    """Ensures that probability vectors must sum to 1."""
    with np.testing.assert_raises(ValueError):
        pretty_bad_measurement(states, probs)


@pytest.mark.parametrize(
    "states, probs",
    [
        # Unequal set of states and probabilities.
        (trine(), [1 / 4, 1 / 4, 1 / 4, 1 / 4]),
    ],
)
def test_pbm_invalid_states(states, probs):
    """Ensures that number of states and number of probabilities are equal."""
    with np.testing.assert_raises(ValueError):
        pretty_bad_measurement(states, probs)
