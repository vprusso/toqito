"""Test pretty_good_measurement."""

import numpy as np
import pytest

from toqito.measurements import pretty_good_measurement
from toqito.states import bell, trine


@pytest.mark.parametrize(
    "states, probs, expected_result",
    [
        # Trine states (with probabilities).
        (
            trine(),
            [1 / 3, 1 / 3, 1 / 3],
            [
                np.array([[2 / 3, 0], [0, 0]]),
                np.array([[1 / 6, 1 / (2 * np.sqrt(3))], [1 / (2 * np.sqrt(3)), 1 / 2]]),
                np.array([[1 / 6, -1 / (2 * np.sqrt(3))], [-1 / (2 * np.sqrt(3)), 1 / 2]]),
            ],
        ),
        # Trine states (without probabilities).
        (
            trine(),
            None,
            [
                np.array([[2 / 3, 0], [0, 0]]),
                np.array([[1 / 6, 1 / (2 * np.sqrt(3))], [1 / (2 * np.sqrt(3)), 1 / 2]]),
                np.array([[1 / 6, -1 / (2 * np.sqrt(3))], [-1 / (2 * np.sqrt(3)), 1 / 2]]),
            ],
        ),
        # Bell states.
        (
            [bell(0), bell(1), bell(2), bell(3)],
            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
            [
                1 / 2 * np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]),
                1 / 2 * np.array([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]),
                1 / 2 * np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]),
                1 / 2 * np.array([[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]]),
            ],
        ),
    ],
)
def test_pretty_good_measurement(states, probs, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_allclose(pretty_good_measurement(states, probs), expected_result)


@pytest.mark.parametrize(
    "states, probs",
    [
        # Invalid vector of probabilities.
        (trine(), [1 / 3, 1 / 2, 1 / 2]),
    ],
)
def test_pgm_invalid_probs(states, probs):
    """Ensures that probability vectors must sum to 1."""
    with np.testing.assert_raises(ValueError):
        pretty_good_measurement(states, probs)


@pytest.mark.parametrize(
    "states, probs",
    [
        # Unequal set of states and probabilities.
        (trine(), [1 / 4, 1 / 4, 1 / 4, 1 / 4]),
    ],
)
def test_pgm_invalid_states(states, probs):
    """Ensures that number of states and number of probabilities are equal."""
    with np.testing.assert_raises(ValueError):
        pretty_good_measurement(states, probs)
