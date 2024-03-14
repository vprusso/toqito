"""Test pusey_barrett_rudolph."""

import numpy as np
import pytest

from toqito.states import pusey_barrett_rudolph


@pytest.mark.parametrize(
    "n, theta, expected_value",
    [
        # When `theta = 0`, this should simply be the [1, 0 ..., 0] vectors.
        (1, 0, [np.array([1, 0]).reshape(-1, 1), np.array([1, 0]).reshape(-1, 1)]),
        # When `theta = 0`, this should simply be the [1, 0 ..., 0] vectors.
        (
            2,
            0,
            [
                np.array([1, 0, 0, 0]).reshape(-1, 1),
                np.array([1, 0, 0, 0]).reshape(-1, 1),
                np.array([1, 0, 0, 0]).reshape(-1, 1),
                np.array([1, 0, 0, 0]).reshape(-1, 1),
            ],
        ),
        # n=1 and theta=0.5
        (
            1,
            0.5,
            [
                np.array([np.cos(1 / 4), np.sin(1 / 4)]).reshape(-1, 1),
                np.array([np.cos(1 / 4), -np.sin(1 / 4)]).reshape(-1, 1),
            ],
        ),
    ],
)
def test_pusey_barrett_rudolph(n, theta, expected_value):
    """Test functions works as expected for valid inputs."""
    states = pusey_barrett_rudolph(n, theta)
    np.testing.assert_equal(states, expected_value)
