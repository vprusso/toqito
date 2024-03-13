"""Test gisin."""

import numpy as np

from toqito.states import gisin


def test_gisin_valid():
    """Standard Gisin state."""
    expected_res = np.array(
        [
            [1 / 4, 0, 0, 0],
            [0, 0.35403671, -0.22732436, 0],
            [0, -0.22732436, 0.14596329, 0],
            [0, 0, 0, 1 / 4],
        ]
    )

    res = gisin(0.5, 1)
    np.testing.assert_allclose(res, expected_res)


def test_gisin_invalid():
    """Invalid Gisin state parameters."""
    with np.testing.assert_raises(ValueError):
        gisin(5, 1)
