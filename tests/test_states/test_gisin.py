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

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gisin_invalid():
    """Invalid Gisin state parameters."""
    with np.testing.assert_raises(ValueError):
        gisin(5, 1)


if __name__ == "__main__":
    np.testing.run_module_suite()
