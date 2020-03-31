"""Tests for gisin_state function."""
import unittest
import numpy as np

from toqito.state.states.gisin import gisin


class TestGisin(unittest.TestCase):
    """Unit test for gisin_state."""

    def test_valid_gisin(self):
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
        self.assertEqual(np.all(bool_mat), True)

    def test_invalid_gisin(self):
        """Invalid Gisin state parameters."""
        with self.assertRaises(ValueError):
            gisin(5, 1)


if __name__ == "__main__":
    unittest.main()
