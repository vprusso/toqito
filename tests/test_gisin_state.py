"""Tests for gisin_state function."""
import itertools
import unittest
import numpy as np

from toqito.states.gisin_state import gisin_state


class TestGisinState(unittest.TestCase):
    """Unit test for gisin_state."""

    def test_valid_gisin(self):
        """Standard Gisin state."""
        expected_res = np.array([[1/4, 0, 0, 0],
                                 [0, 0.35403671, -0.22732436, 0],
                                 [0, -0.22732436, 0.14596329, 0],
                                 [0, 0, 0, 1/4]])

        res = gisin_state(0.5, 1)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_invalid_gisin(self):
        """Invalid Gisin state parameters."""
        with self.assertRaises(ValueError):
            gisin_state(5, 1)


if __name__ == '__main__':
    unittest.main()
