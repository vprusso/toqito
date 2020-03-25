"""Tests for max_mixed function."""
import unittest
import numpy as np

from toqito.state.states.max_mixed import max_mixed


class TestMaxMixed(unittest.TestCase):
    """Unit test for max_mixed."""

    def test_max_mixed_dim_2_full(self):
        """Generate full 2-dimensional maximally mixed state."""
        expected_res = 1 / 2 * np.array([[1, 0], [0, 1]])
        res = max_mixed(2, is_sparse=False)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_max_mixed_dim_2_sparse(self):
        """Generate sparse 2-dimensional maximally mixed state."""
        expected_res = 1 / 2 * np.array([[1, 0], [0, 1]])
        res = max_mixed(2, is_sparse=True)

        bool_mat = np.isclose(res.toarray(), expected_res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == "__main__":
    unittest.main()
