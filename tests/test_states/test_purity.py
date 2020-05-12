"""Tests for purity function."""
import unittest
import numpy as np
from toqito.state_metrics import purity


class TestPurity(unittest.TestCase):
    """Unit test for purity."""

    def test_purity(self):
        """Test for identity matrix."""
        expected_res = 1 / 4
        res = purity(np.identity(4) / 4)
        self.assertEqual(res, expected_res)

    def test_purity_non_density_matrix(self):
        r"""Test purity on non-density matrix."""
        rho = np.array([[1, 2], [3, 4]])

        with self.assertRaises(ValueError):
            purity(rho)


if __name__ == "__main__":
    unittest.main()
