"""Tests for fidelity function."""
import unittest
import cvxpy
import numpy as np

from toqito.state.distance.fidelity import fidelity


class TestFidelity(unittest.TestCase):
    """Unit test for fidelity."""

    def test_fidelity_default(self):
        """Test fidelity default arguments."""
        rho = np.array([[1/2, 0, 0, 1/2],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1/2, 0, 0, 1/2]])
        sigma = rho

        res = fidelity(rho, sigma)
        self.assertEqual(np.isclose(res, 1), True)

    def test_fidelity_cvx(self):
        """Test fidelity for cvx objects."""
        rho = cvxpy.bmat([[1/2, 0, 0, 1/2],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [1/2, 0, 0, 1/2]])
        sigma = rho

        res = fidelity(rho, sigma)
        self.assertEqual(np.isclose(res, 1), True)

    def test_non_square(self):
        """Tests for invalid dim."""
        rho = np.array([[1/2, 0, 0, 1/2],
                        [0, 0, 0, 0],
                        [1/2, 0, 0, 1/2]])
        sigma = np.array([[1/2, 0, 0, 1/2],
                          [0, 0, 0, 0],
                          [1/2, 0, 0, 1/2]])
        with self.assertRaises(ValueError):
            fidelity(rho, sigma)

    def test_invalid_dim(self):
        """Tests for invalid dim."""
        rho = np.array([[1/2, 0, 0, 1/2],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1/2, 0, 0, 1/2]])
        sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        with self.assertRaises(ValueError):
            fidelity(rho, sigma)


if __name__ == '__main__':
    unittest.main()
