"""Tests for fourier_matrix function."""
import unittest
import numpy as np

from toqito.matrix.matrices.fourier_matrix import fourier_matrix


class TestFourierMatrix(unittest.TestCase):
    """Unit test for fourier_matrix."""

    def test_fourier_matrix_dim_2(self):
        """Fourier matrix of dimension 2."""
        expected_res = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                                 [1/np.sqrt(2), -1/np.sqrt(2)]])

        res = fourier_matrix(2)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == '__main__':
    unittest.main()
