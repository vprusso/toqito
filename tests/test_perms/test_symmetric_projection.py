"""Tests for symmetric_projection function."""
import unittest
import numpy as np

from toqito.perms import symmetric_projection


class TestSymmetricProjection(unittest.TestCase):

    """Unit test for symmetric_projection."""

    def test_symmetric_projection_dim_2_pval_1(self):
        """Symmetric_projection where the dimension is 2 and p_val is 1."""
        res = symmetric_projection(dim=2, p_val=1).todense()
        expected_res = np.array([[1, 0], [0, 1]])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_symmetric_projection_2(self):
        """Generates the symmetric_projection where the dimension is 2."""
        res = symmetric_projection(dim=2).todense()
        expected_res = np.array(
            [[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 1]]
        )

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_symmetric_projection_dim_2_partial_true(self):
        """Symmetric_projection where the dimension is 2 and partial is True."""
        res = symmetric_projection(dim=2, p_val=2, partial=True).todense()
        expected_res = np.array(
            [[0, 0, 1], [-1 / np.sqrt(2), 0, 0], [-1 / np.sqrt(2), 0, 0], [0, 1, 0]]
        )

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == "__main__":
    unittest.main()
