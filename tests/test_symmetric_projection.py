"""Tests for symmetric_projection function."""
import unittest
import numpy as np

from toqito.perms.symmetric_projection import symmetric_projection


class TestSymmetricProjection(unittest.TestCase):
    """Unit test for symmetric_projection."""

    def test_symmetric_projection_2(self):
        """Generates the symmetric_projection where the dimension is 2."""
        res = symmetric_projection(2).todense()
        expected_res = np.array([[1, 0, 0, 0],
                                 [0, 0.5, 0.5, 0],
                                 [0, 0.5, 0.5, 0],
                                 [0, 0, 0, 1]])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == '__main__':
    unittest.main()
