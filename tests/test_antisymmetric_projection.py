"""Tests for antisymmetric_projection function."""
import unittest
import numpy as np

from toqito.perms.antisymmetric_projection import antisymmetric_projection


class TestAntisymmetricProjection(unittest.TestCase):
    """Unit test for antisymmetric_projection."""

    def test_antisymmetric_projection_2(self):
        """Generates the antisymmetric_projection where the dimension is 2."""
        res = antisymmetric_projection(2).todense()
        expected_res = np.array([[0, 0, 0, 0],
                                 [0, 0.5, -0.5, 0],
                                 [0, -0.5, 0.5, 0],
                                 [0, 0, 0, 0]])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_antisymmetric_projection_3_3_true(self):
        """
        Generates the antisymmetric_projection where the `dim` is 3, the `p`
        is 3, and `partial` is True.
        """
        res = antisymmetric_projection(3, 3, True).todense()
        self.assertEqual(np.isclose(res[5].item(), -0.40824829), True)


if __name__ == '__main__':
    unittest.main()
