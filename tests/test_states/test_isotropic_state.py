"""Tests for isotropic function."""
import unittest
import numpy as np

from toqito.states import isotropic


class TestIsotropicState(unittest.TestCase):

    """Unit test for isotropic."""

    def test_isotropic_qutrit(self):
        """Generate a qutrit isotropic state with `alpha` = 1/2."""
        res = isotropic(3, 1 / 2)

        self.assertEqual(np.isclose(res[0, 0], 2 / 9), True)
        self.assertEqual(np.isclose(res[4, 4], 2 / 9), True)
        self.assertEqual(np.isclose(res[8, 8], 2 / 9), True)


if __name__ == "__main__":
    unittest.main()
