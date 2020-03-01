"""Tests for isotropic_state function."""
import unittest
import numpy as np

from toqito.state.states.isotropic_state import isotropic_state


class TestIsotropicState(unittest.TestCase):
    """Unit test for isotropic_state."""

    def test_isotropic_qutrit(self):
        """Generate a qutrit isotropic state with `alpha` = 1/2."""
        res = isotropic_state(3, 1/2)

        self.assertEqual(np.isclose(res[0, 0], 2/9), True)
        self.assertEqual(np.isclose(res[4, 4], 2/9), True)
        self.assertEqual(np.isclose(res[8, 8], 2/9), True)


if __name__ == '__main__':
    unittest.main()
