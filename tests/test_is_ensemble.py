"""Tests for is_ensemble function."""
import unittest
import numpy as np

from toqito.states.properties.is_ensemble import is_ensemble


class TestIsEnsemble(unittest.TestCase):
    """Unit test for is_ensemble."""

    def test_is_ensemble_true(self):
        """Test if valid ensemble returns True."""
        rho_0 = np.array([[0.5, 0], [0, 0]])
        rho_1 = np.array([[0, 0], [0, 0.5]])
        states = [rho_0, rho_1]
        self.assertEqual(is_ensemble(states), True)

    def test_is_non_ensemble_non_psd(self):
        """Test if non-valid ensemble returns False."""
        rho_0 = np.array([[0.5, 0], [0, 0]])
        rho_1 = np.array([[-1, -1], [-1, -1]])
        states = [rho_0, rho_1]
        self.assertEqual(is_ensemble(states), False)


if __name__ == "__main__":
    unittest.main()
