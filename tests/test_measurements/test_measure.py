"""Tests for measure function."""
import unittest
import numpy as np

from toqito.states import basis
from toqito.measurement_ops import measure


class TestMeasure(unittest.TestCase):
    """Unit test for measure."""

    def test_measure_state(self):
        """Test measure on quantum state."""
        e_0, e_1 = basis(2, 0), basis(2, 1)
        psi = 1 / np.sqrt(3) * e_0 + np.sqrt(2 / 3) * e_1
        rho = psi * psi.conj().T

        proj_0 = e_0 * e_0.conj().T
        proj_1 = e_1 * e_1.conj().T
        self.assertEqual(np.isclose(measure(proj_0, rho), 1 / 3), True)
        self.assertEqual(np.isclose(measure(proj_1, rho), 2 / 3), True)


if __name__ == "__main__":
    unittest.main()
