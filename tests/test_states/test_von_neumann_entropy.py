"""Tests for von_neumann_entropy function."""
import unittest
import numpy as np

from toqito.states.states.bell import bell
from toqito.states.states.max_mixed import max_mixed
from toqito.states.distance.von_neumann_entropy import von_neumann_entropy


class TestVonNeumannEntropy(unittest.TestCase):
    """Unit test for von_neumann_entropy."""

    def test_von_neumann_entropy_bell_state(self):
        """Entangled state von Neumann entropy should be zero."""
        rho = bell(0) * bell(0).conj().T
        res = von_neumann_entropy(rho)
        self.assertEqual(np.isclose(res, 0), True)

    def test_von_neumann_entropy_max_mixed_statre(self):
        """Von Neumann entropy of the maximally mixed state should be one."""
        res = von_neumann_entropy(max_mixed(2, is_sparse=False))
        self.assertEqual(np.isclose(res, 1), True)


if __name__ == "__main__":
    unittest.main()
