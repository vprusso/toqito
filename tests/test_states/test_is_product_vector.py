"""Tests for is_product_vector function."""
import unittest
import numpy as np

from toqito.core.ket import ket
from toqito.states.states.max_entangled import max_entangled
from toqito.states.properties.is_product_vector import is_product_vector


class TestIsProductVector(unittest.TestCase):
    """Unit test for is_product_vector."""

    def test_is_product_entangled_state(self):
        """Check that is_product_vector returns False for an entangled state."""
        ent_vec = max_entangled(3)
        self.assertEqual(is_product_vector(ent_vec), False)

    def test_is_product_entangled_state_2_sys(self):
        """Check that dimension argument as list is supported."""
        ent_vec = max_entangled(4)
        self.assertEqual(is_product_vector(ent_vec, dim=[4, 4]), False)

    def test_is_product_entangled_state_3_sys(self):
        """Check that dimension argument as list is supported."""
        ent_vec = max_entangled(4)
        self.assertEqual(is_product_vector(ent_vec, dim=[2, 2, 2, 2]), False)

    def test_is_product_separable_state(self):
        """Check that is_product_vector returns True for a separable state."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        sep_vec = (
            1
            / 2
            * (
                np.kron(e_0, e_0)
                - np.kron(e_0, e_1)
                - np.kron(e_1, e_0)
                + np.kron(e_1, e_1)
            )
        )
        self.assertEqual(is_product_vector(sep_vec), True)


if __name__ == "__main__":
    unittest.main()
