"""Test is_product_vector."""
import numpy as np

from toqito.state_props import is_product_vector
from toqito.states import basis
from toqito.states import max_entangled


def test_is_product_entangled_state():
    """Check that is_product_vector returns False for an entangled state."""
    ent_vec = max_entangled(3)
    np.testing.assert_equal(is_product_vector(ent_vec), False)


def test_is_product_entangled_state_2_sys():
    """Check that dimension argument as list is supported."""
    ent_vec = max_entangled(4)
    np.testing.assert_equal(is_product_vector(ent_vec, dim=[4, 4]), False)


def test_is_product_entangled_state_3_sys():
    """Check that dimension argument as list is supported."""
    ent_vec = max_entangled(4)
    np.testing.assert_equal(is_product_vector(ent_vec, dim=[2, 2, 2, 2]), False)


def test_is_product_separable_state():
    """Check that is_product_vector returns True for a separable state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    sep_vec = (
        1 / 2 * (np.kron(e_0, e_0) - np.kron(e_0, e_1) - np.kron(e_1, e_0) + np.kron(e_1, e_1))
    )
    np.testing.assert_equal(is_product_vector(sep_vec), True)


def test_is_product_pure_state_2_2_2():
    """Check to ensure that pure state living in C^2 x C^2 x C^2 is product."""
    pure_vec = 1 / np.sqrt(2) * np.array([1, 0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_equal(is_product_vector(pure_vec, [2, 2, 2]), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
