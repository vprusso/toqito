"""Test is_mutually_orthogonal."""
import numpy as np

from toqito.states import bell
from toqito.state_props import is_mutually_orthogonal


def test_is_mutually_orthogonal_bell_states():
    """Return True for orthogonal Bell vectors."""
    states = [bell(0), bell(1), bell(2), bell(3)]
    np.testing.assert_equal(is_mutually_orthogonal(states), True)


def test_is_not_mutually_orthogonal():
    """Return False for non-orthogonal vectors."""
    states = [np.array([1, 0]), np.array([1, 1])]
    np.testing.assert_equal(is_mutually_orthogonal(states), False)


def test_is_mutually_orthogonal_basis_invalid_input_len():
    """Tests for invalid input len."""
    with np.testing.assert_raises(ValueError):
        vec_list = [np.array([1, 0])]
        is_mutually_orthogonal(vec_list)


if __name__ == "__main__":
    np.testing.run_module_suite()
