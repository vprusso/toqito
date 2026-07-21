"""Test is_product_vector."""

import numpy as np
import pytest

from toqito.state_props import is_product
from toqito.states import bell, max_entangled

e_0, e_1 = np.array([[1], [0]]), np.array([[0], [1]])


@pytest.mark.parametrize(
    "rho, dim, expected_result",
    [
        # Check that is_product_vector returns False for an entangled state.
        (max_entangled(3), None, False),
        # Check that dimension argument as list is supported.
        (max_entangled(4), [4, 4], False),
        # Check that dimension argument as list is supported.
        (max_entangled(4), [2, 2, 2, 2], False),
        # Check that is_product_vector returns True for a separable state.
        (
            1 / 2 * (np.kron(e_0, e_0) - np.kron(e_0, e_1) - np.kron(e_1, e_0) + np.kron(e_1, e_1)),
            None,
            True,
        ),
        # Check to ensure that pure state living in C^2 x C^2 x C^2 is product.
        (1 / np.sqrt(2) * np.array([1, 0, 0, 0, 1, 0, 0, 0]), [2, 2, 2], True),
        # Check to ensure that a separable density matrix is product.
        (np.identity(4), None, True),
        # Check to ensure that an entangled density matrix is not product.
        (bell(0) @ bell(0).conj().T, None, False),
        # Check to ensure that an entangled density matrix is not product (with dimension).
        (bell(0) @ bell(0).conj().T, [2, 2], False),
    ],
)
def test_is_product(rho, dim, expected_result):
    """Test function works as expected for a valid input."""
    ipv, _ = is_product(rho=rho, dim=dim)
    np.testing.assert_equal(ipv, expected_result)


def test_is_product_multipartite_entangled_subfactor():
    """A tripartite state that is product across (12|3) but entangled in (1|2) is not a product state."""
    bell_12 = np.array([1, 0, 0, 1]) / np.sqrt(2)
    state = np.kron(bell_12, np.array([1, 0]))
    ipv, _ = is_product(state, [2, 2, 2])
    assert not np.all(ipv)
