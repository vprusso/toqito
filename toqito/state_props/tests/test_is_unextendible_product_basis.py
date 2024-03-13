"""Test is_unextendible_product_basis."""

import numpy as np
import pytest

from toqito.matrix_ops import tensor
from toqito.state_props import is_unextendible_product_basis
from toqito.states import basis, bell, tile

e_0, e_1 = basis(2, 0), basis(2, 1)
e_p, e_m = (e_0 + e_1) / np.sqrt(2), (e_0 - e_1) / np.sqrt(2)


@pytest.mark.parametrize(
    "states, dims",
    [
        # Check if exception raised if non-product state is passed.
        ([bell(0), bell(1), bell(2), bell(3)], [2, 2]),
        # Check if exception raised if size of a vector does not match product of dims.
        ([bell(0), bell(1), bell(2), bell(3)], [2, 3]),
    ],
)
def test_unextendible_product_basis_invalid(states, dims):
    """Test that invalid input for unextendible product basis is handled."""
    with np.testing.assert_raises(ValueError):
        is_unextendible_product_basis(states, dims)


@pytest.mark.parametrize(
    "states, dims, expected_result",
    [
        # Check if correct answer returned when there are too few vectors.
        ([tensor([e_0, e_1, e_p]), tensor([e_m, e_m, e_m])], [2, 2, 2], False),
        # Check if Tiles[0, 1, 2, 3, 4] is correctly identified as UPB.
        ([tile(0), tile(1), tile(2), tile(3), tile(4)], [3, 3], True),
        # Check if Tiles[0, 1, 2, 3] is correctly identified as non-UPB.
        ([tile(0), tile(1), tile(2), tile(3)], [3, 3], False),
        # Check if Shifts is correctly identified as UPB.
        (
            [tensor([e_0, e_1, e_p]), tensor([e_1, e_p, e_0]), tensor([e_p, e_0, e_1]), tensor([e_m, e_m, e_m])],
            [2, 2, 2],
            True,
        ),
    ],
)
def test_unextendible_product_basis(states, dims, expected_result):
    """Test UPB works as expected for a valid input."""
    res = is_unextendible_product_basis(states, dims)
    np.testing.assert_equal(res[0], expected_result)
