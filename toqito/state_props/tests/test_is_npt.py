"""Test is_npt."""

import numpy as np
import pytest

from toqito.state_props import is_npt
from toqito.states import bell, horodecki


@pytest.mark.parametrize(
    "mat, sys, dim, tol, expected_result",
    [
        # Check that non-NPT matrix returns False with sys specified.
        (np.identity(9), 2, None, None, False),
        # Check that non-NPT matrix returns False with dim and sys specified.
        (np.identity(9), 2, np.round(np.sqrt(9)), None, False),
        # Check that non-NPT matrix returns False.
        (np.identity(9), 2, np.round(np.sqrt(9)), None, False),
        # Check that non-NPT matrix with tolerance returns False.
        (np.identity(9), 2, np.round(np.sqrt(9)), 1e-10, False),
        # Entangled state of dimension 2 will violate NPT criterion.
        (bell(2) @ bell(2).conj().T, 2, None, None, True),
        # Horodecki state is an example of an entangled NPT state.
        (horodecki(0.5, [3, 3]), 2, None, None, False),
    ],
)
def test_is_npt(mat, sys, dim, tol, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_equal(is_npt(mat=mat, sys=sys, dim=dim, tol=tol), expected_result)
