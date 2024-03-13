"""Test is_ppt."""

import numpy as np
import pytest

from toqito.state_props import is_ppt
from toqito.states import bell, horodecki


@pytest.mark.parametrize(
    "mat, sys, dim, tol, expected_result",
    [
        # Check that PPT matrix returns True.
        (np.identity(9), 2, None, None, True),
        # Check that PPT matrix returns True with dim and sys specified.
        (np.identity(9), 2, [3], None, True),
        # Check that PPT matrix returns True.
        (np.identity(9), 2, [3], 1e-10, True),
        # Entangled state of dimension 2 will violate PPT criterion.
        (bell(2) @ bell(2).conj().T, 2, None, None, False),
        # Horodecki state is an example of an entangled PPT state.
        (horodecki(0.5, [3, 3]), 2, None, None, True),
    ],
)
def test_is_ppt(mat, sys, dim, tol, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_equal(is_ppt(mat=mat, sys=sys, dim=dim, tol=tol), expected_result)
