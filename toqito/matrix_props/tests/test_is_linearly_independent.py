"""Test is_linearly_independent."""

import numpy as np
import pytest

from toqito.matrix_props import is_linearly_independent


@pytest.mark.parametrize(
    "vectors, expected_result",
    [
        ([np.array([[1], [0], [1]]), np.array([[1], [1], [0]]), np.array([[0], [0], [1]])], True),
        ([np.array([[0], [1], [2]]), np.array([[1], [2], [3]]), np.array([[3], [5], [7]])], False),
    ],
)
def test_is_linearly_independent(vectors, expected_result):
    """Test for linear independence/dependence of vectors."""
    np.testing.assert_equal(is_linearly_independent(vectors), expected_result)
