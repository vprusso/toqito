"""Test vec."""
import numpy as np
import pytest

from toqito.matrix_ops import vec


@pytest.mark.parametrize("vector, expected_result", [
    # Test standard vec operation on a vector.
    (np.array(np.array([[1, 2], [3, 4]])), np.array([[1], [3], [2], [4]])),
])
def test_vec(vector, expected_result):
    np.testing.assert_array_equal(vec(vector), expected_result)
