"""Test to_density_matrix."""

import numpy as np
import pytest

from toqito.matrix_ops import to_density_matrix


@pytest.mark.parametrize(
    "input_vector, expected_output, exception",
    [
        # Test conversion of 1D vector to density matrix
        (np.array([1, 2, 3]), np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]]), None),
        # Test conversion of column vector to density matrix
        (np.array([[1], [2], [3]]), np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]]), None),
        # Test conversion of row vector to density matrix
        (np.array([[1, 2, 3]]), np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]]), None),
        # Test that square matrix is returned unchanged
        (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), None),
        # Test that non-square matrix raises ValueError
        (np.array([[1, 0, 0], [0, 1, 0]]), None, ValueError),
        # Test that higher-dimensional array raises ValueError
        (np.array([[[1, 0], [0, 1]]]), None, ValueError),
    ],
)
def test_to_density_matrix(input_vector, expected_output, exception):
    """Test vector to density matrix functionality."""
    if exception:
        with pytest.raises(exception):
            to_density_matrix(input_vector)
    else:
        computed_density_matrix = to_density_matrix(input_vector)
        assert (np.abs(computed_density_matrix - expected_output) <= 1e-3).all()
