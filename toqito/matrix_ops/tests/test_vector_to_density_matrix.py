"""Test vector_to_density_matrix."""
import numpy as np
import pytest

from toqito.matrix_ops import vector_to_density_matrix


def test_vector_to_density_matrix_with_1d_vector():
    """Test conversion of 1D vector to density matrix."""
    vector = np.array([1, 2, 3])
    expected_density_matrix = np.array([[1, 2, 3],
                                        [2, 4, 6],
                                        [3, 6, 9]])
    computed_density_matrix = vector_to_density_matrix(vector)
    np.testing.assert_array_equal(computed_density_matrix, expected_density_matrix)


def test_vector_to_density_matrix_with_column_vector():
    """Test conversion of column vector to density matrix."""
    column_vector = np.array([[1], [2], [3]])
    expected_density_matrix = np.array([[1, 2, 3],
                                        [2, 4, 6],
                                        [3, 6, 9]])
    computed_density_matrix = vector_to_density_matrix(column_vector)
    np.testing.assert_array_equal(computed_density_matrix, expected_density_matrix)


def test_vector_to_density_matrix_with_row_vector():
    """Test conversion of row vector to density matrix."""
    row_vector = np.array([[1, 2, 3]])
    expected_density_matrix = np.array([[1, 2, 3],
                                        [2, 4, 6],
                                        [3, 6, 9]])
    computed_density_matrix = vector_to_density_matrix(row_vector)
    np.testing.assert_array_equal(computed_density_matrix, expected_density_matrix)


def test_vector_to_density_matrix_with_square_matrix():
    """Test that square matrix is returned unchanged."""
    square_matrix = np.array([[1, 0], [0, 1]])
    computed_density_matrix = vector_to_density_matrix(square_matrix)
    np.testing.assert_array_equal(computed_density_matrix, square_matrix)


def test_vector_to_density_matrix_with_non_square_matrix():
    """Test that non-square matrix raises ValueError."""
    non_square_matrix = np.array([[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError):
        vector_to_density_matrix(non_square_matrix)


def test_vector_to_density_matrix_with_higher_dimensional_array():
    """Test that higher-dimensional array raises ValueError."""
    higher_dim_array = np.array([[[1, 0], [0, 1]]])
    with pytest.raises(ValueError):
        vector_to_density_matrix(higher_dim_array)
