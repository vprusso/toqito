"""Test calculate_vector_matrix_dimension."""

import numpy as np
import pytest

from toqito.matrix_ops import calculate_vector_matrix_dimension


def test_1d_vector_dimension():
    """Verify dimension calculation for 1D vectors."""
    vector = np.array([1, 2, 3, 4])
    assert calculate_vector_matrix_dimension(vector) == 4


def test_2d_column_vector_dimension():
    """Verify dimension calculation for 2D column vectors."""
    vector_2d_col = np.array([[1], [2], [3], [4]])
    assert calculate_vector_matrix_dimension(vector_2d_col) == 4


def test_2d_row_vector_dimension():
    """Verify dimension calculation for 2D row vectors."""
    vector_2d_row = np.array([[1, 2, 3, 4]])
    assert calculate_vector_matrix_dimension(vector_2d_row) == 4


def test_square_matrix_dimension():
    """Verify dimension calculation for square matrices."""
    matrix = np.array([[1, 0], [0, 1]])
    assert calculate_vector_matrix_dimension(matrix) == 2


def test_non_square_matrix_error():
    """Verify error handling for non-square matrices."""
    non_square_matrix = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        calculate_vector_matrix_dimension(non_square_matrix)


def test_invalid_input_error():
    """Verify error handling for invalid inputs."""
    invalid_input = "not an array"
    with pytest.raises(ValueError):
        calculate_vector_matrix_dimension(invalid_input)


def test_higher_dimensional_array_error():
    """Verify error handling for higher-dimensional arrays."""
    higher_dim_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    with pytest.raises(ValueError):
        calculate_vector_matrix_dimension(higher_dim_array)


def test_numpy_array_with_non_numeric_types():
    """Verify dimension calculation for numpy arrays with non-numeric types."""
    non_numeric_array = np.array(["a", "b", "c", "d"])
    assert calculate_vector_matrix_dimension(non_numeric_array) == 4


def test_single_element_array_dimension():
    """Verify dimension calculation for 1D arrays with a single element."""
    single_element_array = np.array([1])
    assert calculate_vector_matrix_dimension(single_element_array) == 1
