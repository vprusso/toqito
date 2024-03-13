"""Test is_permutation."""

import numpy as np

from toqito.matrix_props import is_permutation


def test_is_simple_permutation():
    """Test if matrix is a simple permutation matrix."""
    mat = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    np.testing.assert_equal(is_permutation(mat), True)


def test_is_not_square_matrix():
    """Test non-square matrix and thus will fail."""
    mat = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_equal(is_permutation(mat), False)


def test_is_not_binary_value_matrix():
    """Test a non-permutation matrix with one element being 9 which is not in set (0,1)."""
    mat = np.array([[0, 0, 0, 9], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    np.testing.assert_equal(is_permutation(mat), False)


def test_is_matrix_with_nonunitary_row_sum():
    """Test a non-permutation matrix where the unitary row sum check will fail."""
    mat = np.array([[0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    np.testing.assert_equal(is_permutation(mat), False)


def test_is_matrix_with_nonunitary_column_sum():
    """Test a non-permutation matrix where the unitary column sum check will fail."""
    mat = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    np.testing.assert_equal(is_permutation(mat), False)


def test_is_matrix_with_fractional_values_and_unitary_sums():
    """Test a non-permutation matrix with fractional values that passes the unitary sum check."""
    mat = np.array([[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_equal(is_permutation(mat), False)


def test_is_matrix_with_negative_values_and_unitary_sums():
    """Test a non-permutation matrix with some negative values that passes the unitary sum check."""
    mat = np.array([[2, -1], [-1, 2]])
    np.testing.assert_equal(is_permutation(mat), False)
