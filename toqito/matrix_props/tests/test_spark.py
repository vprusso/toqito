"""Tests for spark function."""

import numpy as np
import pytest
from toqito.matrix_props import spark

def test_spark_square_matrix():
    """Test spark function with a square matrix."""
    A = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    assert spark(A) == 4  # All columns are linearly independent

def test_spark_non_square_matrix():
    """Test spark function with a non-square matrix."""
    A = np.array([[1, 0, 1, 2], [0, 1, 1, 3]])
    assert spark(A) == 3  # First three columns are linearly dependent

def test_spark_zero_column():
    """Test spark function with a matrix containing a zero column."""
    A = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 0]])
    assert spark(A) == 1

def test_spark_full_rank():
    """Test spark function with a full rank matrix."""
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert spark(A) == 4

def test_spark_linearly_dependent_subset():
    """Test spark function with a matrix having a linearly dependent subset of columns."""
    A = np.array([[1, 2, 3, 4], [0, 1, 1, 2], [1, 0, 1, 1]])
    assert spark(A) == 3  # Columns 2, 3, 4 are linearly dependent

def test_spark_large_matrix():
    """Test spark function with a larger matrix."""
    A = np.array([
        [1, 0, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 0, 1, 1, 0],
        [1, 1, 1, 0, 1]
    ])
    assert spark(A) == 5  # All five columns are needed for linear dependence

def test_spark_property_rank():
    """Test spark function property: spark(A) <= rank(A) + 1."""
    A = np.random.rand(3, 5)
    s = spark(A)
    r = np.linalg.matrix_rank(A)
    assert s <= r + 1

def test_spark_invalid_input():
    """Test spark function with invalid input (1D array)."""
    with pytest.raises(ValueError):
        spark(np.array([1, 2, 3]))

def test_spark_3d_array():
    """Test spark function with invalid input (3D array)."""
    with pytest.raises(ValueError):
        spark(np.random.rand(2, 2, 2))
