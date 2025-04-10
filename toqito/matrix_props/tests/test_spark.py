"""Tests for spark function."""

import numpy as np
import pytest

from toqito.matrix_props import spark


@pytest.mark.parametrize(
    "matrix, expected_spark, description",
    [
        (np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]]), 4, "Square matrix with all columns linearly independent"),
        (np.array([[1, 0, 1, 2], [0, 1, 1, 3]]), 3, "Non-square matrix with first three columns linearly dependent"),
        (np.array([[1, 0, 0], [1, 1, 0], [1, 1, 0]]), 1, "Matrix with a zero column"),
        (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 4, "Full rank matrix"),
        (np.array([[1, 2, 3, 4], [0, 1, 1, 2], [1, 0, 1, 1]]), 3, "Matrix with linearly dependent subset of columns"),
        (
            np.array([[1, 0, 0, 1, 2], [0, 1, 0, 1, 1], [0, 0, 1, 1, 0], [1, 1, 1, 0, 1]]),
            5,
            "Larger matrix with all columns needed for linear dependence",
        ),
    ],
)
def test_spark(matrix, expected_spark, description):
    """Test spark function with various input matrices."""
    assert spark(matrix) == expected_spark, f"Failed for case: {description}"


def test_spark_property_rank():
    """Test spark function property: spark(A) <= rank(A) + 1."""
    A = np.random.rand(3, 5)
    s = spark(A)
    r = np.linalg.matrix_rank(A)
    assert s <= r + 1, "Spark should be <= rank + 1"


@pytest.mark.parametrize(
    "invalid_input",
    [
        np.array([1, 2, 3]),  # 1D array
        np.random.rand(2, 2, 2),  # 3D array
    ],
)
def test_spark_invalid_input(invalid_input):
    """Test spark function with invalid inputs."""
    with pytest.raises(ValueError):
        spark(invalid_input)
