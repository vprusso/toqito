"""Tests for spark of matrix."""

import numpy as np
import pytest

from toqito.matrix_props import spark


@pytest.mark.parametrize("matrix, expected_result", [
    # Spark is 3 since (no set of 1 or 2 columns that are linearly dependent)
    # But there is a set of 3 columns that are linearly dependent.
    (
        np.array([
            [1, 2, 0, 1],
            [1, 2, 0, 2],
            [1, 2, 0, 3],
            [1, 0, -3, 4],
        ]),
        3
    ),
    # Spark of matrix is 1 if there is a column of all zeros.
    (
        np.array([
            [1, 2, 0, 0],
            [1, 2, 0, 0],
            [1, 2, 0, 0],
            [1, 0, -3, 0],
        ]),
        1
    )
])
def test_spark(matrix, expected_result):
    """Tests for computing spark of matrix."""
    np.testing.assert_allclose(spark(matrix), expected_result)


def test_spark_full_rank():
    """All columns are linearly independent."""
    A = np.eye(4)
    assert spark(A) == 5


def test_spark_zero_matrix():
    """Any column is linearly dependent."""
    A = np.zeros((3, 3))
    assert spark(A) == 1

def test_spark_single_column():
    """Single column is always linearly independent."""
    A = np.array([[1], [2], [3]])
    assert spark(A) == 2
