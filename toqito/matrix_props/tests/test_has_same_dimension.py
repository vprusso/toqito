"""Tests for has_same_dimension."""

import pytest

from toqito.matrix_props import has_same_dimension


def test_same_dimension_vectors():
    """Test that a list of vectors with the same dimension passes the check.

    This test verifies that `has_same_dimension` returns `True` when provided with a list of vectors
    that all have the same dimension.
    """
    vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert has_same_dimension(vectors) is True


def test_same_dimension_matrices():
    """Test that a list of square matrices with the same dimension passes the check.

    This test checks that `has_same_dimension` correctly identifies a list of square matrices that
    all have the same dimension, ensuring the function returns `True`.
    """
    matrices = [[[1, 0], [0, 1]], [[2, 3], [4, 5]], [[6, 7], [8, 9]]]
    assert has_same_dimension(matrices) is True


def test_mixed_dimensions():
    """Test that a mixed list of vectors and matrices with different dimensions fails the check.

    This test ensures that `has_same_dimension` returns `False` when provided with a list containing
    both vectors and matrices that do not share the same dimension, indicating inconsistent dimensions.
    """
    mixed = [[1, 2, 3], [[1, 0], [0, 1]]]
    assert has_same_dimension(mixed) is False


def test_empty_list():
    """Test that an empty list raises a ValueError.

    This test verifies that `has_same_dimension` raises a ValueError when provided with an empty list,
    as there are no items to check for consistent dimensions.
    """
    with pytest.raises(ValueError):
        has_same_dimension([])
