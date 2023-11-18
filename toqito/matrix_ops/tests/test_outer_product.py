"""Test outer_product."""
import numpy as np

from toqito.matrix_ops import outer_product


def test_outer_product():
    """Test with two vectors, no complications."""

    v1, v2 = np.array([1, 2, 3]), np.array([4, 5, 6])
    expected_res = np.array([[4, 5, 6], [8, 10, 12], [12, 15, 18]])
    np.testing.assert_equal(outer_product(v1, v2), expected_res)


def test_outer_product_negative():
    """Test with two vectors, with negative input/output values."""

    v1, v2 = np.array([-1, 2, 3]), np.array([4, 5, 6])
    expected_res = np.array([[-4, -5, -6], [8, 10, 12], [12, 15, 18]])
    np.testing.assert_equal(outer_product(v1, v2), expected_res)


def test_outer_product_different_dimensions():
    """Test with two vectors of different dimensions."""

    v1, v2 = np.array([1, 2, 3]), np.array([4, 5, 6, 7])
    with np.testing.assert_raises(ValueError):
        outer_product(v1, v2)


def test_outer_product_different_dimensions_2():
    """Test with a vector and a 2d array."""

    v1, v2 = np.array([1, 2, 3]), np.array([[4, 5, 6], [7, 8, 9]])
    with np.testing.assert_raises(ValueError):
        outer_product(v1, v2)
