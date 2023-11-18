"""Test inner_product."""
import numpy as np

from toqito.matrix_ops import inner_product


def test_inner_product():
    """Test with two vectors, no complications."""

    v1, v2 = np.array([1, 2, 3]), np.array([4, 5, 6])
    expected_res = 32
    np.testing.assert_equal(inner_product(v1, v2), expected_res)


def test_inner_product_negative_input():
    """Test with two vectors, with negative input value."""

    v1, v2 = np.array([-1, 2, 3]), np.array([4, 5, 6])
    expected_res = 24
    np.testing.assert_equal(inner_product(v1, v2), expected_res)


def test_inner_product_negative_output():
    """Test with two vectors, with negative expected output."""

    v1, v2 = np.array([1, 2, -3]), np.array([4, 5, 6])
    expected_res = -4
    np.testing.assert_equal(inner_product(v1, v2), expected_res)


def test_inner_product_different_dimensions():
    """Test with two vectors of different dimensions."""

    v1, v2 = np.array([1, 2, 3]), np.array([4, 5, 6, 7])
    with np.testing.assert_raises(ValueError):
        inner_product(v1, v2)


def test_inner_product_different_dimensions_2():
    """Test with a vector and a 2d array."""

    v1, v2 = np.array([1, 2, 3]), np.array([[4, 5, 6], [7, 8, 9]])
    with np.testing.assert_raises(ValueError):
        inner_product(v1, v2)


if __name__ == "__main__":
    np.testing.run_module_suite()
