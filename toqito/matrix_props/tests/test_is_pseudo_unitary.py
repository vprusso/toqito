"""Test is_pseudo_unitary."""

import numpy as np

from toqito.matrix_props import is_pseudo_unitary
from toqito.rand import random_unitary


def test_is_unitary_random():
    """Test that random unitary matrix with signature diag([1, 1]) return True."""
    mat = random_unitary(2)
    np.testing.assert_equal(is_pseudo_unitary(mat, p=2, q=0), True)


def test_is_pseudo_unitary_lorentz_boost():
    """Test that Lorentz boost matrix with signature diag([1, -1]) returns True."""
    theta = np.random.rand()
    mat = np.array([[np.cosh(theta), np.sinh(theta)], [np.sinh(theta), np.cosh(theta)]])
    np.testing.assert_equal(is_pseudo_unitary(mat, p=1, q=1), True)


def test_is_not_pseudo_unitary():
    """Test that non pseudo unitary matrix returns False."""
    mat = np.array([[1, 0], [1, -1]])
    np.testing.assert_equal(is_pseudo_unitary(mat, p=1, q=1), False)


def test_is_not_pseudo_unitary_incorrect_signature_dimensions():
    """Test that non-unitary matrix returns False."""
    mat = np.array([[1, 0], [1, 1]])
    np.testing.assert_equal(is_pseudo_unitary(mat, p=4, q=5), False)


def test_is_pseudo_unitary_not_square():
    """Input must be a square matrix."""
    mat = np.array([[-1, 1, 1], [1, 2, 3]])
    np.testing.assert_equal(is_pseudo_unitary(mat, p=1, q=1), False)


def test_is_pseudo_unitary_value_error():
    """Input must have p >= 0 and q >= 0."""
    mat = np.array([[1, 0], [0, 1]])
    np.testing.assert_raises_regex(ValueError, "p and q must be non-negative", is_pseudo_unitary, mat, p=-1, q=1)
    np.testing.assert_raises_regex(ValueError, "p and q must be non-negative", is_pseudo_unitary, mat, p=1, q=-1)
