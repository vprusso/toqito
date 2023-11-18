"""Test is_idempotent."""
import numpy as np

from toqito.matrix_props import is_idempotent


def test_is_idempotent_identity_2_by_2():
    """Test that 2x2 identity matrix returns True."""
    mat = np.eye(2)
    np.testing.assert_equal(is_idempotent(mat), True)


def test_is_idempotent_identity_3_by_3():
    """Test that 3x3 identity matrix returns True."""
    mat = np.eye(3)
    np.testing.assert_equal(is_idempotent(mat), True)


def test_is_idempotent_2_by_2():
    """Test that 2x2 idempotent returns True."""
    mat = np.array([[3, -6], [1, -2]])
    np.testing.assert_equal(is_idempotent(mat), True)


def test_is_idempotent_3_by_3():
    """Test that 3x3 idempotent returns True."""
    mat = np.array([[2, -2, -4], [-1, 3, 4], [1, -2, -3]])
    np.testing.assert_equal(is_idempotent(mat), True)


def test_is_non_idempotent():
    """Test non-idempotent matrix."""
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(is_idempotent(mat), False)


def test_is_idempotent_not_square():
    """Idempotent must be a square matrix."""
    mat = np.array([[-1, 1, 1], [1, 2, 3]])
    np.testing.assert_equal(is_idempotent(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
