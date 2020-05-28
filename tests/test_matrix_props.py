"""Test matrix properties."""
import numpy as np

from toqito.matrix_props import is_commuting
from toqito.matrix_props import is_density
from toqito.matrix_props import is_diagonal
from toqito.matrix_props import is_hermitian
from toqito.matrix_props import is_normal
from toqito.matrix_props import is_pd
from toqito.matrix_props import is_projection
from toqito.matrix_props import is_psd
from toqito.matrix_props import is_square
from toqito.matrix_props import is_symmetric
from toqito.matrix_props import is_unitary

from toqito.random import random_density_matrix
from toqito.random import random_unitary


def test_is_commuting_false():
    """Test if non-commuting matrices return False."""
    mat_1 = np.array([[0, 1], [0, 0]])
    mat_2 = np.array([[1, 0], [0, 0]])
    np.testing.assert_equal(is_commuting(mat_1, mat_2), False)


def test_is_commuting_true():
    """Test commuting matrices return True."""
    mat_1 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 2]])
    mat_2 = np.array([[2, 4, 0], [3, 1, 0], [-1, -4, 1]])
    np.testing.assert_equal(is_commuting(mat_1, mat_2), True)


def test_is_density_real_entries():
    """Test if random density matrix with real entries is density matrix."""
    mat = random_density_matrix(2, True)
    np.testing.assert_equal(is_density(mat), True)


def test_is_density_complex_entries():
    """Test if density matrix with complex entries is density matrix."""
    mat = random_density_matrix(4)
    np.testing.assert_equal(is_density(mat), True)


def test_is_diagonal():
    """Test if matrix is diagonal."""
    mat = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    np.testing.assert_equal(is_diagonal(mat), True)


def test_is_non_diagonal():
    """Test non-diagonal matrix."""
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(is_diagonal(mat), False)


def test_non_square():
    """Test on a non-square matrix."""
    mat = np.array([[1, 0, 0], [0, 1, 0]])
    np.testing.assert_equal(is_diagonal(mat), False)


def test_is_hermitian():
    """Test if matrix is Hermitian."""
    mat = np.array([[2, 2 + 1j, 4], [2 - 1j, 3, 1j], [4, -1j, 1]])
    np.testing.assert_equal(is_hermitian(mat), True)


def test_is_non_hermitian():
    """Test non-Hermitian matrix."""
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(is_hermitian(mat), False)


def test_is_normal():
    """Test that normal matrix returns True."""
    mat = np.identity(4)
    np.testing.assert_equal(is_normal(mat), True)


def test_is_not_normal():
    """Test that non-normal matrix returns False."""
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(is_normal(mat), False)


def test_is_pd():
    """Check that positive definite matrix returns True."""
    mat = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    np.testing.assert_equal(is_pd(mat), True)


def test_is_not_pd():
    """Check that non-positive definite matrix returns False."""
    mat = np.array([[-1, -1], [-1, -1]])
    np.testing.assert_equal(is_pd(mat), False)


def test_is_not_pd2():
    """Check that non-square matrix returns False."""
    mat = np.array([[1, 2, 3], [2, 1, 4]])
    np.testing.assert_equal(is_pd(mat), False)


def test_is_projection():
    """Check that projection matrix returns True."""
    mat = np.array([[0, 1], [0, 1]])
    np.testing.assert_equal(is_projection(mat), True)


def test_is_projection_2():
    """Check that projection matrix returns True."""
    mat = np.array([[1, 0], [0, 1]])
    np.testing.assert_equal(is_projection(mat), True)


def test_is_not_pd_non_projection():
    """Check that non-projection matrix returns False."""
    mat = np.array([[-1, -1], [-1, -1]])
    np.testing.assert_equal(is_projection(mat), False)


def test_is_not_pd_non_projection_2():
    """Check that non-projection matrix returns False."""
    mat = np.array([[1, 2, 3], [2, 1, 4]])
    np.testing.assert_equal(is_projection(mat), False)


def test_is_psd():
    """Test that positive semidefinite matrix returns True."""
    mat = np.array([[1, -1], [-1, 1]])
    np.testing.assert_equal(is_psd(mat), True)


def test_is_not_psd():
    """Test that non-positive semidefinite matrix returns False."""
    mat = np.array([[-1, -1], [-1, -1]])
    np.testing.assert_equal(is_psd(mat), False)


def test_is_square():
    """Test that square matrix returns True."""
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_equal(is_square(mat), True)


def test_is_not_square():
    """Test that non-square matrix returns False."""
    mat = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_equal(is_square(mat), False)


def test_is_symmetric():
    """Test that symmetric matrix returns True."""
    mat = np.array([[1, 7, 3], [7, 4, -5], [3, -5, 6]])
    np.testing.assert_equal(is_symmetric(mat), True)


def test_is_not_symmetric():
    """Test that non-symmetric matrix returns False."""
    mat = np.array([[1, 2], [3, 4]])
    np.testing.assert_equal(is_symmetric(mat), False)


def test_is_unitary_random():
    """Test that random unitary matrix returns True."""
    mat = random_unitary(2)
    np.testing.assert_equal(is_unitary(mat), True)


def test_is_unitary_hardcoded():
    """Test that hardcoded unitary matrix returns True."""
    mat = np.array([[0, 1], [1, 0]])
    np.testing.assert_equal(is_unitary(mat), True)


def test_is_not_unitary():
    """Test that non-unitary matrix returns False."""
    mat = np.array([[1, 0], [1, 1]])
    np.testing.assert_equal(is_unitary(mat), False)


def test_is_not_unitary_matrix():
    """Test that non-unitary matrix returns False."""
    mat = np.array([[1, 0], [1, 1]])
    np.testing.assert_equal(is_unitary(mat), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
