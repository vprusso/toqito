"""Test partial_transpose."""
import cvxpy
import numpy as np

from cvxpy.atoms.affine.vstack import Vstack

from toqito.channels import partial_transpose
from toqito.states import bell


def test_partial_transpose():
    """
    Default partial_transpose.

    By default, the partial_transpose function performs the transposition
    on the second subsystem.
    """
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[1, 5, 3, 7], [2, 6, 4, 8], [9, 13, 11, 15], [10, 14, 12, 16]])

    res = partial_transpose(test_input_mat)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_sys():
    """
    Default partial transpose `sys` argument.

    By specifying the `sys` argument, you can perform the transposition on
    the first subsystem instead:
    """
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[1, 2, 9, 10], [5, 6, 13, 14], [3, 4, 11, 12], [7, 8, 15, 16]])

    res = partial_transpose(test_input_mat, 1)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_sys_vec():
    """Partial transpose on matrix with `sys` defined as vector."""
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]])

    res = partial_transpose(test_input_mat, [1, 2])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_sys_vec_dim_vec():
    """Variables `sys` and `dim` defined as vector."""
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]])

    res = partial_transpose(test_input_mat, [1, 2], [2, 2])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_norm_diff():
    """
    Apply partial transpose to first and second subsystem.

    Applying the transpose to both the first and second subsystems results
    in the standard transpose of the matrix.
    """
    test_input_mat = np.arange(1, 17).reshape(4, 4)
    res = np.linalg.norm(partial_transpose(test_input_mat, [1, 2]) - test_input_mat.conj().T)
    expected_res = 0

    np.testing.assert_equal(np.isclose(res, expected_res), True)


def test_partial_transpose_16_by_16():
    """Partial transpose on a 16-by-16 matrix."""
    test_input_mat = np.arange(1, 257).reshape(16, 16)
    res = partial_transpose(test_input_mat, [1, 3], [2, 2, 2, 2])
    first_expected_row = np.array(
        [1, 2, 33, 34, 5, 6, 37, 38, 129, 130, 161, 162, 133, 134, 165, 166]
    )

    first_expected_col = np.array([1, 17, 3, 19, 65, 81, 67, 83, 9, 25, 11, 27, 73, 89, 75, 91])

    np.testing.assert_equal(np.allclose(res[0, :], first_expected_row), True)
    np.testing.assert_equal(np.allclose(res[:, 0], first_expected_col), True)


def test_partial_transpose_bell_state():
    """Test partial transpose on a Bell state."""
    rho = bell(2) * bell(2).conj().T
    expected_res = np.array(
        [[0, 0, 0, 1 / 2], [0, 1 / 2, 0, 0], [0, 0, 1 / 2, 0], [1 / 2, 0, 0, 0]]
    )
    res = partial_transpose(rho)
    np.testing.assert_equal(np.allclose(res, expected_res), True)


def test_partial_transpose_non_square_matrix():
    """Matrix must be square."""
    with np.testing.assert_raises(ValueError):
        test_input_mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [13, 14, 15, 16]])
        partial_transpose(test_input_mat)


def test_partial_transpose_non_square_matrix_2():
    """Matrix must be square."""
    with np.testing.assert_raises(ValueError):
        rho = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        partial_transpose(rho, 2, [2])


def test_partial_transpose_cvxpy():
    """Test partial transpose on cvxpy objects."""
    x_var = cvxpy.Variable((4, 4), hermitian=True)
    x_pt = partial_transpose(x_var)
    np.testing.assert_equal(isinstance(x_pt, Vstack), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
