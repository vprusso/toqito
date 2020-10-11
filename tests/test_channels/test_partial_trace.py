"""Test partial_trace."""
import cvxpy
import numpy as np

from cvxpy.atoms.affine.vstack import Vstack
from toqito.channels import partial_trace


def test_partial_trace():
    """
    Standard call to partial_trace.

    By default, the partial_trace function takes the trace over the second
    subsystem.
    """
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[7, 11], [23, 27]])

    res = partial_trace(test_input_mat)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_sys():
    """
    Specify the `sys` argument.

    By specifying the `sys` argument, you can perform the partial trace
    the first subsystem instead:
    """
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[12, 14], [20, 22]])

    res = partial_trace(test_input_mat, 1)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_sys_int_dim_int():
    """
    Default second subsystem.

    By default, the partial_transpose function takes the trace over
    the second subsystem.
    """
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[7, 11], [23, 27]])

    res = partial_trace(test_input_mat, 2, 2)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_sys_int_dim_int_2():
    """
    Default second subsystem.

    By default, the partial_transpose function takes the trace over
    the second subsystem.
    """
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = 34

    res = partial_trace(test_input_mat, 2, 1)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_8_by_8():
    """Test for 8-by-8 matrix."""
    test_input_mat = np.arange(1, 65).reshape(8, 8)
    res = partial_trace(test_input_mat, [1, 2], [2, 2, 2])

    expected_res = np.array([[112, 116], [144, 148]])

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_8_by_8_2():
    """Test for 8-by-8 matrix."""
    test_input_mat = np.arange(1, 65).reshape(8, 8)
    res = partial_trace(test_input_mat, [1], [2, 2, 2])

    expected_res = np.array(
        [[38, 40, 42, 44], [54, 56, 58, 60], [70, 72, 74, 76], [86, 88, 90, 92]]
    )

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_8_by_8_3():
    """Test for 8-by-8 matrix."""
    test_input_mat = np.arange(1, 65).reshape(8, 8)
    res = partial_trace(test_input_mat, 3, [2, 2, 2])

    expected_res = np.array(
        [[11, 15, 19, 23], [43, 47, 51, 55], [75, 79, 83, 87], [107, 111, 115, 119]]
    )

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_16_by_16():
    """Test for 16-by-16 matrix."""
    test_input_mat = np.arange(1, 257).reshape(16, 16)
    res = partial_trace(test_input_mat, [1, 3], [2, 2, 2, 2])

    expected_res = np.array(
        [
            [344, 348, 360, 364],
            [408, 412, 424, 428],
            [600, 604, 616, 620],
            [664, 668, 680, 684],
        ]
    )

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_16_by_16_2():
    """Test for 16-by-16 matrix."""
    test_input_mat = np.arange(1, 257).reshape(16, 16)
    res = partial_trace(test_input_mat, [1, 2], [2, 2, 2, 2])

    expected_res = np.array(
        [
            [412, 416, 420, 424],
            [476, 480, 484, 488],
            [540, 544, 548, 552],
            [604, 608, 612, 616],
        ]
    )

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_64_by_64():
    """Test for 64-by-64 matrix."""
    test_input_mat = np.arange(1, 4097).reshape(64, 64)
    res = partial_trace(test_input_mat, [1, 2, 3], [4, 4, 4])

    expected_res = np.array([[131104]])

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_64_by_64_2():
    """Test for 64-by-64 matrix."""
    test_input_mat = np.arange(1, 4097).reshape(64, 64)
    res = partial_trace(test_input_mat, [1, 2], [4, 4, 4])

    expected_res = np.array(
        [
            [31216, 31232, 31248, 31264],
            [32240, 32256, 32272, 32288],
            [33264, 33280, 33296, 33312],
            [34288, 34304, 34320, 34336],
        ]
    )

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_trace_invalid_sys_arg():
    """The `sys` argument must be either a list or int."""
    with np.testing.assert_raises(ValueError):
        rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
        partial_trace(rho, "invalid_input")


def test_partial_trace_non_square_matrix_dim_2():
    """Matrix must be square for partial trace."""
    with np.testing.assert_raises(ValueError):
        rho = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        partial_trace(rho, 2, [2])


def test_partial_trace_cvxpy():
    """Test partial trace on cvxpy objects."""
    x_var = cvxpy.Variable((4, 4), hermitian=True)
    x_pt = partial_trace(x_var)
    np.testing.assert_equal(isinstance(x_pt, Vstack), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
