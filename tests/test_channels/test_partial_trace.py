"""Test partial_trace."""
import numpy as np

from toqito.channels import partial_trace


def test_partial_trace():
    """
    Standard call to partial_trace.

    By default, the partial_trace function takes the trace over the second
    subsystem.
    """
    test_input_mat = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )

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
    test_input_mat = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )

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
    test_input_mat = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )

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
    test_input_mat = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )

    expected_res = 34

    res = partial_trace(test_input_mat, 2, 1)

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


def test_partial_trace_invalid_sys_arg():
    """The `sys` argument must be either a list or int."""
    with np.testing.assert_raises(ValueError):
        rho = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )
        partial_trace(rho, "invalid_input")


def test_partial_trace_non_square_matrix_dim_2():
    """Matrix must be square for partial trace."""
    with np.testing.assert_raises(ValueError):
        rho = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        partial_trace(rho, 2, [2])


if __name__ == "__main__":
    np.testing.run_module_suite()
