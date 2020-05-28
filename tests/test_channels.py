"""Tests for channels."""
import numpy as np

from toqito.channels import choi
from toqito.channels import dephasing
from toqito.channels import depolarizing
from toqito.channels import partial_trace
from toqito.channels import partial_transpose
from toqito.channels import realignment
from toqito.channels import reduction

from toqito.channel_ops import apply_map
from toqito.states import bell


def test_choi_standard():
    """The standard Choi map."""
    expected_res = np.array(
        [
            [1, 0, 0, 0, -1, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 1, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [-1, 0, 0, 0, -1, 0, 0, 0, 1],
        ]
    )

    res = choi()

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_choi_reduction():
    """
    The reduction map is the map R defined by: R(X) = Tr(X)I - X.

    The reduction map is the Choi map that arises when a = 0, b = c = 1.
    """
    expected_res = np.array(
        [
            [0, 0, 0, 0, -1, 0, 0, 0, -1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [-1, 0, 0, 0, -1, 0, 0, 0, 0],
        ]
    )

    res = choi(0, 1, 1)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_dephasing_completely_dephasing():
    """The completely dephasing channel kills everything off diagonal."""
    test_input_mat = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )

    expected_res = np.array([[1, 0, 0, 0], [0, 6, 0, 0], [0, 0, 11, 0], [0, 0, 0, 16]])

    res = apply_map(test_input_mat, dephasing(4))

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_dephasing_partially_dephasing():
    """The partially dephasing channel for `p = 0.5`."""
    test_input_mat = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )

    expected_res = np.array(
        [[17.5, 0, 0, 0], [0, 20, 0, 0], [0, 0, 22.5, 0], [0, 0, 0, 25]]
    )

    res = apply_map(test_input_mat, dephasing(4, 0.5))

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_depolarizing_complete_depolarizing():
    """Maps every density matrix to the maximally-mixed state."""
    test_input_mat = np.array(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )

    expected_res = (
        1
        / 4
        * np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )
    )

    res = apply_map(test_input_mat, depolarizing(4))

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_depolarizing_partially_depolarizing():
    """The partially depolarizing channel for `p = 0.5`."""
    test_input_mat = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )

    expected_res = np.array(
        [
            [17.125, 0.25, 0.375, 0.5],
            [0.625, 17.75, 0.875, 1],
            [1.125, 1.25, 18.375, 1.5],
            [1.625, 1.75, 1.875, 19],
        ]
    )

    res = apply_map(test_input_mat, depolarizing(4, 0.5))

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


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


def test_partial_transpose():
    """
    Default partial_transpose.

    By default, the partial_transpose function performs the transposition
    on the second subsystem.
    """
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array(
        [[1, 5, 3, 7], [2, 6, 4, 8], [9, 13, 11, 15], [10, 14, 12, 16]]
    )

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

    expected_res = np.array(
        [[1, 2, 9, 10], [5, 6, 13, 14], [3, 4, 11, 12], [7, 8, 15, 16]]
    )

    res = partial_transpose(test_input_mat, 1)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_sys_vec():
    """Partial transpose on matrix with `sys` defined as vector."""
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array(
        [[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]]
    )

    res = partial_transpose(test_input_mat, [1, 2])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_sys_vec_dim_vec():
    """Variables `sys` and `dim` defined as vector."""
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array(
        [[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]]
    )

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
    res = np.linalg.norm(
        partial_transpose(test_input_mat, [1, 2]) - test_input_mat.conj().T
    )
    expected_res = 0

    np.testing.assert_equal(np.isclose(res, expected_res), True)


def test_partial_transpose_16_by_16():
    """Partial transpose on a 16-by-16 matrix."""
    test_input_mat = np.arange(1, 257).reshape(16, 16)
    res = partial_transpose(test_input_mat, [1, 3], [2, 2, 2, 2])
    first_expected_row = np.array(
        [1, 2, 33, 34, 5, 6, 37, 38, 129, 130, 161, 162, 133, 134, 165, 166]
    )

    first_expected_col = np.array(
        [1, 17, 3, 19, 65, 81, 67, 83, 9, 25, 11, 27, 73, 89, 75, 91]
    )

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


def test_realignment_two_qubit():
    """
    Standard realignment map.

    When viewed as a map on block matrices, the realignment map takes each
    block of the original matrix and makes its vectorization the rows of
    the realignment matrix. This is illustrated by the following small
    example:
    """
    test_input_mat = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )

    expected_res = np.array(
        [[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]]
    )

    res = realignment(test_input_mat)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


# def test_realignment_non_square(self):
#     """
#     The realignment map sends |i⟩⟨j|⊗|k⟩⟨ℓ| to |i⟩⟨k|⊗|j⟩⟨ℓ|. Thus it
#     changes the dimensions of matrices if the subsystems aren't square
#     and of the same size. The following code computes the realignment of
#     an operator X∈M5,2⊗M3,7:
#     """
#     test_input_mat = np.reshape(list(range(1, 211)), (15, 14))
#     expected_res = np.array(
#         [[1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 29,
#           30, 31, 32, 33, 34, 35],
#          [8, 9, 10, 11, 12, 13, 14, 22, 23, 24, 25, 26, 27, 28,
#           36, 37, 38, 39, 40, 41, 42],
#          [43, 44, 45, 46, 47, 48, 49, 57, 58, 59, 60, 61, 62, 63,
#           71, 72, 73, 74, 75, 76, 77],
#          [50, 51, 52, 53, 54, 55, 56, 64, 65, 66, 67, 68, 69, 70,
#           78, 79, 80, 81, 82, 83, 84],
#          [85, 86, 87, 88, 89, 90, 91, 99, 100, 101, 102, 103, 104,
#           105, 113, 114, 115, 116, 117, 118, 119],
#          [92, 93, 94, 95, 96, 97, 98, 106, 107, 108, 109, 110, 111,
#           112, 120, 121, 122, 123, 124, 125, 126],
#          [127, 128, 129, 130, 131, 132, 133, 141, 142, 143, 144, 145,
#           146, 147, 155, 156, 157, 158, 159, 160, 161],
#          [134, 135, 136, 137, 138, 139, 140, 148, 149, 150, 151, 152,
#           153, 154, 162, 163, 164, 165, 166, 167, 168],
#          [169, 170, 171, 172, 173, 174, 175, 183, 184, 185, 186, 187,
#           188, 189, 197, 198, 199, 200, 201, 202, 203],
#          [176, 177, 178, 179, 180, 181, 182, 190, 191, 192, 193, 194,
#           195, 196, 204, 205, 206, 207, 208, 209, 210]
#          ])
#     res = realignment(test_input_mat, [[5, 3], [2, 7]])
#     bool_mat = np.isclose(expected_res, res)
#     self.assertEqual(np.all(bool_mat), True)

# def test_realignment_int_dim(self):
#     """
#     """
#     test_input_mat = np.array([[1, 2, 3, 4],
#                                [5, 6, 7, 8],
#                                [9, 10, 11, 12],
#                                [13, 14, 15, 16]])
#
#     expected_res = np.array([[1, 5, 9, 13, 2, 6, 10, 14, 3, 7,
#     11, 15, 4, 8, 12, 16]])
#
#     res = realignment(test_input_mat, 1)
#
#     bool_mat = np.isclose(expected_res, res)
#     self.assertEqual(np.all(bool_mat), True)


def test_reduction_map():
    """Test for the standard reduction map."""
    res = reduction(3)
    np.testing.assert_equal(res[4, 0], -1)
    np.testing.assert_equal(res[8, 0], -1)
    np.testing.assert_equal(res[1, 1], 1)
    np.testing.assert_equal(res[2, 2], 1)
    np.testing.assert_equal(res[3, 3], 1)
    np.testing.assert_equal(res[0, 4], -1)
    np.testing.assert_equal(res[8, 4], -1)
    np.testing.assert_equal(res[5, 5], 1)
    np.testing.assert_equal(res[6, 6], 1)
    np.testing.assert_equal(res[7, 7], 1)
    np.testing.assert_equal(res[0, 8], -1)
    np.testing.assert_equal(res[4, 8], -1)


if __name__ == "__main__":
    np.testing.run_module_suite()
