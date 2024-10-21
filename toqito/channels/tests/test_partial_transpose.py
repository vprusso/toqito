"""Test partial_transpose."""

import cvxpy
import numpy as np
from cvxpy.atoms.affine.vstack import Vstack

from toqito.channels import partial_transpose
from toqito.states import bell


def test_partial_transpose_bipartite():
    """Partial transpose of bipartite systems."""
    rho = np.arange(16).reshape(4, 4)

    # Partial transpose of first subsystem:
    res = partial_transpose(rho, [0], [2, 2])
    expected_res = np.array([[0, 1, 8, 9], [4, 5, 12, 13], [2, 3, 10, 11], [6, 7, 14, 15]])
    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose of second subsystem:
    res = partial_transpose(rho, [1], [2, 2])
    expected_res = np.array([[0, 4, 2, 6], [1, 5, 3, 7], [8, 12, 10, 14], [9, 13, 11, 15]])
    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Full transpose:
    res = partial_transpose(rho, [0, 1], [2, 2])
    expected_res = np.array([[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]])
    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_sys_int():
    """Partial transpose `sys` argument is provided as `int`."""
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[1, 2, 9, 10], [5, 6, 13, 14], [3, 4, 11, 12], [7, 8, 15, 16]])

    res = partial_transpose(test_input_mat, sys=0)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose():
    """Default partial_transpose.

    By default, the partial_transpose function performs the transposition
    on the second subsystem.
    """
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[1, 5, 3, 7], [2, 6, 4, 8], [9, 13, 11, 15], [10, 14, 12, 16]])

    res = partial_transpose(test_input_mat)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_sys():
    """Default partial transpose `sys` argument.

    By specifying the `sys` argument, you can perform the transposition on
    the first subsystem instead:
    """
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[1, 2, 9, 10], [5, 6, 13, 14], [3, 4, 11, 12], [7, 8, 15, 16]])

    res = partial_transpose(test_input_mat, [0])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_sys_vec():
    """Partial transpose on matrix with `sys` defined as vector."""
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]])

    res = partial_transpose(test_input_mat, [0, 1])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_sys_vec_dim_vec():
    """Variables `sys` and `dim` defined as vector."""
    test_input_mat = np.arange(1, 17).reshape(4, 4)

    expected_res = np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]])

    res = partial_transpose(test_input_mat, [0, 1], [2, 2])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_norm_diff():
    """Apply partial transpose to first and second subsystem.

    Applying the transpose to both the first and second subsystems results
    in the standard transpose of the matrix.
    """
    test_input_mat = np.arange(1, 17).reshape(4, 4)
    res = np.linalg.norm(partial_transpose(test_input_mat, [0, 1]) - test_input_mat.conj().T)
    expected_res = 0

    np.testing.assert_equal(np.isclose(res, expected_res), True)


def test_partial_transpose_8_by_8_subsystems_2_2_2():
    """Partial transpose on a 8-by-8 matrix on 2 x 2 x 2 subsystems."""
    test_input_mat = np.arange(1, 65).reshape(8, 8)

    # Partial transpose on first subsystem:
    pt_1 = partial_transpose(test_input_mat, [0], [2, 2, 2])
    expected_pt_1 = np.array(
        [
            [1, 2, 3, 4, 33, 34, 35, 36],
            [9, 10, 11, 12, 41, 42, 43, 44],
            [17, 18, 19, 20, 49, 50, 51, 52],
            [25, 26, 27, 28, 57, 58, 59, 60],
            [5, 6, 7, 8, 37, 38, 39, 40],
            [13, 14, 15, 16, 45, 46, 47, 48],
            [21, 22, 23, 24, 53, 54, 55, 56],
            [29, 30, 31, 32, 61, 62, 63, 64],
        ]
    )
    bool_mat = np.isclose(expected_pt_1, pt_1)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose on second subsystem:
    pt_2 = partial_transpose(test_input_mat, [1], [2, 2, 2])
    expected_pt_2 = np.array(
        [
            [1, 2, 17, 18, 5, 6, 21, 22],
            [9, 10, 25, 26, 13, 14, 29, 30],
            [3, 4, 19, 20, 7, 8, 23, 24],
            [11, 12, 27, 28, 15, 16, 31, 32],
            [33, 34, 49, 50, 37, 38, 53, 54],
            [41, 42, 57, 58, 45, 46, 61, 62],
            [35, 36, 51, 52, 39, 40, 55, 56],
            [43, 44, 59, 60, 47, 48, 63, 64],
        ]
    )
    bool_mat = np.isclose(expected_pt_2, pt_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose on third subsystem:
    pt_3 = partial_transpose(test_input_mat, [2], [2, 2, 2])
    expected_pt_3 = np.array(
        [
            [1, 9, 3, 11, 5, 13, 7, 15],
            [2, 10, 4, 12, 6, 14, 8, 16],
            [17, 25, 19, 27, 21, 29, 23, 31],
            [18, 26, 20, 28, 22, 30, 24, 32],
            [33, 41, 35, 43, 37, 45, 39, 47],
            [34, 42, 36, 44, 38, 46, 40, 48],
            [49, 57, 51, 59, 53, 61, 55, 63],
            [50, 58, 52, 60, 54, 62, 56, 64],
        ]
    )
    bool_mat = np.isclose(expected_pt_3, pt_3)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose on first and second subsystem:
    pt_1_2 = partial_transpose(test_input_mat, [0, 1], [2, 2, 2])
    expected_pt_1_2 = np.array(
        [
            [1, 2, 17, 18, 33, 34, 49, 50],
            [9, 10, 25, 26, 41, 42, 57, 58],
            [3, 4, 19, 20, 35, 36, 51, 52],
            [11, 12, 27, 28, 43, 44, 59, 60],
            [5, 6, 21, 22, 37, 38, 53, 54],
            [13, 14, 29, 30, 45, 46, 61, 62],
            [7, 8, 23, 24, 39, 40, 55, 56],
            [15, 16, 31, 32, 47, 48, 63, 64],
        ]
    )
    bool_mat = np.isclose(expected_pt_1_2, pt_1_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose on first and third subsystem:
    pt_1_3 = partial_transpose(test_input_mat, [0, 2], [2, 2, 2])
    expected_pt_1_3 = np.array(
        [
            [1, 9, 3, 11, 33, 41, 35, 43],
            [2, 10, 4, 12, 34, 42, 36, 44],
            [17, 25, 19, 27, 49, 57, 51, 59],
            [18, 26, 20, 28, 50, 58, 52, 60],
            [5, 13, 7, 15, 37, 45, 39, 47],
            [6, 14, 8, 16, 38, 46, 40, 48],
            [21, 29, 23, 31, 53, 61, 55, 63],
            [22, 30, 24, 32, 54, 62, 56, 64],
        ]
    )
    bool_mat = np.isclose(expected_pt_1_3, pt_1_3)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose on second and third subsystem:
    pt_2_3 = partial_transpose(test_input_mat, [1, 2], [2, 2, 2])
    expected_pt_2_3 = np.array(
        [
            [1, 9, 17, 25, 5, 13, 21, 29],
            [2, 10, 18, 26, 6, 14, 22, 30],
            [3, 11, 19, 27, 7, 15, 23, 31],
            [4, 12, 20, 28, 8, 16, 24, 32],
            [33, 41, 49, 57, 37, 45, 53, 61],
            [34, 42, 50, 58, 38, 46, 54, 62],
            [35, 43, 51, 59, 39, 47, 55, 63],
            [36, 44, 52, 60, 40, 48, 56, 64],
        ]
    )
    bool_mat = np.isclose(expected_pt_2_3, pt_2_3)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_8_by_8_subsystems_2_4():
    """Partial transpose on a 8-by-8 matrix on 2 x 4 subsystems."""
    test_input_mat = np.arange(1, 65).reshape(8, 8)

    # Partial transpose on the first subsystem:
    pt_1 = partial_transpose(test_input_mat, [0], [2, 4])
    expected_pt_1 = np.array(
        [
            [1, 2, 3, 4, 33, 34, 35, 36],
            [9, 10, 11, 12, 41, 42, 43, 44],
            [17, 18, 19, 20, 49, 50, 51, 52],
            [25, 26, 27, 28, 57, 58, 59, 60],
            [5, 6, 7, 8, 37, 38, 39, 40],
            [13, 14, 15, 16, 45, 46, 47, 48],
            [21, 22, 23, 24, 53, 54, 55, 56],
            [29, 30, 31, 32, 61, 62, 63, 64],
        ]
    )
    bool_mat = np.isclose(expected_pt_1, pt_1)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose on the second subsystem:
    pt_2 = partial_transpose(test_input_mat, [1], [2, 4])
    expected_pt_2 = np.array(
        [
            [1, 9, 17, 25, 5, 13, 21, 29],
            [2, 10, 18, 26, 6, 14, 22, 30],
            [3, 11, 19, 27, 7, 15, 23, 31],
            [4, 12, 20, 28, 8, 16, 24, 32],
            [33, 41, 49, 57, 37, 45, 53, 61],
            [34, 42, 50, 58, 38, 46, 54, 62],
            [35, 43, 51, 59, 39, 47, 55, 63],
            [36, 44, 52, 60, 40, 48, 56, 64],
        ]
    )
    bool_mat = np.isclose(expected_pt_2, pt_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose on the first and second subsystem:
    pt_1_2 = partial_transpose(test_input_mat, [0, 1], [2, 4])
    expected_pt_1_2 = np.array(
        [
            [1, 9, 17, 25, 33, 41, 49, 57],
            [2, 10, 18, 26, 34, 42, 50, 58],
            [3, 11, 19, 27, 35, 43, 51, 59],
            [4, 12, 20, 28, 36, 44, 52, 60],
            [5, 13, 21, 29, 37, 45, 53, 61],
            [6, 14, 22, 30, 38, 46, 54, 62],
            [7, 15, 23, 31, 39, 47, 55, 63],
            [8, 16, 24, 32, 40, 48, 56, 64],
        ]
    )
    bool_mat = np.isclose(expected_pt_1_2, pt_1_2)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_8_by_8_subsystems_4_2():
    """Partial transpose on a 8-by-8 matrix on 4 x 2 subsystems."""
    test_input_mat = np.arange(1, 65).reshape(8, 8)

    # Partial transpose on the first subsystem:
    pt_1 = partial_transpose(test_input_mat, [0], [4, 2])
    expected_pt_1 = np.array(
        [
            [1, 2, 17, 18, 33, 34, 49, 50],
            [9, 10, 25, 26, 41, 42, 57, 58],
            [3, 4, 19, 20, 35, 36, 51, 52],
            [11, 12, 27, 28, 43, 44, 59, 60],
            [5, 6, 21, 22, 37, 38, 53, 54],
            [13, 14, 29, 30, 45, 46, 61, 62],
            [7, 8, 23, 24, 39, 40, 55, 56],
            [15, 16, 31, 32, 47, 48, 63, 64],
        ]
    )
    bool_mat = np.isclose(expected_pt_1, pt_1)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose on the second subsystem:
    pt_2 = partial_transpose(test_input_mat, [1], [4, 2])
    expected_pt_2 = np.array(
        [
            [1, 9, 3, 11, 5, 13, 7, 15],
            [2, 10, 4, 12, 6, 14, 8, 16],
            [17, 25, 19, 27, 21, 29, 23, 31],
            [18, 26, 20, 28, 22, 30, 24, 32],
            [33, 41, 35, 43, 37, 45, 39, 47],
            [34, 42, 36, 44, 38, 46, 40, 48],
            [49, 57, 51, 59, 53, 61, 55, 63],
            [50, 58, 52, 60, 54, 62, 56, 64],
        ]
    )
    bool_mat = np.isclose(expected_pt_2, pt_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose on the first and second subsystem:
    pt_1_2 = partial_transpose(test_input_mat, [0, 1], [4, 2])
    expected_pt_1_2 = np.array(
        [
            [1, 9, 17, 25, 33, 41, 49, 57],
            [2, 10, 18, 26, 34, 42, 50, 58],
            [3, 11, 19, 27, 35, 43, 51, 59],
            [4, 12, 20, 28, 36, 44, 52, 60],
            [5, 13, 21, 29, 37, 45, 53, 61],
            [6, 14, 22, 30, 38, 46, 54, 62],
            [7, 15, 23, 31, 39, 47, 55, 63],
            [8, 16, 24, 32, 40, 48, 56, 64],
        ]
    )
    bool_mat = np.isclose(expected_pt_1_2, pt_1_2)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_6_by_6_subsystems_2_3():
    """Partial transpose on a 6-by-6 matrix on 2 x 3 subsystems."""
    test_input_mat = np.arange(1, 37).reshape(6, 6)

    # Partial transpose on first subsystem:
    pt_1 = partial_transpose(test_input_mat, [0], [2, 3])
    expected_pt_1 = np.array(
        [
            [1, 2, 3, 19, 20, 21],
            [7, 8, 9, 25, 26, 27],
            [13, 14, 15, 31, 32, 33],
            [4, 5, 6, 22, 23, 24],
            [10, 11, 12, 28, 29, 30],
            [16, 17, 18, 34, 35, 36],
        ]
    )
    bool_mat = np.isclose(expected_pt_1, pt_1)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose on second subsystem:
    pt_2 = partial_transpose(test_input_mat, [1], [2, 3])
    expected_pt_2 = np.array(
        [
            [1, 7, 13, 4, 10, 16],
            [2, 8, 14, 5, 11, 17],
            [3, 9, 15, 6, 12, 18],
            [19, 25, 31, 22, 28, 34],
            [20, 26, 32, 23, 29, 35],
            [21, 27, 33, 24, 30, 36],
        ]
    )
    bool_mat = np.isclose(expected_pt_2, pt_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose on first and second subsystems:
    pt_1_2 = partial_transpose(test_input_mat, [0, 1], [2, 3])
    expected_pt_1_2 = np.array(
        [
            [1, 7, 13, 19, 25, 31],
            [2, 8, 14, 20, 26, 32],
            [3, 9, 15, 21, 27, 33],
            [4, 10, 16, 22, 28, 34],
            [5, 11, 17, 23, 29, 35],
            [6, 12, 18, 24, 30, 36],
        ]
    )
    bool_mat = np.isclose(expected_pt_1_2, pt_1_2)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_6_by_6_subsystems_3_2():
    """Partial transpose on a 6-by-6 matrix on 3 x 2 subsystems."""
    test_input_mat = np.arange(1, 37).reshape(6, 6)

    # Partial transpose on first subsystem:
    pt_1 = partial_transpose(test_input_mat, [0], [3, 2])
    expected_pt_1 = np.array(
        [
            [1, 2, 13, 14, 25, 26],
            [7, 8, 19, 20, 31, 32],
            [3, 4, 15, 16, 27, 28],
            [9, 10, 21, 22, 33, 34],
            [5, 6, 17, 18, 29, 30],
            [11, 12, 23, 24, 35, 36],
        ]
    )
    bool_mat = np.isclose(expected_pt_1, pt_1)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose on second subsystem:
    pt_2 = partial_transpose(test_input_mat, [1], [3, 2])
    expected_pt_2 = np.array(
        [
            [1, 7, 3, 9, 5, 11],
            [2, 8, 4, 10, 6, 12],
            [13, 19, 15, 21, 17, 23],
            [14, 20, 16, 22, 18, 24],
            [25, 31, 27, 33, 29, 35],
            [26, 32, 28, 34, 30, 36],
        ]
    )
    bool_mat = np.isclose(expected_pt_2, pt_2)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose on first and second subsystems:
    pt_1_2 = partial_transpose(test_input_mat, [0, 1], [3, 2])
    expected_pt_1_2 = np.array(
        [
            [1, 7, 13, 19, 25, 31],
            [2, 8, 14, 20, 26, 32],
            [3, 9, 15, 21, 27, 33],
            [4, 10, 16, 22, 28, 34],
            [5, 11, 17, 23, 29, 35],
            [6, 12, 18, 24, 30, 36],
        ]
    )
    bool_mat = np.isclose(expected_pt_1_2, pt_1_2)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_16_by_16_subsystems_2_2_2_2():
    """Partial transpose on a 16-by-16 matrix on 2 x 2 x 2 x 2 subsystems."""
    rho = np.arange(256).reshape(16, 16)

    # Partial transpose of first subsystem:
    res = partial_transpose(rho, [0], [2, 2, 2, 2])
    expected_res = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 128, 129, 130, 131, 132, 133, 134, 135],
            [16, 17, 18, 19, 20, 21, 22, 23, 144, 145, 146, 147, 148, 149, 150, 151],
            [32, 33, 34, 35, 36, 37, 38, 39, 160, 161, 162, 163, 164, 165, 166, 167],
            [48, 49, 50, 51, 52, 53, 54, 55, 176, 177, 178, 179, 180, 181, 182, 183],
            [64, 65, 66, 67, 68, 69, 70, 71, 192, 193, 194, 195, 196, 197, 198, 199],
            [80, 81, 82, 83, 84, 85, 86, 87, 208, 209, 210, 211, 212, 213, 214, 215],
            [96, 97, 98, 99, 100, 101, 102, 103, 224, 225, 226, 227, 228, 229, 230, 231],
            [112, 113, 114, 115, 116, 117, 118, 119, 240, 241, 242, 243, 244, 245, 246, 247],
            [8, 9, 10, 11, 12, 13, 14, 15, 136, 137, 138, 139, 140, 141, 142, 143],
            [24, 25, 26, 27, 28, 29, 30, 31, 152, 153, 154, 155, 156, 157, 158, 159],
            [40, 41, 42, 43, 44, 45, 46, 47, 168, 169, 170, 171, 172, 173, 174, 175],
            [56, 57, 58, 59, 60, 61, 62, 63, 184, 185, 186, 187, 188, 189, 190, 191],
            [72, 73, 74, 75, 76, 77, 78, 79, 200, 201, 202, 203, 204, 205, 206, 207],
            [88, 89, 90, 91, 92, 93, 94, 95, 216, 217, 218, 219, 220, 221, 222, 223],
            [104, 105, 106, 107, 108, 109, 110, 111, 232, 233, 234, 235, 236, 237, 238, 239],
            [120, 121, 122, 123, 124, 125, 126, 127, 248, 249, 250, 251, 252, 253, 254, 255],
        ]
    )
    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose of second subsystem:
    res = partial_transpose(rho, [1], [2, 2, 2, 2])
    expected_res = np.array(
        [
            [0, 1, 2, 3, 64, 65, 66, 67, 8, 9, 10, 11, 72, 73, 74, 75],
            [16, 17, 18, 19, 80, 81, 82, 83, 24, 25, 26, 27, 88, 89, 90, 91],
            [32, 33, 34, 35, 96, 97, 98, 99, 40, 41, 42, 43, 104, 105, 106, 107],
            [48, 49, 50, 51, 112, 113, 114, 115, 56, 57, 58, 59, 120, 121, 122, 123],
            [4, 5, 6, 7, 68, 69, 70, 71, 12, 13, 14, 15, 76, 77, 78, 79],
            [20, 21, 22, 23, 84, 85, 86, 87, 28, 29, 30, 31, 92, 93, 94, 95],
            [36, 37, 38, 39, 100, 101, 102, 103, 44, 45, 46, 47, 108, 109, 110, 111],
            [52, 53, 54, 55, 116, 117, 118, 119, 60, 61, 62, 63, 124, 125, 126, 127],
            [128, 129, 130, 131, 192, 193, 194, 195, 136, 137, 138, 139, 200, 201, 202, 203],
            [144, 145, 146, 147, 208, 209, 210, 211, 152, 153, 154, 155, 216, 217, 218, 219],
            [160, 161, 162, 163, 224, 225, 226, 227, 168, 169, 170, 171, 232, 233, 234, 235],
            [176, 177, 178, 179, 240, 241, 242, 243, 184, 185, 186, 187, 248, 249, 250, 251],
            [132, 133, 134, 135, 196, 197, 198, 199, 140, 141, 142, 143, 204, 205, 206, 207],
            [148, 149, 150, 151, 212, 213, 214, 215, 156, 157, 158, 159, 220, 221, 222, 223],
            [164, 165, 166, 167, 228, 229, 230, 231, 172, 173, 174, 175, 236, 237, 238, 239],
            [180, 181, 182, 183, 244, 245, 246, 247, 188, 189, 190, 191, 252, 253, 254, 255],
        ]
    )
    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose of third subsystem
    res = partial_transpose(rho, [2], [2, 2, 2, 2])
    expected_res = np.array(
        [
            [0, 1, 32, 33, 4, 5, 36, 37, 8, 9, 40, 41, 12, 13, 44, 45],
            [16, 17, 48, 49, 20, 21, 52, 53, 24, 25, 56, 57, 28, 29, 60, 61],
            [2, 3, 34, 35, 6, 7, 38, 39, 10, 11, 42, 43, 14, 15, 46, 47],
            [18, 19, 50, 51, 22, 23, 54, 55, 26, 27, 58, 59, 30, 31, 62, 63],
            [64, 65, 96, 97, 68, 69, 100, 101, 72, 73, 104, 105, 76, 77, 108, 109],
            [80, 81, 112, 113, 84, 85, 116, 117, 88, 89, 120, 121, 92, 93, 124, 125],
            [66, 67, 98, 99, 70, 71, 102, 103, 74, 75, 106, 107, 78, 79, 110, 111],
            [82, 83, 114, 115, 86, 87, 118, 119, 90, 91, 122, 123, 94, 95, 126, 127],
            [128, 129, 160, 161, 132, 133, 164, 165, 136, 137, 168, 169, 140, 141, 172, 173],
            [144, 145, 176, 177, 148, 149, 180, 181, 152, 153, 184, 185, 156, 157, 188, 189],
            [130, 131, 162, 163, 134, 135, 166, 167, 138, 139, 170, 171, 142, 143, 174, 175],
            [146, 147, 178, 179, 150, 151, 182, 183, 154, 155, 186, 187, 158, 159, 190, 191],
            [192, 193, 224, 225, 196, 197, 228, 229, 200, 201, 232, 233, 204, 205, 236, 237],
            [208, 209, 240, 241, 212, 213, 244, 245, 216, 217, 248, 249, 220, 221, 252, 253],
            [194, 195, 226, 227, 198, 199, 230, 231, 202, 203, 234, 235, 206, 207, 238, 239],
            [210, 211, 242, 243, 214, 215, 246, 247, 218, 219, 250, 251, 222, 223, 254, 255],
        ]
    )
    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose of fourth subsystem
    res = partial_transpose(rho, [3], [2, 2, 2, 2])
    expected_res = np.array(
        [
            [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30],
            [1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31],
            [32, 48, 34, 50, 36, 52, 38, 54, 40, 56, 42, 58, 44, 60, 46, 62],
            [33, 49, 35, 51, 37, 53, 39, 55, 41, 57, 43, 59, 45, 61, 47, 63],
            [64, 80, 66, 82, 68, 84, 70, 86, 72, 88, 74, 90, 76, 92, 78, 94],
            [65, 81, 67, 83, 69, 85, 71, 87, 73, 89, 75, 91, 77, 93, 79, 95],
            [96, 112, 98, 114, 100, 116, 102, 118, 104, 120, 106, 122, 108, 124, 110, 126],
            [97, 113, 99, 115, 101, 117, 103, 119, 105, 121, 107, 123, 109, 125, 111, 127],
            [128, 144, 130, 146, 132, 148, 134, 150, 136, 152, 138, 154, 140, 156, 142, 158],
            [129, 145, 131, 147, 133, 149, 135, 151, 137, 153, 139, 155, 141, 157, 143, 159],
            [160, 176, 162, 178, 164, 180, 166, 182, 168, 184, 170, 186, 172, 188, 174, 190],
            [161, 177, 163, 179, 165, 181, 167, 183, 169, 185, 171, 187, 173, 189, 175, 191],
            [192, 208, 194, 210, 196, 212, 198, 214, 200, 216, 202, 218, 204, 220, 206, 222],
            [193, 209, 195, 211, 197, 213, 199, 215, 201, 217, 203, 219, 205, 221, 207, 223],
            [224, 240, 226, 242, 228, 244, 230, 246, 232, 248, 234, 250, 236, 252, 238, 254],
            [225, 241, 227, 243, 229, 245, 231, 247, 233, 249, 235, 251, 237, 253, 239, 255],
        ]
    )
    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose of first and second subsystem:
    res = partial_transpose(rho, [0, 1], [2, 2, 2, 2])
    expected_res = np.array(
        [
            [0, 1, 2, 3, 64, 65, 66, 67, 128, 129, 130, 131, 192, 193, 194, 195],
            [16, 17, 18, 19, 80, 81, 82, 83, 144, 145, 146, 147, 208, 209, 210, 211],
            [32, 33, 34, 35, 96, 97, 98, 99, 160, 161, 162, 163, 224, 225, 226, 227],
            [48, 49, 50, 51, 112, 113, 114, 115, 176, 177, 178, 179, 240, 241, 242, 243],
            [4, 5, 6, 7, 68, 69, 70, 71, 132, 133, 134, 135, 196, 197, 198, 199],
            [20, 21, 22, 23, 84, 85, 86, 87, 148, 149, 150, 151, 212, 213, 214, 215],
            [36, 37, 38, 39, 100, 101, 102, 103, 164, 165, 166, 167, 228, 229, 230, 231],
            [52, 53, 54, 55, 116, 117, 118, 119, 180, 181, 182, 183, 244, 245, 246, 247],
            [8, 9, 10, 11, 72, 73, 74, 75, 136, 137, 138, 139, 200, 201, 202, 203],
            [24, 25, 26, 27, 88, 89, 90, 91, 152, 153, 154, 155, 216, 217, 218, 219],
            [40, 41, 42, 43, 104, 105, 106, 107, 168, 169, 170, 171, 232, 233, 234, 235],
            [56, 57, 58, 59, 120, 121, 122, 123, 184, 185, 186, 187, 248, 249, 250, 251],
            [12, 13, 14, 15, 76, 77, 78, 79, 140, 141, 142, 143, 204, 205, 206, 207],
            [28, 29, 30, 31, 92, 93, 94, 95, 156, 157, 158, 159, 220, 221, 222, 223],
            [44, 45, 46, 47, 108, 109, 110, 111, 172, 173, 174, 175, 236, 237, 238, 239],
            [60, 61, 62, 63, 124, 125, 126, 127, 188, 189, 190, 191, 252, 253, 254, 255],
        ]
    )
    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose of first and third subsystem:
    res = partial_transpose(rho, [0, 2], [2, 2, 2, 2])
    expected_res = np.array(
        [
            [0, 1, 32, 33, 4, 5, 36, 37, 128, 129, 160, 161, 132, 133, 164, 165],
            [16, 17, 48, 49, 20, 21, 52, 53, 144, 145, 176, 177, 148, 149, 180, 181],
            [2, 3, 34, 35, 6, 7, 38, 39, 130, 131, 162, 163, 134, 135, 166, 167],
            [18, 19, 50, 51, 22, 23, 54, 55, 146, 147, 178, 179, 150, 151, 182, 183],
            [64, 65, 96, 97, 68, 69, 100, 101, 192, 193, 224, 225, 196, 197, 228, 229],
            [80, 81, 112, 113, 84, 85, 116, 117, 208, 209, 240, 241, 212, 213, 244, 245],
            [66, 67, 98, 99, 70, 71, 102, 103, 194, 195, 226, 227, 198, 199, 230, 231],
            [82, 83, 114, 115, 86, 87, 118, 119, 210, 211, 242, 243, 214, 215, 246, 247],
            [8, 9, 40, 41, 12, 13, 44, 45, 136, 137, 168, 169, 140, 141, 172, 173],
            [24, 25, 56, 57, 28, 29, 60, 61, 152, 153, 184, 185, 156, 157, 188, 189],
            [10, 11, 42, 43, 14, 15, 46, 47, 138, 139, 170, 171, 142, 143, 174, 175],
            [26, 27, 58, 59, 30, 31, 62, 63, 154, 155, 186, 187, 158, 159, 190, 191],
            [72, 73, 104, 105, 76, 77, 108, 109, 200, 201, 232, 233, 204, 205, 236, 237],
            [88, 89, 120, 121, 92, 93, 124, 125, 216, 217, 248, 249, 220, 221, 252, 253],
            [74, 75, 106, 107, 78, 79, 110, 111, 202, 203, 234, 235, 206, 207, 238, 239],
            [90, 91, 122, 123, 94, 95, 126, 127, 218, 219, 250, 251, 222, 223, 254, 255],
        ]
    )
    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose of first and fourth subsystem
    res = partial_transpose(rho, [0, 3], [2, 2, 2, 2])
    expected_res = np.array(
        [
            [0, 16, 2, 18, 4, 20, 6, 22, 128, 144, 130, 146, 132, 148, 134, 150],
            [1, 17, 3, 19, 5, 21, 7, 23, 129, 145, 131, 147, 133, 149, 135, 151],
            [32, 48, 34, 50, 36, 52, 38, 54, 160, 176, 162, 178, 164, 180, 166, 182],
            [33, 49, 35, 51, 37, 53, 39, 55, 161, 177, 163, 179, 165, 181, 167, 183],
            [64, 80, 66, 82, 68, 84, 70, 86, 192, 208, 194, 210, 196, 212, 198, 214],
            [65, 81, 67, 83, 69, 85, 71, 87, 193, 209, 195, 211, 197, 213, 199, 215],
            [96, 112, 98, 114, 100, 116, 102, 118, 224, 240, 226, 242, 228, 244, 230, 246],
            [97, 113, 99, 115, 101, 117, 103, 119, 225, 241, 227, 243, 229, 245, 231, 247],
            [8, 24, 10, 26, 12, 28, 14, 30, 136, 152, 138, 154, 140, 156, 142, 158],
            [9, 25, 11, 27, 13, 29, 15, 31, 137, 153, 139, 155, 141, 157, 143, 159],
            [40, 56, 42, 58, 44, 60, 46, 62, 168, 184, 170, 186, 172, 188, 174, 190],
            [41, 57, 43, 59, 45, 61, 47, 63, 169, 185, 171, 187, 173, 189, 175, 191],
            [72, 88, 74, 90, 76, 92, 78, 94, 200, 216, 202, 218, 204, 220, 206, 222],
            [73, 89, 75, 91, 77, 93, 79, 95, 201, 217, 203, 219, 205, 221, 207, 223],
            [104, 120, 106, 122, 108, 124, 110, 126, 232, 248, 234, 250, 236, 252, 238, 254],
            [105, 121, 107, 123, 109, 125, 111, 127, 233, 249, 235, 251, 237, 253, 239, 255],
        ]
    )
    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose of second and third subsystem:
    res = partial_transpose(rho, [1, 2], [2, 2, 2, 2])
    expected_res = np.array(
        [
            [0, 1, 32, 33, 64, 65, 96, 97, 8, 9, 40, 41, 72, 73, 104, 105],
            [16, 17, 48, 49, 80, 81, 112, 113, 24, 25, 56, 57, 88, 89, 120, 121],
            [2, 3, 34, 35, 66, 67, 98, 99, 10, 11, 42, 43, 74, 75, 106, 107],
            [18, 19, 50, 51, 82, 83, 114, 115, 26, 27, 58, 59, 90, 91, 122, 123],
            [4, 5, 36, 37, 68, 69, 100, 101, 12, 13, 44, 45, 76, 77, 108, 109],
            [20, 21, 52, 53, 84, 85, 116, 117, 28, 29, 60, 61, 92, 93, 124, 125],
            [6, 7, 38, 39, 70, 71, 102, 103, 14, 15, 46, 47, 78, 79, 110, 111],
            [22, 23, 54, 55, 86, 87, 118, 119, 30, 31, 62, 63, 94, 95, 126, 127],
            [128, 129, 160, 161, 192, 193, 224, 225, 136, 137, 168, 169, 200, 201, 232, 233],
            [144, 145, 176, 177, 208, 209, 240, 241, 152, 153, 184, 185, 216, 217, 248, 249],
            [130, 131, 162, 163, 194, 195, 226, 227, 138, 139, 170, 171, 202, 203, 234, 235],
            [146, 147, 178, 179, 210, 211, 242, 243, 154, 155, 186, 187, 218, 219, 250, 251],
            [132, 133, 164, 165, 196, 197, 228, 229, 140, 141, 172, 173, 204, 205, 236, 237],
            [148, 149, 180, 181, 212, 213, 244, 245, 156, 157, 188, 189, 220, 221, 252, 253],
            [134, 135, 166, 167, 198, 199, 230, 231, 142, 143, 174, 175, 206, 207, 238, 239],
            [150, 151, 182, 183, 214, 215, 246, 247, 158, 159, 190, 191, 222, 223, 254, 255],
        ]
    )
    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose of second and fourth subsystem:
    res = partial_transpose(rho, [1, 3], [2, 2, 2, 2])
    expected_res = np.array(
        [
            [0, 16, 2, 18, 64, 80, 66, 82, 8, 24, 10, 26, 72, 88, 74, 90],
            [1, 17, 3, 19, 65, 81, 67, 83, 9, 25, 11, 27, 73, 89, 75, 91],
            [32, 48, 34, 50, 96, 112, 98, 114, 40, 56, 42, 58, 104, 120, 106, 122],
            [33, 49, 35, 51, 97, 113, 99, 115, 41, 57, 43, 59, 105, 121, 107, 123],
            [4, 20, 6, 22, 68, 84, 70, 86, 12, 28, 14, 30, 76, 92, 78, 94],
            [5, 21, 7, 23, 69, 85, 71, 87, 13, 29, 15, 31, 77, 93, 79, 95],
            [36, 52, 38, 54, 100, 116, 102, 118, 44, 60, 46, 62, 108, 124, 110, 126],
            [37, 53, 39, 55, 101, 117, 103, 119, 45, 61, 47, 63, 109, 125, 111, 127],
            [128, 144, 130, 146, 192, 208, 194, 210, 136, 152, 138, 154, 200, 216, 202, 218],
            [129, 145, 131, 147, 193, 209, 195, 211, 137, 153, 139, 155, 201, 217, 203, 219],
            [160, 176, 162, 178, 224, 240, 226, 242, 168, 184, 170, 186, 232, 248, 234, 250],
            [161, 177, 163, 179, 225, 241, 227, 243, 169, 185, 171, 187, 233, 249, 235, 251],
            [132, 148, 134, 150, 196, 212, 198, 214, 140, 156, 142, 158, 204, 220, 206, 222],
            [133, 149, 135, 151, 197, 213, 199, 215, 141, 157, 143, 159, 205, 221, 207, 223],
            [164, 180, 166, 182, 228, 244, 230, 246, 172, 188, 174, 190, 236, 252, 238, 254],
            [165, 181, 167, 183, 229, 245, 231, 247, 173, 189, 175, 191, 237, 253, 239, 255],
        ]
    )
    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)

    # Partial transpose of third and fourth subsystem
    res = partial_transpose(rho, [2, 3], [2, 2, 2, 2])
    expected_res = np.array(
        [
            [0, 16, 32, 48, 4, 20, 36, 52, 8, 24, 40, 56, 12, 28, 44, 60],
            [1, 17, 33, 49, 5, 21, 37, 53, 9, 25, 41, 57, 13, 29, 45, 61],
            [2, 18, 34, 50, 6, 22, 38, 54, 10, 26, 42, 58, 14, 30, 46, 62],
            [3, 19, 35, 51, 7, 23, 39, 55, 11, 27, 43, 59, 15, 31, 47, 63],
            [64, 80, 96, 112, 68, 84, 100, 116, 72, 88, 104, 120, 76, 92, 108, 124],
            [65, 81, 97, 113, 69, 85, 101, 117, 73, 89, 105, 121, 77, 93, 109, 125],
            [66, 82, 98, 114, 70, 86, 102, 118, 74, 90, 106, 122, 78, 94, 110, 126],
            [67, 83, 99, 115, 71, 87, 103, 119, 75, 91, 107, 123, 79, 95, 111, 127],
            [128, 144, 160, 176, 132, 148, 164, 180, 136, 152, 168, 184, 140, 156, 172, 188],
            [129, 145, 161, 177, 133, 149, 165, 181, 137, 153, 169, 185, 141, 157, 173, 189],
            [130, 146, 162, 178, 134, 150, 166, 182, 138, 154, 170, 186, 142, 158, 174, 190],
            [131, 147, 163, 179, 135, 151, 167, 183, 139, 155, 171, 187, 143, 159, 175, 191],
            [192, 208, 224, 240, 196, 212, 228, 244, 200, 216, 232, 248, 204, 220, 236, 252],
            [193, 209, 225, 241, 197, 213, 229, 245, 201, 217, 233, 249, 205, 221, 237, 253],
            [194, 210, 226, 242, 198, 214, 230, 246, 202, 218, 234, 250, 206, 222, 238, 254],
            [195, 211, 227, 243, 199, 215, 231, 247, 203, 219, 235, 251, 207, 223, 239, 255],
        ]
    )
    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_transpose_bell_state():
    """Test partial transpose on a Bell state."""
    rho = bell(2) @ bell(2).conj().T
    expected_res = np.array([[0, 0, 0, 1 / 2], [0, 1 / 2, 0, 0], [0, 0, 1 / 2, 0], [1 / 2, 0, 0, 0]])
    res = partial_transpose(rho)
    np.testing.assert_equal(np.allclose(res, expected_res), True)


def test_partial_transpose_non_square_matrix():
    """Matrix must be square."""
    with np.testing.assert_raises(ValueError):
        test_input_mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [13, 14, 15, 16]])
        partial_transpose(test_input_mat)


def test_partial_transpose_non_square():
    """Test partial transpose on non square matrices ."""
    rho = np.kron(np.eye(2, 3), np.ones((2, 3)))
    rho = np.kron(rho, np.eye(2, 3))

    dim = np.array([[2, 2, 2], [3, 3, 3]])

    res = partial_transpose(rho, sys=1, dim=dim)

    expected = np.kron(np.eye(2, 3), np.ones((3, 2)))
    expected = np.kron(expected, np.eye(2, 3))
    np.testing.assert_equal(np.allclose(res, expected), True)


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


def test_partial_transpose_three_subsystems():
    """Test partial transpose on 3 - subsystems ."""
    mat = np.arange(64).reshape((8, 8))
    input_mat = np.kron(np.eye(2, 2), mat)

    res = partial_transpose(input_mat, [1, 2, 3], [2, 2, 2, 2])

    expected = np.kron(np.eye(2, 2), mat.T)
    np.testing.assert_equal(np.allclose(res, expected), True)
