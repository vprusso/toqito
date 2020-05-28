"""Tests for perms."""
import numpy as np

from toqito.perms import antisymmetric_projection
from toqito.perms import perm_sign
from toqito.perms import permutation_operator
from toqito.perms import permute_systems
from toqito.perms import swap
from toqito.perms import swap_operator
from toqito.perms import symmetric_projection
from toqito.perms import unique_perms


def test_antisymmetric_projection_d_2_p_1():
    """Dimension is 2 and p is equal to 1."""
    res = antisymmetric_projection(2, 1).todense()
    expected_res = np.array([[1, 0], [0, 1]])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_antisymmetric_projection_p_larger_than_d():
    """The `p` value is greater than the dimension `d`."""
    res = antisymmetric_projection(2, 3).todense()
    expected_res = np.zeros((8, 8))

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_antisymmetric_projection_2():
    """The dimension is 2."""
    res = antisymmetric_projection(2).todense()
    expected_res = np.array(
        [[0, 0, 0, 0], [0, 0.5, -0.5, 0], [0, -0.5, 0.5, 0], [0, 0, 0, 0]]
    )

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_antisymmetric_projection_3_3_true():
    """The `dim` is 3, the `p` is 3, and `partial` is True."""
    res = antisymmetric_projection(3, 3, True).todense()
    np.testing.assert_equal(np.isclose(res[5].item(), -0.40824829), True)


def test_perm_sign_small_example_even():
    """Small example when permutation is even."""
    res = perm_sign([1, 2, 3, 4])
    np.testing.assert_equal(res, 1)


def test_perm_sign_small_example_odd():
    """Small example when permutation is odd."""
    res = perm_sign([1, 2, 4, 3, 5])
    np.testing.assert_equal(res, -1)


def test_permutation_operator_standard_swap():
    """Generates the standard swap operator on two qubits."""
    expected_res = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    res = permutation_operator(2, [2, 1])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_permutation_operator_standard_swap_list_dim():
    """Generates the standard swap operator on two qubits."""
    expected_res = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    res = permutation_operator([2, 2], [2, 1])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_permutation_operator_sparse_option():
    """Sparse swap operator on two qutrits."""
    res = permutation_operator(3, [2, 1], False, True)

    np.testing.assert_equal(res[0][0], 1)


def test_permutate_systems_m2_m2():
    """Permute system for dim = [2,1]."""
    test_input_mat = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )

    expected_res = np.array(
        [[1, 3, 2, 4], [9, 11, 10, 12], [5, 7, 6, 8], [13, 15, 14, 16]]
    )

    res = permute_systems(test_input_mat, [2, 1])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def permutate_systems_test_2_3_1():
    """Test permute systems for dim = [2,3,1]."""
    test_input_mat = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16],
            [17, 18, 19, 20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29, 30, 31, 32],
            [33, 34, 35, 36, 37, 38, 39, 40],
            [41, 42, 43, 44, 45, 46, 47, 48],
            [49, 50, 51, 52, 53, 54, 55, 56],
            [57, 58, 59, 60, 61, 62, 63, 64],
        ]
    )

    expected_res = np.array(
        [
            [1, 5, 2, 6, 3, 7, 4, 8],
            [33, 37, 34, 38, 35, 39, 36, 40],
            [9, 13, 10, 14, 11, 15, 12, 16],
            [41, 45, 42, 46, 43, 47, 44, 48],
            [17, 21, 18, 22, 19, 23, 20, 24],
            [49, 53, 50, 54, 51, 55, 52, 56],
            [25, 29, 26, 30, 27, 31, 28, 32],
            [57, 61, 58, 62, 59, 63, 60, 64],
        ]
    )

    res = permute_systems(test_input_mat, [2, 3, 1])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_swap_matrix():
    """Tests swap operation on matrix."""
    test_mat = np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]])

    expected_res = np.array(
        [[1, 9, 5, 13], [3, 11, 7, 15], [2, 10, 6, 14], [4, 12, 8, 16]]
    )

    res = swap(test_mat)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_swap_vector_1():
    """Tests swap operation on vector."""
    test_vec = np.array([1, 2, 3, 4])

    expected_res = np.array([1, 3, 2, 4])

    res = swap(test_vec)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_swap_int_dim():
    """Test swap operation when int is provided."""
    test_mat = np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]])

    expected_res = np.array(
        [[1, 9, 5, 13], [3, 11, 7, 15], [2, 10, 6, 14], [4, 12, 8, 16]]
    )

    res = swap(test_mat, [1, 2], 2)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_swap_invalid_dim():
    """Invalid dim parameters."""
    test_mat = np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]])
    with np.testing.assert_raises(ValueError):
        swap(test_mat, [1, 2], 5)


def test_swap_invalid_sys():
    """Invalid sys parameters."""
    test_mat = np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]])
    with np.testing.assert_raises(ValueError):
        swap(test_mat, [0])


def test_swap_invalid_sys_len():
    """Invalid sys parameters."""
    test_mat = np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]])
    with np.testing.assert_raises(ValueError):
        swap(test_mat, [1], 2)


def test_swap_operator_num():
    """Tests swap operator when argument is number."""
    expected_res = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    res = swap_operator(2)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_swap_operator_vec_dims():
    """Tests swap operator when argument is vector of dims."""
    expected_res = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    res = swap_operator([2, 2])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_symmetric_projection_dim_2_pval_1():
    """Symmetric_projection where the dimension is 2 and p_val is 1."""
    res = symmetric_projection(dim=2, p_val=1).todense()
    expected_res = np.array([[1, 0], [0, 1]])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_symmetric_projection_2():
    """Generates the symmetric_projection where the dimension is 2."""
    res = symmetric_projection(dim=2).todense()
    expected_res = np.array(
        [[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 1]]
    )

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_symmetric_projection_dim_2_partial_true():
    """Symmetric_projection where the dimension is 2 and partial is True."""
    res = symmetric_projection(dim=2, p_val=2, partial=True).todense()
    expected_res = np.array(
        [[0, 0, 1], [-1 / np.sqrt(2), 0, 0], [-1 / np.sqrt(2), 0, 0], [0, 1, 0]]
    )

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_unique_perms_len():
    """Checks the number of unique perms."""
    vec = [1, 1, 2, 2, 1, 2, 1, 3, 3, 3]
    np.testing.assert_equal(len(list(unique_perms(vec))), 4200)


if __name__ == "__main__":
    np.testing.run_module_suite()
