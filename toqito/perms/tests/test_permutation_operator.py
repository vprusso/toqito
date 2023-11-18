"""Test permutation_operator."""
import numpy as np

from toqito.perms import permutation_operator


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

    expected_res = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_permutation_operator_dim_3_perm_1_2():
    """Test permutation operator when dim is 3 and perm is [1, 2]."""
    res = permutation_operator(3, [1, 2])
    expected_res = np.identity(9)
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_permutation_operator_dim_2_perm_1_3_2():
    """Test permutation operator when dim is 2 and perm is [1, 3, 2]."""
    res = permutation_operator(2, [1, 3, 2])
    expected_res = np.array(
        [
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        ]
    )
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_permutation_operator_dim_2_2_perm_1_2():
    """Test permutation operator when dim is [2, 2] and perm is [1, 2]"""
    res = permutation_operator([2, 2], [1, 2])
    expected_res = np.identity(4)
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_permutation_operator_dim_2_2_perm_2_1():
    """Test permutation operator when dim is [2, 2] and perm is [2, 1]"""
    res = permutation_operator([2, 2], [2, 1])
    expected_res = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
