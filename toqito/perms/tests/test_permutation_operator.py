"""Test permutation_operator."""

import numpy as np

from toqito.perms import permutation_operator


def test_permutation_operator_standard_swap():
    """Generates the standard swap operator on two qubits with zero-based indexing."""
    expected_res = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    res = permutation_operator(2, [1, 0])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_permutation_operator_standard_swap_list_dim():
    """Generates the standard swap operator on two qubits with zero-based indexing."""
    expected_res = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    res = permutation_operator([2, 2], [1, 0])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_permutation_operator_sparse_option():
    """Sparse swap operator on two qutrits with zero-based indexing."""
    res = permutation_operator(3, [1, 0], False, True)

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


def test_permutation_operator_dim_3_perm_0_1():
    """Test permutation operator when dim is 3 and perm is [0, 1] using zero-based indexing."""
    res = permutation_operator(3, [0, 1])
    expected_res = np.identity(9)
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_permutation_operator_dim_2_perm_0_2_1():
    """Test permutation operator when dim is 2 and perm is [0, 2, 1] using zero-based indexing."""
    res = permutation_operator(2, [0, 2, 1])
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


def test_permutation_operator_dim_2_2_perm_0_1():
    """Test permutation operator when dim is [2, 2] and perm is [0, 1] using zero-based indexing."""
    res = permutation_operator([2, 2], [0, 1])
    expected_res = np.identity(4)
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_permutation_operator_dim_2_2_perm_1_0():
    """Test permutation operator when dim is [2, 2] and perm is [1, 0] using zero-based indexing."""
    res = permutation_operator([2, 2], [1, 0])
    expected_res = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)
