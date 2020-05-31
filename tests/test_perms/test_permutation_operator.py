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

    np.testing.assert_equal(res[0][0], 1)


if __name__ == "__main__":
    np.testing.run_module_suite()
