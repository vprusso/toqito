"""Test swap_operator."""
import numpy as np

from toqito.perms import swap_operator


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


if __name__ == "__main__":
    np.testing.run_module_suite()
