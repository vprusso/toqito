"""Test swap."""
import numpy as np

from toqito.perms import swap


def test_swap_matrix():
    """Tests swap operation on matrix."""
    test_mat = np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]])

    expected_res = np.array([[1, 9, 5, 13], [3, 11, 7, 15], [2, 10, 6, 14], [4, 12, 8, 16]])

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

    expected_res = np.array([[1, 9, 5, 13], [3, 11, 7, 15], [2, 10, 6, 14], [4, 12, 8, 16]])

    res = swap(test_mat, [1, 2], 2)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_swap_8_by_8():
    """Test swap on an 8x8 matrix."""
    test_mat = np.arange(64).reshape(8, 8).T

    expected_res = np.array(
        [
            [0, 16, 8, 24, 32, 48, 40, 56],
            [2, 18, 10, 26, 34, 50, 42, 58],
            [1, 17, 9, 25, 33, 49, 41, 57],
            [3, 19, 11, 27, 35, 51, 43, 59],
            [4, 20, 12, 28, 36, 52, 44, 60],
            [6, 22, 14, 30, 38, 54, 46, 62],
            [5, 21, 13, 29, 37, 53, 45, 61],
            [7, 23, 15, 31, 39, 55, 47, 63],
        ]
    )
    res = swap(test_mat, [2, 3], [2, 2, 2])

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


if __name__ == "__main__":
    np.testing.run_module_suite()
