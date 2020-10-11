"""Test permute_systems."""
import numpy as np

from toqito.perms import permute_systems


def test_permute_systems_vec():
    """Permute system for perm = [2,1] and vector [1, 2, 3, 4]."""
    test_input_mat = np.array([1, 2, 3, 4])
    expected_res = np.array([1, 3, 2, 4])

    res = permute_systems(test_input_mat, [2, 1])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_permute_systems_m2_m2():
    """Permute system for perm = [2,1] and dim = [2, 2]."""
    test_input_mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

    expected_res = np.array([[1, 3, 2, 4], [9, 11, 10, 12], [5, 7, 6, 8], [13, 15, 14, 16]])

    res = permute_systems(test_input_mat, [2, 1], [2, 2])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_permute_systems_m2_m2_np_array():
    """Permute system for perm = np.array([2,1])."""
    test_input_mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

    expected_res = np.array([[1, 3, 2, 4], [9, 11, 10, 12], [5, 7, 6, 8], [13, 15, 14, 16]])

    res = permute_systems(test_input_mat, np.array([2, 1]))

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def permute_systems_test_2_3_1():
    """Test permute systems for perm = [2,3,1]."""
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


def test_permute_systems_invalid_perm_vector():
    """Invalid input for permute systems."""
    with np.testing.assert_raises(ValueError):
        test_input_mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        permute_systems(test_input_mat, [0, 1])


if __name__ == "__main__":
    np.testing.run_module_suite()
