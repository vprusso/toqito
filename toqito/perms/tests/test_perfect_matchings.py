"""Test perfect_matchings."""
import numpy as np

from toqito.perms import perfect_matchings


def test_perfect_matchings_num_4():
    """All perfect matchings of size 4."""
    res = perfect_matchings(4)
    expected_res = np.array([[1, 2, 3, 4], [1, 3, 2, 4], [1, 4, 3, 2]])
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_perfect_matchings_num_4_list():
    """All perfect matchings of size 4 with input as list."""
    res = perfect_matchings([1, 2, 3, 4])
    expected_res = np.array([[1, 2, 3, 4], [1, 3, 2, 4], [1, 4, 3, 2]])
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_perfect_matchings_odd():
    """There are no perfect matchings of an odd number of objects."""
    res = perfect_matchings(5)
    expected_res = np.zeros((0, 5))
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_perfect_matchings_num_6():
    """All perfect matchings of size 6."""
    res = perfect_matchings(6)

    expected_res = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 5, 4, 6],
            [1, 2, 3, 6, 5, 4],
            [1, 3, 2, 4, 5, 6],
            [1, 3, 2, 5, 4, 6],
            [1, 3, 2, 6, 5, 4],
            [1, 4, 3, 2, 5, 6],
            [1, 4, 3, 5, 2, 6],
            [1, 4, 3, 6, 5, 2],
            [1, 5, 3, 4, 2, 6],
            [1, 5, 3, 2, 4, 6],
            [1, 5, 3, 6, 2, 4],
            [1, 6, 3, 4, 5, 2],
            [1, 6, 3, 5, 4, 2],
            [1, 6, 3, 2, 5, 4],
        ]
    )
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
