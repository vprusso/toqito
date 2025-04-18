"""Test perfect_matchings."""

import numpy as np

from toqito.perms import perfect_matchings


def test_perfect_matchings_num_4():
    """All perfect matchings of size 4."""
    res = perfect_matchings(4)
    expected_res = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 2, 1]])
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_perfect_matchings_num_4_list():
    """All perfect matchings of size 4 with input as list."""
    res = perfect_matchings([0, 1, 2, 3])
    expected_res = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 2, 1]])
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
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 4, 3, 5],
            [0, 1, 2, 5, 4, 3],
            [0, 2, 1, 3, 4, 5],
            [0, 2, 1, 4, 3, 5],
            [0, 2, 1, 5, 4, 3],
            [0, 3, 2, 1, 4, 5],
            [0, 3, 2, 4, 1, 5],
            [0, 3, 2, 5, 4, 1],
            [0, 4, 2, 3, 1, 5],
            [0, 4, 2, 1, 3, 5],
            [0, 4, 2, 5, 1, 3],
            [0, 5, 2, 3, 4, 1],
            [0, 5, 2, 4, 3, 1],
            [0, 5, 2, 1, 4, 3],
        ]
    )
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)
