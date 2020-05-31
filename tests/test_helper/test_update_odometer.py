"""Test update_odometer."""
import numpy as np

from toqito.helper import update_odometer


def test_update_odometer_0_0():
    """Update odometer from [2, 2] to [0, 0]."""
    vec = np.array([2, 2])
    upper_lim = np.array([3, 2])
    res = update_odometer(vec, upper_lim)

    bool_mat = np.isclose([0, 0], res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_update_odometer_0_1():
    """Update odometer from [0, 0] to [0, 1]."""
    vec = np.array([0, 0])
    upper_lim = np.array([3, 2])
    res = update_odometer(vec, upper_lim)

    bool_mat = np.isclose([0, 1], res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_update_odometer_1_0():
    """Update odometer from [0, 1] to [1, 0]."""
    vec = np.array([0, 1])
    upper_lim = np.array([3, 2])
    res = update_odometer(vec, upper_lim)

    bool_mat = np.isclose([1, 0], res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_update_odometer_2_0():
    """Update odometer from [1, 1] to [2, 0]."""
    vec = np.array([1, 1])
    upper_lim = np.array([3, 2])
    res = update_odometer(vec, upper_lim)

    bool_mat = np.isclose([2, 0], res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_update_odometer_2_1():
    """Update odometer from [2, 0] to [2, 1]."""
    vec = np.array([2, 0])
    upper_lim = np.array([3, 2])
    res = update_odometer(vec, upper_lim)

    bool_mat = np.isclose([2, 1], res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_update_odometer_2_2():
    """Update odometer from [2, 1] to [0, 0]."""
    vec = np.array([2, 1])
    upper_lim = np.array([3, 2])
    res = update_odometer(vec, upper_lim)

    bool_mat = np.isclose([0, 0], res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_update_odometer_empty():
    """Return `None` if empty lists are provided."""
    vec = np.array([])
    upper_lim = np.array([])
    res = update_odometer(vec, upper_lim)

    bool_mat = np.isclose([], res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
