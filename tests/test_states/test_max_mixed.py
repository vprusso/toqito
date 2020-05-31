"""Test max_mixed."""
import numpy as np

from toqito.states import max_mixed


def test_max_mixed_dim_2_full():
    """Generate full 2-dimensional maximally mixed state."""
    expected_res = 1 / 2 * np.array([[1, 0], [0, 1]])
    res = max_mixed(2, is_sparse=False)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_max_mixed_dim_2_sparse():
    """Generate sparse 2-dimensional maximally mixed state."""
    expected_res = 1 / 2 * np.array([[1, 0], [0, 1]])
    res = max_mixed(2, is_sparse=True)

    bool_mat = np.isclose(res.toarray(), expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
