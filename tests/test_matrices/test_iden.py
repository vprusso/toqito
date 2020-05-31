"""Test iden."""
import numpy as np

from toqito.matrices import iden


def test_iden_full():
    """Full 2-dimensional identity matrix."""
    expected_res = np.array([[1, 0], [0, 1]])
    res = iden(2, False)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_iden_sparse():
    """Sparse 2-dimensional identity matrix."""
    expected_res = np.array([[1, 0], [0, 1]])
    res = iden(2, True).toarray()

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
