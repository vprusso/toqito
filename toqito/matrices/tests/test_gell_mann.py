"""Test gell_mann."""
import numpy as np

from scipy.sparse import csr_matrix

from toqito.matrices import gell_mann


def test_gell_mann_idx_0():
    """Gell-Mann operator for index = 0."""
    expected_res = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    res = gell_mann(0)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_idx_1():
    """Gell-Mann operator for index = 1."""
    expected_res = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    res = gell_mann(1)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_idx_2():
    """Gell-Mann operator for index = 2."""
    expected_res = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
    res = gell_mann(2)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_idx_3():
    """Gell-Mann operator for index = 3."""
    expected_res = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    res = gell_mann(3)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_idx_4():
    """Gell-Mann operator for index = 4."""
    expected_res = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])

    res = gell_mann(4)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_idx_5():
    """Gell-Mann operator for index = 5."""
    expected_res = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])

    res = gell_mann(5)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_idx_6():
    """Gell-Mann operator for index = 6."""
    expected_res = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])

    res = gell_mann(6)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_idx_7():
    """Gell-Mann operator for index = 7."""
    expected_res = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])

    res = gell_mann(7)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_idx_8():
    """Gell-Mann operator for index = 8."""
    expected_res = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / np.sqrt(3)
    res = gell_mann(8)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_invalid_idx():
    """Invalid Gell-Mann parameters."""
    with np.testing.assert_raises(ValueError):
        gell_mann(9)


def test_gell_mann_sparse():
    """Test sparse Gell-Mann matrix."""
    res = gell_mann(3, is_sparse=True)
    np.testing.assert_equal(isinstance(res, csr_matrix), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
