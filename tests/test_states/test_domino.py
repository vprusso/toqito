"""Test domino."""
import numpy as np

from toqito.states import basis
from toqito.states import domino


def test_domino_0():
    """Domino with index = 0."""
    e_1 = basis(3, 1)
    expected_res = np.kron(e_1, e_1)
    res = domino(0)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_domino_1():
    """Domino with index = 1."""
    e_0, e_1 = basis(3, 0), basis(3, 1)
    expected_res = np.kron(e_0, 1 / np.sqrt(2) * (e_0 + e_1))
    res = domino(1)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_domino_2():
    """Domino with index = 2."""
    e_0, e_1 = basis(3, 0), basis(3, 1)
    expected_res = np.kron(e_0, 1 / np.sqrt(2) * (e_0 - e_1))
    res = domino(2)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_domino_3():
    """Domino with index = 3."""
    e_1, e_2 = basis(3, 1), basis(3, 2)
    expected_res = np.kron(e_2, 1 / np.sqrt(2) * (e_1 + e_2))
    res = domino(3)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_domino_4():
    """Domino with index = 4."""
    e_1, e_2 = basis(3, 1), basis(3, 2)
    expected_res = np.kron(e_2, 1 / np.sqrt(2) * (e_1 - e_2))
    res = domino(4)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_domino_5():
    """Domino with index = 5."""
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    expected_res = np.kron(1 / np.sqrt(2) * (e_1 + e_2), e_0)
    res = domino(5)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_domino_6():
    """Domino with index = 6."""
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    expected_res = np.kron(1 / np.sqrt(2) * (e_1 - e_2), e_0)
    res = domino(6)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_domino_7():
    """Domino with index = 7."""
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    expected_res = np.kron(1 / np.sqrt(2) * (e_0 + e_1), e_2)
    res = domino(7)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_domino_8():
    """Domino with index = 8."""
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    expected_res = np.kron(1 / np.sqrt(2) * (e_0 - e_1), e_2)
    res = domino(8)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_domino_invalid_index():
    """Tests for invalid index input."""
    with np.testing.assert_raises(ValueError):
        domino(9)


if __name__ == "__main__":
    np.testing.run_module_suite()
