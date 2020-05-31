"""Test basis."""
import numpy as np

from toqito.states import basis


def test_basis_ket_0():
    """Test for `|0>`."""
    expected_res = np.array([[1], [0]])
    res = basis(2, 0)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_basis_ket_1():
    """Test for `|1>`."""
    expected_res = np.array([[0], [1]])
    res = basis(2, 1)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_basis_ket_0000():
    """Test for `|0000>`."""
    expected_res = np.array([[1], [0], [0], [0]])
    res = basis(4, 0)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_basis_invalid_dim():
    """Tests for invalid dimension inputs."""
    with np.testing.assert_raises(ValueError):
        basis(4, 4)


if __name__ == "__main__":
    np.testing.run_module_suite()
