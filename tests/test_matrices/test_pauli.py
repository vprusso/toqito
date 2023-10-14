"""Test pauli."""
import numpy as np

from scipy.sparse import issparse

from toqito.matrices import pauli


def test_pauli_str_sparse():
    """Pauli-I operator with argument "I"."""
    res = pauli("I", True)
    np.testing.assert_equal(issparse(res), True)


def test_pauli_int_sparse():
    """Pauli-I operator with argument "I"."""
    res = pauli(0, True)
    np.testing.assert_equal(issparse(res), True)


def test_pauli_i():
    """Pauli-I operator with argument "I"."""
    expected_res = np.array([[1, 0], [0, 1]])
    res = pauli("I")

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_pauli_x():
    """Pauli-X operator with argument "X"."""
    expected_res = np.array([[0, 1], [1, 0]])
    res = pauli("X")

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_pauli_1():
    """Pauli-X operator with argument 1."""
    expected_res = np.array([[0, 1], [1, 0]])
    res = pauli(1)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_pauli_y():
    """Pauli-Y operator with argument "Y"."""
    expected_res = np.array([[0, -1j], [1j, 0]])
    res = pauli("Y")

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_pauli_2():
    """Pauli-Y operator with argument 2."""
    expected_res = np.array([[0, -1j], [1j, 0]])
    res = pauli(2)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_pauli_z():
    """Pauli-Z operator with argument "Z"."""
    expected_res = np.array([[1, 0], [0, -1]])
    res = pauli("Z")

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_pauli_3():
    """Pauli-Z operator with argument 3."""
    expected_res = np.array([[1, 0], [0, -1]])
    res = pauli(3)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_pauli_int_list():
    """Test with list of Paulis of ints."""
    expected_res = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    res = pauli([1, 1])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_pauli_str_list():
    """Test with list of Paulis of str."""
    expected_res = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    res = pauli(["x", "x"])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
