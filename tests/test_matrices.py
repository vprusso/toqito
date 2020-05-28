"""Test matrices."""
import numpy as np

from scipy.sparse import csr_matrix, issparse

from toqito.matrices import cnot
from toqito.matrices import fourier
from toqito.matrices import gell_mann
from toqito.matrices import gen_gell_mann
from toqito.matrices import gen_pauli
from toqito.matrices import hadamard
from toqito.matrices import iden
from toqito.matrices import pauli


def test_cnot():
    """Test standard CNOT gate."""
    res = cnot()
    expected_res = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_fourier_dim_2():
    """Fourier matrix of dimension 2."""
    expected_res = np.array(
        [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]]
    )

    res = fourier(2)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


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


def test_gell_mann_identity():
    """Generalized Gell-Mann operator identity."""
    expected_res = np.array([[1, 0], [0, 1]])
    res = gen_gell_mann(0, 0, 2)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_pauli_x():
    """Generalized Gell-Mann operator Pauli-X."""
    expected_res = np.array([[0, 1], [1, 0]])
    res = gen_gell_mann(0, 1, 2)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_pauli_y():
    """Generalized Gell-Mann operator Pauli-Y."""
    expected_res = np.array([[0, -1j], [1j, 0]])
    res = gen_gell_mann(1, 0, 2)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_pauli_z():
    """Generalized Gell-Mann operator Pauli-Z."""
    expected_res = np.array([[1, 0], [0, -1]])
    res = gen_gell_mann(1, 1, 2)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_dim_3_1():
    """Generalized Gell-Mann operator 3-dimensional."""
    expected_res = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    res = gen_gell_mann(0, 1, 3)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_dim_3_2():
    """Generalized Gell-Mann operator 3-dimensional."""
    expected_res = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    res = gen_gell_mann(0, 2, 3)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_dim_3_3():
    """Generalized Gell-Mann operator 3-dimensional."""
    expected_res = np.array(
        [[1 / np.sqrt(3), 0, 0], [0, 1 / np.sqrt(3), 0], [0, 0, -2 * 1 / np.sqrt(3)],]
    )
    res = gen_gell_mann(2, 2, 3)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_dim_4_1():
    """Generalized Gell-Mann operator 4-dimensional."""
    expected_res = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    res = gen_gell_mann(2, 3, 4)

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gell_mann_sparse_2():
    """Generalized Gell-Mann operator sparse."""
    res = gen_gell_mann(205, 34, 500, True)

    assert res[34, 205] == -1j
    assert res[205, 34] == 1j


def test_gen_pauli_1_0_2():
    """Generalized Pauli operator for k_1 = 1, k_2 = 0, and dim = 2."""
    dim = 2
    k_1 = 1
    k_2 = 0

    # Pauli-X operator.
    expected_res = np.array([[0, 1], [1, 0]])
    res = gen_pauli(k_1, k_2, dim)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gen_pauli_0_1_2():
    """Generalized Pauli operator for k_1 = 0, k_2 = 1, and dim = 2."""
    dim = 2
    k_1 = 0
    k_2 = 1

    # Pauli-Z operator.
    expected_res = np.array([[1, 0], [0, -1]])
    res = gen_pauli(k_1, k_2, dim)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gen_pauli_1_1_2():
    """Generalized Pauli operator for k_1 = 1, k_2 = 1, and dim = 2."""
    dim = 2
    k_1 = 1
    k_2 = 1

    # Pauli-Y operator.
    expected_res = np.array([[0, -1], [1, 0]])
    res = gen_pauli(k_1, k_2, dim)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_hadamard_0():
    """Test for Hadamard function when n = 0."""
    res = hadamard(0)
    expected_res = np.array([[1]])
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_hadamard_1():
    """Test for Hadamard function when n = 1."""
    res = hadamard(1)
    expected_res = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_hadamard_2():
    """Test for Hadamard function when n = 2."""
    res = hadamard(2)
    expected_res = (
        1 / 2 * np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
    )
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_hadamard_3():
    """Test for Hadamard function when n = 3."""
    res = hadamard(3)
    expected_res = (
        1
        / (2 ** (3 / 2))
        * np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, -1, 1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1],
                [1, 1, 1, 1, -1, -1, -1, -1],
                [1, -1, 1, -1, -1, 1, -1, 1],
                [1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1],
            ]
        )
    )
    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_hadamard_negative():
    """Input must be non-negative."""
    with np.testing.assert_raises(ValueError):
        hadamard(-1)


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
