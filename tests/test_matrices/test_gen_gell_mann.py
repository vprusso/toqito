"""Test gen_gell_mann."""
import numpy as np

from toqito.matrices import gen_gell_mann


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
        [
            [1 / np.sqrt(3), 0, 0],
            [0, 1 / np.sqrt(3), 0],
            [0, 0, -2 * 1 / np.sqrt(3)],
        ]
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


if __name__ == "__main__":
    np.testing.run_module_suite()
