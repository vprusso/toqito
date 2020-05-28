"""Tests for states."""
import numpy as np

from toqito.states import basis
from toqito.states import bell
from toqito.states import chessboard
from toqito.states import domino
from toqito.states import gen_bell
from toqito.states import ghz
from toqito.states import gisin
from toqito.states import horodecki
from toqito.states import isotropic
from toqito.states import max_entangled
from toqito.states import max_mixed
from toqito.states import tile
from toqito.states import w_state
from toqito.states import werner

from toqito.matrix_ops import tensor


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


def test_bell_0():
    """Generate the Bell state: `1/sqrt(2) * (|00> + |11>)`."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = 1 / np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))

    res = bell(0)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_bell_1():
    """Generates the Bell state: `1/sqrt(2) * (|00> - |11>)`."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = 1 / np.sqrt(2) * (np.kron(e_0, e_0) - np.kron(e_1, e_1))

    res = bell(1)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_bell_2():
    """Generates the Bell state: `1/sqrt(2) * (|01> + |10>)`."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = 1 / np.sqrt(2) * (np.kron(e_0, e_1) + np.kron(e_1, e_0))

    res = bell(2)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_bell_3():
    """Generates the Bell state: `1/sqrt(2) * (|01> - |10>)`."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = 1 / np.sqrt(2) * (np.kron(e_0, e_1) - np.kron(e_1, e_0))

    res = bell(3)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_bell_invalid():
    """Ensures that an integer above 3 is error-checked."""
    with np.testing.assert_raises(ValueError):
        bell(4)


def test_chessboard():
    """The chessboard_state."""
    res = chessboard([1, 2, 3, 4, 5, 6], 7, 8)
    np.testing.assert_equal(np.isclose(res[0][0], 0.22592592592592592), True)


def test_chessboard_default_s():
    """The chessboard_state with default `s_param`."""
    res = chessboard([1, 2, 3, 4, 5, 6], 7)
    np.testing.assert_equal(np.isclose(res[0][0], 0.29519938056523426), True)


def test_chessboard_default_s_t():
    """The chessboard_state with default `s_param` and `t_param`."""
    res = chessboard([1, 2, 3, 4, 5, 6])
    np.testing.assert_equal(np.isclose(res[0][0], 0.3863449236810438), True)


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


def test_gen_bell_0_0_2():
    """Generalized Bell state for k_1 = k_2 = 0 and dim = 2."""
    dim = 2
    k_1 = 0
    k_2 = 0

    expected_res = bell(0) * bell(0).conj().T

    res = gen_bell(k_1, k_2, dim)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gen_bell_0_1_2():
    """Generalized Bell state for k_1 = 0, k_2 = 1 and dim = 2."""
    dim = 2
    k_1 = 0
    k_2 = 1

    expected_res = bell(1) * bell(1).conj().T

    res = gen_bell(k_1, k_2, dim)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gen_bell_1_0_2():
    """Generalized Bell state for k_1 = 1, k_2 = 0 and dim = 2."""
    dim = 2
    k_1 = 1
    k_2 = 0

    expected_res = bell(2) * bell(2).conj().T

    res = gen_bell(k_1, k_2, dim)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gen_bell_1_1_2():
    """Generalized Bell state for k_1 = 1, k_2 = 1 and dim = 2."""
    dim = 2
    k_1 = 1
    k_2 = 1

    expected_res = bell(3) * bell(3).conj().T

    res = gen_bell(k_1, k_2, dim)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_ghz_2_3():
    """Produces the 3-qubit GHZ state: `1/sqrt(2) * (|000> + |111>)`."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = 1 / np.sqrt(2) * (tensor(e_0, e_0, e_0) + tensor(e_1, e_1, e_1))

    res = ghz(2, 3).toarray()

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_ghz_4_7():
    r"""
    The following generates the following GHZ state in `(C^4)^{\otimes 7}`.

    `1/sqrt(30) * (|0000000> + 2|1111111> + 3|2222222> + 4|3333333>)`.
    """
    e0_4 = np.array([[1], [0], [0], [0]])
    e1_4 = np.array([[0], [1], [0], [0]])
    e2_4 = np.array([[0], [0], [1], [0]])
    e3_4 = np.array([[0], [0], [0], [1]])

    expected_res = (
        1
        / np.sqrt(30)
        * (
            tensor(e0_4, e0_4, e0_4, e0_4, e0_4, e0_4, e0_4)
            + 2 * tensor(e1_4, e1_4, e1_4, e1_4, e1_4, e1_4, e1_4)
            + 3 * tensor(e2_4, e2_4, e2_4, e2_4, e2_4, e2_4, e2_4)
            + 4 * tensor(e3_4, e3_4, e3_4, e3_4, e3_4, e3_4, e3_4)
        )
    )

    res = ghz(4, 7, np.array([1, 2, 3, 4]) / np.sqrt(30)).toarray()

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_ghz_invalid_dim():
    """Tests for invalid dimensions."""
    with np.testing.assert_raises(ValueError):
        ghz(1, 2)


def test_ghz_invalid_qubits():
    """Tests for invalid number of qubits."""
    with np.testing.assert_raises(ValueError):
        ghz(2, 1)


def test_ghz_invalid_coeff():
    """Tests for invalid coefficients."""
    with np.testing.assert_raises(ValueError):
        ghz(2, 3, [1, 2, 3, 4, 5])


def test_gisin_valid():
    """Standard Gisin state."""
    expected_res = np.array(
        [
            [1 / 4, 0, 0, 0],
            [0, 0.35403671, -0.22732436, 0],
            [0, -0.22732436, 0.14596329, 0],
            [0, 0, 0, 1 / 4],
        ]
    )

    res = gisin(0.5, 1)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_gisin_invalid():
    """Invalid Gisin state parameters."""
    with np.testing.assert_raises(ValueError):
        gisin(5, 1)


def test_horodecki_state_3_3_default():
    """The 3-by-3 Horodecki state (no dimensions specified on input)."""
    expected_res = np.array(
        [
            [0.1000, 0, 0, 0, 0.1000, 0, 0, 0, 0.1000],
            [0, 0.1000, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.1000, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.1000, 0, 0, 0, 0, 0],
            [0.1000, 0, 0, 0, 0.1000, 0, 0, 0, 0.1000],
            [0, 0, 0, 0, 0, 0.1000, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.1500, 0, 0.0866],
            [0, 0, 0, 0, 0, 0, 0, 0.1000, 0],
            [0.1000, 0, 0, 0, 0.1000, 0, 0.0866, 0, 0.1500],
        ]
    )

    res = horodecki(0.5)
    bool_mat = np.isclose(expected_res, res, atol=0.0001)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_horodecki_state_3_3():
    """The 3-by-3 Horodecki state."""
    expected_res = np.array(
        [
            [0.1000, 0, 0, 0, 0.1000, 0, 0, 0, 0.1000],
            [0, 0.1000, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.1000, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.1000, 0, 0, 0, 0, 0],
            [0.1000, 0, 0, 0, 0.1000, 0, 0, 0, 0.1000],
            [0, 0, 0, 0, 0, 0.1000, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.1500, 0, 0.0866],
            [0, 0, 0, 0, 0, 0, 0, 0.1000, 0],
            [0.1000, 0, 0, 0, 0.1000, 0, 0.0866, 0, 0.1500],
        ]
    )

    res = horodecki(0.5, [3, 3])
    bool_mat = np.isclose(expected_res, res, atol=0.0001)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_horodecki_state_2_4():
    """The 2-by-4 Horodecki state."""
    expected_res = np.array(
        [
            [0.1111, 0, 0, 0, 0, 0.1111, 0, 0],
            [0, 0.1111, 0, 0, 0, 0, 0.1111, 0],
            [0, 0, 0.1111, 0, 0, 0, 0, 0.1111],
            [0, 0, 0, 0.1111, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.1667, 0, 0.0962],
            [0.1111, 0, 0, 0, 0, 0.1111, 0, 0],
            [0, 0.1111, 0, 0, 0, 0, 0.1111, 0],
            [0, 0, 0.1111, 0, 0, 0.0962, 0, 0.1667],
        ]
    )

    res = horodecki(0.5, [2, 4])
    bool_mat = np.isclose(expected_res, res, atol=0.2)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_horodecki_invalid_a_param():
    """Tests for invalid a_param inputs."""
    with np.testing.assert_raises(ValueError):
        horodecki(-5)
    with np.testing.assert_raises(ValueError):
        horodecki(5)


def test_horodecki_invalid_dim():
    """Tests for invalid dimension inputs."""
    with np.testing.assert_raises(ValueError):
        horodecki(0.5, [3, 4])


def test_isotropic_qutrit():
    """Generate a qutrit isotropic state with `alpha` = 1/2."""
    res = isotropic(3, 1 / 2)

    np.testing.assert_equal(np.isclose(res[0, 0], 2 / 9), True)
    np.testing.assert_equal(np.isclose(res[4, 4], 2 / 9), True)
    np.testing.assert_equal(np.isclose(res[8, 8], 2 / 9), True)


def test_max_ent_2():
    """Generate maximally entangled state: `1/sqrt(2) * (|00> + |11>)`."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = 1 / np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
    res = max_entangled(2)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_max_ent_2_0_0():
    """Generate maximally entangled state: `|00> + |11>`."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = 1 * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
    res = max_entangled(2, False, False)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


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


def test_tile_0():
    r"""Generate the Tile state for index = 0:
    .. math::
        |\psi_0 \rangle = \frac{1}{\sqrt{2}} |0 \rangle
        \left(|0\rangle - |1\rangle \right).
    """
    e_0, e_1 = basis(3, 0), basis(3, 1)
    expected_res = 1 / np.sqrt(2) * e_0 * (e_0 - e_1)
    res = tile(0)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tile_1():
    r"""Generate the Tile state for index = 1:
    .. math::
        |\psi_1\rangle = \frac{1}{\sqrt{2}}
        \left(|0\rangle - |1\rangle \right) |2\rangle
    """
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    expected_res = 1 / np.sqrt(2) * (e_0 - e_1) * e_2
    res = tile(1)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tile_2():
    r"""Generate the Tile state for index = 2:
    .. math::
        |\psi_2\rangle = \frac{1}{\sqrt{2}} |2\rangle
        \left(|1\rangle - |2\rangle \right)
    """
    e_1, e_2 = basis(3, 1), basis(3, 2)
    expected_res = 1 / np.sqrt(2) * e_2 * (e_1 - e_2)
    res = tile(2)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tile_3():
    r"""Generate the Tile state for index = 3:
    .. math::
        |\psi_3\rangle = \frac{1}{\sqrt{2}}
        \left(|1\rangle - |2\rangle \right) |0\rangle
    """
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    expected_res = 1 / np.sqrt(2) * (e_1 - e_2) * e_0
    res = tile(3)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tile_4():
    r"""Generate the Tile state for index = 4:
    .. math::
        |\psi_4\rangle = \frac{1}{3}
        \left(|0\rangle + |1\rangle + |2\rangle)\right)
        \left(|0\rangle + |1\rangle + |2\rangle.
    """
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    expected_res = 1 / 3 * (e_0 + e_1 + e_2) * (e_0 + e_1 + e_2)
    res = tile(4)

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_tile_invalid():
    """Ensures that an integer above 4 is error-checked."""
    with np.testing.assert_raises(ValueError):
        tile(5)


def test_w_state_3():
    """The 3-qubit W-state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = (
        1
        / np.sqrt(3)
        * (tensor(e_1, e_0, e_0) + tensor(e_0, e_1, e_0) + tensor(e_0, e_0, e_1))
    )

    res = w_state(3)

    bool_mat = np.isclose(res, expected_res, atol=0.2)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_w_state_generalized():
    """Generalized 4-qubit W-state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = (
        1
        / np.sqrt(30)
        * (
            tensor(e_1, e_0, e_0, e_0)
            + 2 * tensor(e_0, e_1, e_0, e_0)
            + 3 * tensor(e_0, e_0, e_1, e_0)
            + 4 * tensor(e_0, e_0, e_0, e_1)
        )
    )

    coeffs = np.array([1, 2, 3, 4]) / np.sqrt(30)
    res = w_state(4, coeffs)

    bool_mat = np.isclose(res, expected_res, atol=0.2)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_w_state_invalid_num_qubits():
    """Number of qubits needs to be greater than 2."""
    with np.testing.assert_raises(ValueError):
        w_state(1)


def test_w_state_invalid_coeff_list():
    """Length of coeff list needs to be equal to number of qubits."""
    with np.testing.assert_raises(ValueError):
        w_state(4, [1, 2, 3])


def test_werner_qutrit():
    """Test for qutrit Werner state."""
    res = werner(3, 1 / 2)
    np.testing.assert_equal(np.isclose(res[0][0], 0.0666666), True)
    np.testing.assert_equal(np.isclose(res[1][3], -0.066666), True)


def test_werner_multipartite():
    """Test for multipartite Werner state."""
    res = werner(2, [0.01, 0.02, 0.03, 0.04, 0.05])
    np.testing.assert_equal(np.isclose(res[0][0], 0.1127, atol=1e-02), True)


def test_werner_invalid_alpha():
    """Test for invalid `alpha` parameter."""
    with np.testing.assert_raises(ValueError):
        werner(3, [1, 2])


if __name__ == "__main__":
    np.testing.run_module_suite()
