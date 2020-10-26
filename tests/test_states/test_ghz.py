"""Test ghz."""
import numpy as np

from toqito.states import basis
from toqito.states import ghz
from toqito.matrix_ops import tensor


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
        ghz(0, 2)


def test_ghz_invalid_qubits():
    """Tests for invalid number of qubits."""
    with np.testing.assert_raises(ValueError):
        ghz(2, 0)


def test_ghz_invalid_coeff():
    """Tests for invalid coefficients."""
    with np.testing.assert_raises(ValueError):
        ghz(2, 3, [1, 2, 3, 4, 5])


if __name__ == "__main__":
    np.testing.run_module_suite()
