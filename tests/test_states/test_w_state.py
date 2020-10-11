"""Test w_state."""
import numpy as np

from toqito.states import basis
from toqito.states import w_state
from toqito.matrix_ops import tensor


def test_w_state_3():
    """The 3-qubit W-state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = (
        1 / np.sqrt(3) * (tensor(e_1, e_0, e_0) + tensor(e_0, e_1, e_0) + tensor(e_0, e_0, e_1))
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


if __name__ == "__main__":
    np.testing.run_module_suite()
