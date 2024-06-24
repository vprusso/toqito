"""Test ghz."""

import numpy as np
import pytest

from toqito.matrix_ops import tensor
from toqito.states import ghz

e_0, e_1 = np.array([[1], [0]]), np.array([[0], [1]])
ghz_2_3 = 1 / np.sqrt(2) * (tensor(e_0, e_0, e_0) + tensor(e_1, e_1, e_1))


@pytest.mark.parametrize(
    "dim, num_qubits, coeff, expected_res",
    [
        # Produces the 3-qubit GHZ state: `1/sqrt(2) * (|000> + |111>)`.
        (2, 3, None, ghz_2_3),
    ],
)
def test_ghz(dim, num_qubits, coeff, expected_res):
    """Test function works as expected for a valid input."""
    res = ghz(dim, num_qubits, coeff)
    np.testing.assert_allclose(res, expected_res)


def test_ghz_4_7():
    r"""The following generates the following GHZ state in `(C^4)^{\otimes 7}`.

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

    res = ghz(4, 7, np.array([1, 2, 3, 4]) / np.sqrt(30))
    np.testing.assert_allclose(res, expected_res)


@pytest.mark.parametrize(
    "dim, num_qubits, coeff",
    [
        # Invalid dimensions.
        (0, 2, None),
        # Invalid qubits.
        (2, 0, None),
        # Invalid coefficients.
        (2, 3, [1, 2, 3, 4, 5]),
    ],
)
def test_ghz_invalid_input(dim, num_qubits, coeff):
    """Tests for invalid dimensions."""
    with np.testing.assert_raises(ValueError):
        ghz(dim, num_qubits, coeff)
