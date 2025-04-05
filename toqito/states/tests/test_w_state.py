"""Test w_state."""

import numpy as np
import pytest

from toqito.matrix_ops import tensor
from toqito.states import basis, w_state


def test_w_state_3():
    """The 3-qubit W-state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    expected_res = 1 / np.sqrt(3) * (tensor(e_1, e_0, e_0) + tensor(e_0, e_1, e_0) + tensor(e_0, e_0, e_1))

    res = w_state(3)
    np.testing.assert_allclose(res, expected_res, atol=0.2)


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
    np.testing.assert_allclose(res, expected_res, atol=0.2)


def test_w_state_with_non_normalized_coeffs():
    """Test W state with non-normalized coefficients gets normalized internally."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    coeffs = [1, 2, 3, 4]  # Not normalized
    norm = np.linalg.norm(coeffs)
    normalized_coeffs = np.array(coeffs) / norm

    expected_res = (
        normalized_coeffs[0] * tensor(e_1, e_0, e_0, e_0)
        + normalized_coeffs[1] * tensor(e_0, e_1, e_0, e_0)
        + normalized_coeffs[2] * tensor(e_0, e_0, e_1, e_0)
        + normalized_coeffs[3] * tensor(e_0, e_0, e_0, e_1)
    )

    res = w_state(4, coeffs)
    np.testing.assert_allclose(res, expected_res, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "idx, coeff",
    [
        # Number of qubits needs to be greater than 2.
        (1, None),
        # Length of coefficient list needs to be equal to number of qubits.
        (4, [1, 2, 3]),
    ],
)
def test_w_state_invalid(idx, coeff):
    """Test function works as expected for an invalid input."""
    with np.testing.assert_raises(ValueError):
        w_state(idx, coeff)
