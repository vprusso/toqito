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
    np.testing.assert_allclose(res, expected_res, atol=1e-12)
    # The returned state is normalized (the previous np.around(., 4) broke this).
    np.testing.assert_allclose(np.linalg.norm(res), 1.0, atol=1e-12)


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
    np.testing.assert_allclose(res, expected_res, atol=1e-12)


def test_w_state_complex_coefficients():
    """Complex coefficients are preserved (not truncated to real)."""
    coeffs = np.array([1, 1j, 1, 1j]) / 2
    res = w_state(4, coeffs)

    assert np.iscomplexobj(res)
    np.testing.assert_allclose(np.linalg.norm(res), 1.0, atol=1e-12)
    # Excitation on the last qubit sits at index 1 and carries coeff for that qubit.
    np.testing.assert_allclose(res[1, 0], coeffs[3])


def test_w_state_column_coefficients():
    """A column-shaped coefficient vector behaves the same as a flat one."""
    flat = w_state(3, np.ones(3) / np.sqrt(3))
    column = w_state(3, np.ones((3, 1)) / np.sqrt(3))
    assert np.array_equal(column, flat)


def test_w_state_exact_values_and_dtype():
    """The state is filled exactly (no rounding) and the coefficient dtype is preserved."""
    # Coefficients [0, 3, 4] normalize to exactly [0.0, 0.6, 0.8] in float64.
    res = w_state(3, [0, 3, 4])
    expected = np.zeros((8, 1))
    expected[1, 0] = 0.8  # excitation on qubit 2 (index 2**0) carries coeff[2].
    expected[2, 0] = 0.6  # excitation on qubit 1 (index 2**1) carries coeff[1].
    assert np.array_equal(res, expected)

    # A unit-norm integer coefficient vector is not renormalized, so its dtype survives.
    res = w_state(2, [1, 0])
    assert res.dtype == np.array([1, 0]).dtype
    assert np.array_equal(res, np.array([[0], [0], [1], [0]]))


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
