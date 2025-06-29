"""Tests for phase damping channel."""

import re

import numpy as np
import pytest

from toqito.channels import phase_damping


@pytest.mark.parametrize("gamma", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_kraus_operators(gamma):
    """Test if the function returns correct Kraus operators for given gamma."""
    kraus_ops = phase_damping(None, gamma=gamma)

    k0_expected = np.diag([1, np.sqrt(1 - gamma)])
    k1_expected = np.diag([0, np.sqrt(gamma)])
    expected_kraus_ops = [k0_expected, k1_expected]

    for i in range(2):
        np.testing.assert_almost_equal(kraus_ops[i], expected_kraus_ops[i])

    completeness_sum = np.zeros((2, 2), dtype=complex)
    for k in kraus_ops:
        completeness_sum += k.conj().T @ k
    np.testing.assert_almost_equal(completeness_sum, np.eye(2))


@pytest.mark.parametrize(
    "rho, gamma",
    [
        # Ground state |0⟩⟨0|.
        (np.array([[1, 0], [0, 0]]), 0.3),
        # Exfootcited state |1⟩⟨1|.
        (np.array([[0, 0], [0, 1]]), 0.4),
        # Superposition state (|0⟩+|1⟩)(⟨0|+⟨1|)/2.
        (np.array([[0.5, 0.5], [0.5, 0.5]]), 0.5),
        # Mixed state.
        (np.array([[0.7, 0.2j], [-0.2j, 0.3]]), 0.6),
        # Complex mixed state.
        (np.array([[0.7, 0.2 + 0.1j], [0.2 - 0.1j, 0.3]], dtype=complex), 0.7),
    ],
)
def test_apply_to_states(rho, gamma):
    """Apply the channel to various input states rho."""
    kraus_ops = phase_damping(None, gamma=gamma)

    expected_output = np.zeros((2, 2), dtype=complex)
    for k in kraus_ops:
        expected_output += k @ rho @ k.conj().T

    result = phase_damping(rho, gamma=gamma)

    np.testing.assert_almost_equal(result, expected_output)
    np.testing.assert_almost_equal(np.trace(result), np.trace(rho))
    np.testing.assert_almost_equal(result, result.conj().T)


@pytest.mark.parametrize(
    "gamma, error_message",
    [
        (-0.1, "Gamma must be between 0 and 1."),
        (1.1, "Gamma must be between 0 and 1."),
    ],
)
def test_invalid_gamma(gamma, error_message):
    """Test that invalid gamma values raise an error."""
    with pytest.raises(ValueError, match=re.escape(error_message)):
        phase_damping(gamma=gamma)


@pytest.mark.parametrize(
    "rho",
    [
        # 3x3 matrix.
        np.eye(3),
        # 2x3 matrix.
        np.array([[1, 0, 0], [0, 1, 0]]),
        # 1D array.
        np.array([1, 0]),
        # 2x4 matrix.
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
    ],
)
def test_invalid_dimension(rho):
    """Test that invalid matrix dimensions raise an error."""
    with pytest.raises(ValueError, match="Input matrix must be 2x2 for the phase damping channel."):
        phase_damping(rho, gamma=0.5)


def test_input_and_return_type():
    """Test for input handling and return types."""
    # Test Kraus operators when input_mat is None.
    kraus_ops = phase_damping(None, gamma=0.4)
    assert len(kraus_ops) == 2
    for op in kraus_ops:
        assert op.shape == (2, 2)

    # Test integer input matrix conversion to complex.
    input_mat = np.array([[1, 0], [0, 0]], dtype=int)
    result = phase_damping(input_mat, gamma=0.4)
    assert result.dtype == complex
