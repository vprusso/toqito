"""Tests for generalized amplitude damping channel."""

import numpy as np
import pytest

from toqito.channels import generalized_amplitude_damping


@pytest.mark.parametrize("prob, gamma", [(0.0, 0.0), (0.3, 0.5), (0.5, 0.7), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)])
def test_kraus_operators(prob, gamma):
    """Test if the function returns correct Kraus operators for given probability and gamma."""
    kraus_ops = generalized_amplitude_damping(None, prob=prob, gamma=gamma)

    k0_expected = np.sqrt(prob) * np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    k1_expected = np.sqrt(prob) * np.sqrt(gamma) * np.array([[0, 1], [0, 0]])
    k2_expected = np.sqrt(1 - prob) * np.array([[np.sqrt(1 - gamma), 0], [0, 1]])
    k3_expected = np.sqrt(1 - prob) * np.sqrt(gamma) * np.array([[0, 0], [1, 0]])
    expected_kraus_ops = [k0_expected, k1_expected, k2_expected, k3_expected]

    for i in range(4):
        np.testing.assert_almost_equal(kraus_ops[i], expected_kraus_ops[i])

    completeness_sum = np.zeros((2, 2), dtype=complex)
    for k in kraus_ops:
        completeness_sum += k.conj().T @ k
    np.testing.assert_almost_equal(completeness_sum, np.eye(2))


@pytest.mark.parametrize(
    "rho, prob, gamma",
    [
        # Ground state |0⟩⟨0|.
        (np.array([[1, 0], [0, 0]]), 0.3, 0.4),
        # Excited state |1⟩⟨1|.
        (np.array([[0, 0], [0, 1]]), 0.3, 0.4),
        # Superposition state (|0⟩+|1⟩)(⟨0|+⟨1|)/2.
        (np.array([[0.5, 0.5], [0.5, 0.5]]), 0.3, 0.4),
        # Mixed state.
        (np.array([[0.7, 0.2j], [-0.2j, 0.3]]), 0.3, 0.4),
        # Complex mixed state.
        (np.array([[0.7, 0.2 + 0.1j], [0.2 - 0.1j, 0.3]], dtype=complex), 0.4, 0.6),
    ],
)
def test_apply_to_states(rho, prob, gamma):
    """Apply the channel to various input states rho."""
    kraus_ops = generalized_amplitude_damping(None, prob=prob, gamma=gamma)

    expected_output = np.zeros((2, 2), dtype=complex)
    for k in kraus_ops:
        expected_output += k @ rho @ k.conj().T

    result = generalized_amplitude_damping(rho, prob=prob, gamma=gamma)

    np.testing.assert_almost_equal(result, expected_output)
    np.testing.assert_almost_equal(np.trace(result), np.trace(rho))
    np.testing.assert_almost_equal(result, result.conj().T)


@pytest.mark.parametrize(
    "prob, error_message",
    [
        (-0.1, "Probability must be between 0 and 1."),
        (1.1, "Probability must be between 0 and 1."),
    ],
)
def test_invalid_prob(prob, error_message):
    """Test that invalid probabilities raise an error."""
    with pytest.raises(ValueError, match=error_message):
        generalized_amplitude_damping(prob=prob, gamma=0.5)


@pytest.mark.parametrize(
    "gamma, error_message",
    [
        (-0.1, "Probability must be between 0 and 1."),
        (1.1, "Probability must be between 0 and 1."),
    ],
)
def test_invalid_gamma(gamma, error_message):
    """Test that invalid gamma values raise an error."""
    with pytest.raises(ValueError, match=error_message):
        generalized_amplitude_damping(prob=0.5, gamma=gamma)


@pytest.mark.parametrize(
    "rho",
    [
        np.eye(3),  # 3x3 matrix.
        np.array([[1, 0, 0], [0, 1, 0]]),  # 2x3 matrix.
        np.array([1, 0]),  # 1D array.
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),  # 2x4 matrix.
    ],
)
def test_invalid_dimension(rho):
    """Test that invalid matrix dimensions raise an error."""
    with pytest.raises(ValueError, match="Input matrix must be 2x2 for the generalized amplitude damping channel."):
        generalized_amplitude_damping(rho, prob=0.3, gamma=0.5)


def test_input_and_return_type():
    """Test for input handling and return types."""
    # Test Kraus operators when input_mat is None.
    kraus_ops = generalized_amplitude_damping(None, prob=0.3, gamma=0.4)
    assert len(kraus_ops) == 4
    for op in kraus_ops:
        assert op.shape == (2, 2)

    # Test integer input matrix conversion to complex.
    input_mat = np.array([[1, 0], [0, 0]], dtype=int)
    result = generalized_amplitude_damping(input_mat, prob=0.3, gamma=0.4)
    assert result.dtype == complex
