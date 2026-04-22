"""Tests for amplitude damping channel."""

import re

import numpy as np
import pytest

from toqito.channel_ops import apply_channel
from toqito.channels import amplitude_damping


@pytest.mark.parametrize(
    "rho, prob, gamma, expected_kraus",
    [
        # Kraus operator test cases.
        (None, 0.0, 0.0, True),
        (None, 0.3, 0.5, True),
        (None, 0.5, 0.7, True),
        (None, 1.0, 1.0, True),
        (None, 0.0, 1.0, True),
        (None, 1.0, 0.0, True),
        # Density matrix test cases.
        (np.array([[1, 0], [0, 0]]), 0.3, 0.4, False),
        (np.array([[0, 0], [0, 1]]), 0.3, 0.4, False),
        (np.array([[0.5, 0.5], [0.5, 0.5]]), 0.3, 0.4, False),
        (np.array([[0.7, 0.2j], [-0.2j, 0.3]]), 0.3, 0.4, False),
        (np.array([[0.7, 0.2 + 0.1j], [0.2 - 0.1j, 0.3]], dtype=complex), 0.4, 0.6, False),
    ],
)
def test_amplitude_damping(rho, prob, gamma, expected_kraus):
    """Test amplitude damping for both Kraus operators and application to states."""
    if expected_kraus:
        result = amplitude_damping(prob=prob, gamma=gamma)
        k0_expected = np.sqrt(prob) * np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        k1_expected = np.sqrt(prob) * np.sqrt(gamma) * np.array([[0, 1], [0, 0]])
        k2_expected = np.sqrt(1 - prob) * np.array([[np.sqrt(1 - gamma), 0], [0, 1]])
        k3_expected = np.sqrt(1 - prob) * np.sqrt(gamma) * np.array([[0, 0], [1, 0]])
        expected_kraus_ops = [k0_expected, k1_expected, k2_expected, k3_expected]

        for i in range(4):
            np.testing.assert_almost_equal(result[i], expected_kraus_ops[i])

        completeness_sum = np.zeros((2, 2), dtype=complex)
        for k in result:
            completeness_sum += k.conj().T @ k
        np.testing.assert_almost_equal(completeness_sum, np.eye(2))
    else:
        kraus_ops = amplitude_damping(prob=prob, gamma=gamma)
        result = apply_channel(rho, kraus_ops)

        expected_output = np.zeros((2, 2), dtype=complex)
        for k in kraus_ops:
            expected_output += k @ rho @ k.conj().T

        np.testing.assert_almost_equal(result, expected_output)
        np.testing.assert_almost_equal(np.trace(result), np.trace(rho))
        np.testing.assert_almost_equal(result, result.conj().T)


@pytest.mark.parametrize(
    "param, value, error_message",
    [
        ("prob", -0.1, "Probability must be between 0 and 1."),
        ("prob", 1.1, "Probability must be between 0 and 1."),
        ("gamma", -0.1, "Gamma (damping rate) must be between 0 and 1."),
        ("gamma", 1.1, "Gamma (damping rate) must be between 0 and 1."),
    ],
)
def test_invalid_parameters(param, value, error_message):
    """Test that invalid probabilities and gamma values raise appropriate errors."""
    kwargs = {"prob": 0.5, "gamma": 0.5}
    kwargs[param] = value

    with pytest.raises(ValueError, match=re.escape(error_message)):
        amplitude_damping(**kwargs)


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
    with pytest.raises(ValueError, match="Input matrix must be 2x2 for the generalized amplitude damping channel."):
        amplitude_damping(rho, prob=0.3, gamma=0.5)


def test_input_and_return_type():
    """Test for input handling and return types."""
    kraus_ops = amplitude_damping(prob=0.3, gamma=0.4)
    assert len(kraus_ops) == 4
    for op in kraus_ops:
        assert op.shape == (2, 2)


def test_input_mat_is_deprecated():
    """Passing `input_mat` still works but emits a DeprecationWarning."""
    rho = np.array([[1.0, 0.0], [0.0, 0.0]])
    with pytest.warns(DeprecationWarning, match="apply_channel"):
        legacy = amplitude_damping(rho, prob=0.3, gamma=0.4)
    modern = apply_channel(rho, amplitude_damping(prob=0.3, gamma=0.4))
    np.testing.assert_almost_equal(legacy, modern)
