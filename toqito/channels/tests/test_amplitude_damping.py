"""Tests for amplitude damping channel."""

import re

import numpy as np
import pytest

from toqito.channels import amplitude_damping


@pytest.mark.parametrize(
    "prob, gamma",
    [
        (0.0, 0.0),
        (0.3, 0.5),
        (0.5, 0.7),
        (1.0, 1.0),
        (0.0, 1.0),
        (1.0, 0.0),
    ],
)
def test_kraus_operators(prob, gamma):
    """Test that amplitude_damping returns correct Kraus operators when apply_channel=False."""
    result = amplitude_damping(prob=prob, gamma=gamma)

    # Test Kraus operators.
    k0_expected = np.sqrt(prob) * np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    k1_expected = np.sqrt(prob) * np.sqrt(gamma) * np.array([[0, 1], [0, 0]])
    k2_expected = np.sqrt(1 - prob) * np.array([[np.sqrt(1 - gamma), 0], [0, 1]])
    k3_expected = np.sqrt(1 - prob) * np.sqrt(gamma) * np.array([[0, 0], [1, 0]])
    expected_kraus_ops = [k0_expected, k1_expected, k2_expected, k3_expected]

    # Check if returned operators match expected ones.
    assert isinstance(result, list)
    assert len(result) == 4
    for i in range(4):
        np.testing.assert_almost_equal(result[i], expected_kraus_ops[i])

    # Check completeness relation.
    completeness_sum = np.zeros((2, 2), dtype=complex)
    for k in result:
        completeness_sum += k.conj().T @ k
    np.testing.assert_almost_equal(completeness_sum, np.eye(2))


@pytest.mark.parametrize(
    "rho, prob, gamma",
    [
        (np.array([[1, 0], [0, 0]]), 0.3, 0.4),
        (np.array([[0, 0], [0, 1]]), 0.3, 0.4),
        (np.array([[0.5, 0.5], [0.5, 0.5]]), 0.3, 0.4),
        (np.array([[0.7, 0.2j], [-0.2j, 0.3]]), 0.3, 0.4),
        (np.array([[0.7, 0.2 + 0.1j], [0.2 - 0.1j, 0.3]], dtype=complex), 0.4, 0.6),
    ],
)
def test_apply_channel_true(rho, prob, gamma):
    """Test that amplitude_damping applies the channel when apply_channel=True."""
    result = amplitude_damping(rho, prob=prob, gamma=gamma, apply_channel=True)

    # Compute expected output by applying Kraus operators manually.
    kraus_ops = amplitude_damping(prob=prob, gamma=gamma)
    expected_output = np.zeros((2, 2), dtype=complex)
    for k in kraus_ops:
        expected_output += k @ rho @ k.conj().T

    np.testing.assert_almost_equal(result, expected_output)
    np.testing.assert_almost_equal(np.trace(result), np.trace(rho))
    np.testing.assert_almost_equal(result, result.conj().T)  # Check hermiticity


def test_apply_channel_false_with_input_mat_returns_kraus():
    """Test that apply_channel=False returns Kraus operators even when input_mat is provided."""
    input_mat = np.array([[1, 0], [0, 0]])
    result = amplitude_damping(input_mat, prob=0.3, gamma=0.4, apply_channel=False)

    # Should return Kraus operators, not applied result
    assert isinstance(result, list)
    assert len(result) == 4

    # Verify these are the Kraus operators
    k0_expected = np.sqrt(0.3) * np.array([[1, 0], [0, np.sqrt(1 - 0.4)]])
    np.testing.assert_almost_equal(result[0], k0_expected)


def test_apply_channel_true_without_input_mat_raises():
    """Test that apply_channel=True with input_mat=None raises ValueError."""
    with pytest.raises(ValueError, match="input_mat is required when apply_channel=True"):
        amplitude_damping(prob=0.3, gamma=0.4, apply_channel=True)


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
    """Test that invalid matrix dimensions raise an error when apply_channel=True."""
    with pytest.raises(ValueError, match="Input matrix must be 2x2 for the generalized amplitude damping channel."):
        amplitude_damping(rho, prob=0.3, gamma=0.5, apply_channel=True)


def test_input_and_return_type():
    """Test for input handling and return types."""
    # Test Kraus operators when input_mat is None.
    kraus_ops = amplitude_damping(None, prob=0.3, gamma=0.4)
    assert len(kraus_ops) == 4
    for op in kraus_ops:
        assert op.shape == (2, 2)

    # Test integer input matrix conversion to complex when apply_channel=True.
    input_mat = np.array([[1, 0], [0, 0]], dtype=int)
    result = amplitude_damping(input_mat, prob=0.3, gamma=0.4, apply_channel=True)
    assert result.dtype == complex


def test_default_behavior_returns_kraus():
    """Test that default behavior (apply_channel=False) returns Kraus operators."""
    result = amplitude_damping(prob=0.5, gamma=0.3)
    assert isinstance(result, list)
    assert len(result) == 4
