"""Test cases for the bit-flip channel."""

import numpy as np
import pytest

from toqito.channels import bitflip


@pytest.mark.parametrize("prob", [0.0, 0.3, 0.5, 1.0])
def test_kraus_operators(prob):
    """Test if the function returns correct Kraus operators for given probability."""
    kraus_ops = bitflip(None, prob=prob)  # Get Kraus operators
    expected_kraus_ops = [
        np.sqrt(1 - prob) * np.eye(2),
        np.sqrt(prob) * np.array([[0, 1], [1, 0]]),
    ]

    np.testing.assert_almost_equal(kraus_ops, expected_kraus_ops)


@pytest.mark.parametrize(
    "rho, expected_output, prob",
    [
        (np.array([[1, 0], [0, 0]]), (1 - 0.3) * np.array([[1, 0], [0, 0]]) + 0.3 * np.array([[0, 0], [0, 1]]), 0.3),
        (np.array([[0, 0], [0, 1]]), (1 - 0.3) * np.array([[0, 0], [0, 1]]) + 0.3 * np.array([[1, 0], [0, 0]]), 0.3),
    ],
)
def test_apply_to_state(rho, expected_output, prob):
    """Test bitflip application to |0><0| and |1><1| states."""
    result = bitflip(rho, prob=prob)
    np.testing.assert_almost_equal(result, expected_output)


@pytest.mark.parametrize(
    "rho, prob, expected_output",
    [
        # Test probability 0 (no change).
        (np.array([[0.5, 0.5], [0.5, 0.5]]), 0, np.array([[0.5, 0.5], [0.5, 0.5]])),
        # Test probability 1 (always flips, but same for max mixed state).
        (np.array([[0.5, 0.5], [0.5, 0.5]]), 1, np.array([[0.5, 0.5], [0.5, 0.5]])),
    ],
)
def test_bitflip_probabilities(rho, prob, expected_output):
    """Test bitflip channel with probabilities 0 and 1."""
    result = bitflip(rho, prob=prob)
    np.testing.assert_almost_equal(result, expected_output)


@pytest.mark.parametrize(
    "prob, error_message",
    [
        (-0.1, "Probability must be between 0 and 1."),
        (1.1, "Probability must be between 0 and 1."),
    ],
)
def test_invalid_probability(prob, error_message):
    """Test that invalid probabilities raise an error."""
    with pytest.raises(ValueError, match=error_message):
        bitflip(prob=prob)


@pytest.mark.parametrize(
    "rho",
    [
        np.eye(3),  # 3x3 matrix
        np.array([[1, 0, 0], [0, 1, 0]]),  # 2x3 matrix
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),  # 2x4 matrix
    ],
)
def test_invalid_dimension(rho):
    """Test that invalid matrix dimensions raise an error."""
    with pytest.raises(ValueError, match="Input matrix must be 2x2 for the bitflip channel."):
        bitflip(rho, prob=0.3)


def test_apply_to_mixed_state():
    """Test bitflip channel on a mixed state."""
    prob = 0.4
    rho = np.array([[0.7, 0.2], [0.2, 0.3]])
    expected_output = (1 - prob) * rho + prob * np.array([[0.3, 0.2], [0.2, 0.7]])
    result = bitflip(rho, prob=prob)

    np.testing.assert_almost_equal(result, expected_output)
