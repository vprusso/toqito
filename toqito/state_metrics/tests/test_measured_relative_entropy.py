"""Tests for measured relative entropy."""

import numpy as np
import pytest

from toqito.matrices import pauli
from toqito.state_metrics import measured_relative_entropy


def bernoulli_relative_entropy(r: np.ndarray, s: np.ndarray, alpha: float) -> float:
    """Bernoulli relative entropy."""
    rnorm = np.linalg.norm(r)
    snorm = np.linalg.norm(s)
    p = (1 + rnorm * np.cos(alpha)) / 2
    phi = np.arccos(np.dot(r, s) / rnorm / snorm)
    q = (1 + snorm * np.cos(alpha - phi)) / 2
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def state(vec: np.ndarray) -> np.ndarray:
    """Vector to matrix representation of state."""
    return 0.5 * (pauli("I") + vec[0] * pauli("X") + vec[1] * pauli("Y") + vec[2] * pauli("Z"))


def qubit_measured_relative_entropy(r: np.ndarray, s: np.ndarray) -> float:
    """Measured relative entropy for qubit states."""
    # sampling 1000 points in the interval [0, 2pi] usually achieves enough precision
    n = 1000
    results = np.zeros(n)
    for i in range(n):
        alpha = 2 * np.pi * i / n
        results[i] = bernoulli_relative_entropy(r, s, alpha)
    return np.max(results)


err = 1e-5

r1 = np.array([0.9, 0.05, -0.02])
s1 = np.array([-0.8, 0.1, 0.1])

r2 = np.array([0.1, 0.2, 0.3])
s2 = np.array([-0.1, -0.2, -0.3])

r3 = np.array([-0.4, -0.12, 0.35])
s3 = np.array([0.23, -0.15, 0.06])

r4 = np.array([0.1, 0.1, 0.1])
s4 = np.array([0.1, 0.1, 0.1])


@pytest.mark.parametrize(
    "r, s, err",
    [
        (r1, s1, err),
        (r2, s2, err),
        (r3, s3, err),
        # test when states are the same
        (r4, s4, err),
    ],
)
def test_measured_relative_entropy(r: np.ndarray, s: np.ndarray, err: float):
    """Test functions works as expected for valid inputs."""
    rho = state(r)
    sigma = state(s)
    calculated_result = measured_relative_entropy(rho, sigma, err)
    expected = qubit_measured_relative_entropy(r, s)
    assert abs(calculated_result - expected) <= 1e-04


r5 = np.array([1, 1, 1])
s5 = np.array([0.5, 0.5, 0.5])

r6 = np.array([0.1, 0.1, 0.1])
s6 = np.array([1, 1, 1])


@pytest.mark.parametrize(
    "r, s, err, expected_msg",
    [
        # rho not density operator
        (r5, s5, err, "Measured relative entropy is only defined if rho is a density operator."),
        # sigma is not positive semi-definite
        (r6, s6, err, "Measured relative entropy is only defined if sigma is positive semi-definite."),
    ],
)
def test_meausred_relative_entropy_invalid_input(r: np.ndarray, s: np.ndarray, err: float, expected_msg: str):
    """Test function raises an error for invalid inputs."""
    rho = state(r)
    sigma = state(s)
    with pytest.raises(ValueError, match=expected_msg):
        measured_relative_entropy(rho, sigma, err)
