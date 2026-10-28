"""Tests for measured relative entropy."""

import numpy as np
import pytest
from scipy.special import rel_entr

from toqito.matrices import pauli
from toqito.state_metrics import measured_relative_entropy


def bernoulli_relative_entropy(r: np.ndarray, s: np.ndarray, alpha: float) -> float:
    """Bernoulli relative entropy."""
    rnorm = np.linalg.norm(r)
    snorm = np.linalg.norm(s)
    p = (1 + rnorm * np.cos(alpha)) / 2
    phi = np.arccos(np.dot(r, s) / rnorm / snorm)
    q = (1 + snorm * np.cos(alpha - phi)) / 2
    # rel_entr treats p = 0 as contributing 0, so pure states (|r| = 1) are handled.
    return rel_entr(p, q) + rel_entr(1 - p, 1 - q)


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


eps = 1e-5

r1 = np.array([0.9, 0.05, -0.02])
s1 = np.array([-0.8, 0.1, 0.1])

r2 = np.array([0.1, 0.2, 0.3])
s2 = np.array([-0.1, -0.2, -0.3])

r3 = np.array([-0.4, -0.12, 0.35])
s3 = np.array([0.23, -0.15, 0.06])

r4 = np.array([0.1, 0.1, 0.1])
s4 = np.array([0.1, 0.1, 0.1])

# Pure rho (|r| = 1) against full-rank sigma. The previous implementation inverted sqrtm(rho)
# here and read a platform-dependent quadrature size off a singular matrix.
r7 = np.array([0.0, 1.0, 0.0])
s7 = np.array([0.3, 0.2, 0.1])

r8 = np.array([0.6, 0.0, 0.8])
s8 = np.array([-0.1, 0.4, 0.2])


@pytest.mark.parametrize(
    "r, s, eps",
    [
        (r1, s1, eps),
        (r2, s2, eps),
        (r3, s3, eps),
        # test when states are the same
        (r4, s4, eps),
        # pure rho, full-rank sigma
        (r7, s7, eps),
        (r8, s8, eps),
    ],
)
def test_measured_relative_entropy(r: np.ndarray, s: np.ndarray, eps: float):
    """Test function works as expected for valid inputs."""
    rho = state(r)
    sigma = state(s)
    calculated_result = measured_relative_entropy(rho, sigma, eps)
    expected = qubit_measured_relative_entropy(r, s)
    assert abs(calculated_result - expected) <= 1e-04


r5 = np.array([1, 1, 1])
s5 = np.array([0.5, 0.5, 0.5])

r6 = np.array([0.1, 0.1, 0.1])
s6 = np.array([1, 1, 1])


@pytest.mark.parametrize(
    "r, s, eps, expected_msg",
    [
        # rho not density operator
        (r5, s5, eps, "Measured relative entropy is only defined if rho is a density operator."),
        # sigma is not positive semi-definite
        (r6, s6, eps, "Measured relative entropy is only defined if sigma is positive semi-definite."),
    ],
)
def test_meausred_relative_entropy_invalid_input(r: np.ndarray, s: np.ndarray, eps: float, expected_msg: str):
    """Test function raises an epsor for invalid inputs."""
    rho = state(r)
    sigma = state(s)
    with pytest.raises(ValueError, match=expected_msg):
        measured_relative_entropy(rho, sigma, eps)


def test_measured_relative_entropy_pure_against_maximally_mixed():
    """|0><0| against I/2 is ln(2): the computational-basis measurement attains the Umegaki value."""
    rho = np.diag([1.0, 0.0])
    sigma = np.identity(2) / 2
    assert abs(measured_relative_entropy(rho, sigma, eps) - np.log(2)) <= 1e-4


def test_measured_relative_entropy_singular_sigma_on_support():
    """A singular sigma whose support contains rho is handled by compressing onto that support."""
    plus = np.array([[1.0], [1.0], [0.0]]) / np.sqrt(2)
    rho = plus @ plus.conj().T
    sigma = np.diag([0.5, 0.5, 0.0])
    assert abs(measured_relative_entropy(rho, sigma, eps) - np.log(2)) <= 1e-4


def test_measured_relative_entropy_commuting_singular_sigma():
    """Commuting rank-deficient inputs reduce to the classical relative entropy of the spectra."""
    rho = np.diag([0.5, 0.5, 0.0])
    sigma = np.diag([0.2, 0.8, 0.0])
    expected = 0.5 * np.log(0.5 / 0.2) + 0.5 * np.log(0.5 / 0.8)
    assert abs(measured_relative_entropy(rho, sigma, eps) - expected) <= 1e-4


def test_measured_relative_entropy_support_not_contained():
    """If rho has weight outside the support of sigma, the value is infinite and an error is raised."""
    rho = np.diag([0.7, 0.3])
    sigma = np.diag([1.0, 0.0])
    with pytest.raises(ValueError, match="support of rho is not contained in the support of sigma"):
        measured_relative_entropy(rho, sigma, eps)
