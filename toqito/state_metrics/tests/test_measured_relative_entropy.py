"""Tests for measured relative entropy."""

import numpy as np
import pytest

from toqito.matrices import pauli
from toqito.state_metrics import measured_relative_entropy


def D_bern(r, s, alpha):
    """Bernoulli relative entropy."""
    rnorm = np.linalg.norm(r)
    snorm = np.linalg.norm(s)
    p = (1 + rnorm * np.cos(alpha)) / 2
    phi = np.arccos(np.dot(r, s) / rnorm / snorm)
    q = (1 + snorm * np.cos(alpha - phi)) / 2
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def state(vec):
    """Vector to matrix representation of state."""
    I2 = pauli("I")
    X = pauli("X")
    Y = pauli("Y")
    Z = pauli("Z")
    return 0.5 * (I2 + vec[0] * X + vec[1] * Y + vec[2] * Z)


def Dmk_qubit(r, s):
    """Measured relative entropy for qubit states."""
    n = 1000
    results = np.zeros(n)
    for i in range(n):
        alpha = 2 * np.pi * i / n
        results[i] = D_bern(r, s, alpha)
    return np.max(results)


r1 = np.array([0.9, 0.05, -0.02])
s1 = np.array([-0.8, 0.1, 0.1])
err1 = 10e-3

r2 = np.array([0.1, 0.2, 0.3])
s2 = np.array([-0.1, -0.2, -0.3])
err2 = 10e-3

r3 = np.array([-0.4, -0.12, 0.35])
s3 = np.array([0.23, -0.15, 0.06])
err3 = 10e-3

r4 = np.array([0.1, 0.1, 0.1])
s4 = np.array([0.1, 0.1, 0.1])
err4 = 10e-5


@pytest.mark.parametrize(
    "r, s, err",
    [
        (r1, s1, err1),
        (r2, s2, err2),
        (r3, s3, err3),
        # test when states are the same
        (r4, s4, err4),
    ],
)
def test_measured_relative_entropy(r, s, err):
    """Test functions works as expected for valid inputs."""
    rho = state(r)
    sigma = state(s)
    calculated_result = measured_relative_entropy(rho, sigma, err)
    expected = Dmk_qubit(r, s)
    assert abs(calculated_result - expected) <= 1e-03
