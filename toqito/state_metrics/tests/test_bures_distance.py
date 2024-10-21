"""Tests for bures_distance."""

import numpy as np

from toqito.state_metrics import bures_distance
from toqito.states import basis


def test_bures_distance_default():
    """Test bures_distance default arguments."""
    rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    sigma = rho

    res = bures_distance(rho, sigma)
    np.testing.assert_equal(np.isclose(res, 0), True)


def test_bures_distance_non_identical_states_1():
    """Test the bures_distance between two non-identical states."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = 3 / 4 * e_0 @ e_0.conj().T + 1 / 4 * e_1 @ e_1.conj().T
    sigma = 2 / 3 * e_0 @ e_0.conj().T + 1 / 3 * e_1 @ e_1.conj().T
    np.testing.assert_equal(np.isclose(bures_distance(rho, sigma), 0.0918, rtol=1e-03), True)


def test_bures_distance_non_identical_states_2():
    """Test the bures_distance between two non-identical states."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = 3 / 4 * e_0 @ e_0.conj().T + 1 / 4 * e_1 @ e_1.conj().T
    sigma = 1 / 8 * e_0 @ e_0.conj().T + 7 / 8 * e_1 @ e_1.conj().T
    np.testing.assert_equal(np.isclose(bures_distance(rho, sigma), 0.6724, rtol=1e-03), True)


def test_bures_distance_pure_states():
    """Test the bures_distance between two pure states."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_plus = (e_0 + e_1) / np.sqrt(2)
    rho = e_plus @ e_plus.conj().T
    sigma = e_0 @ e_0.conj().T
    np.testing.assert_equal(np.isclose(bures_distance(rho, sigma), 0.765, rtol=1e-03), True)


def test_bures_distance_non_square():
    """Tests for invalid dim."""
    rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    sigma = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    with np.testing.assert_raises(ValueError):
        bures_distance(rho, sigma)


def test_bures_distance_invalid_dim():
    """Tests for invalid dim."""
    rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with np.testing.assert_raises(ValueError):
        bures_distance(rho, sigma)
