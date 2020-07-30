"""Tests for purity."""
import numpy as np

from toqito.states import werner
from toqito.state_props import purity


def test_purity():
    """Test for identity matrix."""
    expected_res = 1 / 4
    res = purity(np.identity(4) / 4)
    np.testing.assert_equal(res, expected_res)


def test_purity_non_density_matrix():
    r"""Test purity on non-density matrix."""
    rho = np.array([[1, 2], [3, 4]])

    with np.testing.assert_raises(ValueError):
        purity(rho)


def test_purity_werner_state():
    """Test purity of mixed Werner state."""
    res = purity(werner(2, 1 / 4))
    np.testing.assert_equal(np.isclose(res, 0.2653, atol=4), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
