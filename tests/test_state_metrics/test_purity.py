"""Tests for purity."""
import numpy as np

from toqito.state_metrics import purity


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


if __name__ == "__main__":
    np.testing.run_module_suite()
