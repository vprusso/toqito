"""Tests for von_neumann_entropy."""
import numpy as np

from toqito.state_props import von_neumann_entropy
from toqito.states import bell
from toqito.states import max_mixed


def test_von_neumann_entropy_bell_state():
    """Entangled state von Neumann entropy should be zero."""
    rho = bell(0) * bell(0).conj().T
    res = von_neumann_entropy(rho)
    np.testing.assert_equal(np.isclose(res, 0), True)


def test_von_neumann_entropy_max_mixed_statre():
    """Von Neumann entropy of the maximally mixed state should be one."""
    res = von_neumann_entropy(max_mixed(2, is_sparse=False))
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_von_neumann_non_density_matrix():
    r"""Test von Neumann entropy on non-density matrix."""
    rho = np.array([[1, 2], [3, 4]])

    with np.testing.assert_raises(ValueError):
        von_neumann_entropy(rho)


if __name__ == "__main__":
    np.testing.run_module_suite()
