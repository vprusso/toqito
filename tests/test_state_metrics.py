"""Testing metrics on quantum states."""
import cvxpy
import numpy as np


from toqito.state_metrics import fidelity
from toqito.state_metrics import helstrom_holevo
from toqito.state_metrics import hilbert_schmidt
from toqito.state_metrics import purity
from toqito.state_metrics import sub_fidelity
from toqito.state_metrics import trace_distance
from toqito.state_metrics import trace_norm
from toqito.state_metrics import von_neumann_entropy

from toqito.states import basis
from toqito.states import bell
from toqito.states import max_mixed


def test_fidelity_default():
    """Test fidelity default arguments."""
    rho = np.array(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )
    sigma = rho

    res = fidelity(rho, sigma)
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_fidelity_cvx():
    """Test fidelity for cvx objects."""
    rho = cvxpy.bmat(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )
    sigma = rho

    res = fidelity(rho, sigma)
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_fidelity_non_identical_states_1():
    """Test the fidelity between two non-identical states."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    sigma = 2 / 3 * e_0 * e_0.conj().T + 1 / 3 * e_1 * e_1.conj().T
    np.testing.assert_equal(np.isclose(fidelity(rho, sigma), 0.996, rtol=1e-03), True)


def test_fidelity_non_identical_states_2():
    """Test the fidelity between two non-identical states."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    sigma = 1 / 8 * e_0 * e_0.conj().T + 7 / 8 * e_1 * e_1.conj().T
    np.testing.assert_equal(np.isclose(fidelity(rho, sigma), 0.774, rtol=1e-03), True)


def test_fidelity_non_square():
    """Tests for invalid dim."""
    rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    sigma = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
    with np.testing.assert_raises(ValueError):
        fidelity(rho, sigma)


def test_fidelity_invalid_dim():
    """Tests for invalid dim."""
    rho = np.array(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )
    sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with np.testing.assert_raises(ValueError):
        fidelity(rho, sigma)


def test_helstrom_holevo_same_state():
    """Test Helstrom-Holevo distance on same state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00 = np.kron(e_0, e_0)
    e_11 = np.kron(e_1, e_1)

    u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
    rho = u_vec * u_vec.conj().T
    sigma = rho

    res = helstrom_holevo(rho, sigma)

    np.testing.assert_equal(np.isclose(res, 1 / 2), True)


def test_helstrom_holevo_non_density_matrix():
    """Test Helstrom-Holevo distance on non-density matrix."""
    rho = np.array([[1, 2], [3, 4]])
    sigma = np.array([[5, 6], [7, 8]])

    with np.testing.assert_raises(ValueError):
        helstrom_holevo(rho, sigma)


def test_hilbert_schmidt_bell():
    r"""Test Hilbert-Schmidt distance on two Bell states."""

    rho = bell(0) * bell(0).conj().T
    sigma = bell(3) * bell(3).conj().T

    res = hilbert_schmidt(rho, sigma)

    np.testing.assert_equal(np.isclose(res, 1), True)


def test_hilbert_schmidt_non_density_matrix():
    r"""Test Hilbert-Schmidt distance on non-density matrix."""
    rho = np.array([[1, 2], [3, 4]])
    sigma = np.array([[5, 6], [7, 8]])

    with np.testing.assert_raises(ValueError):
        hilbert_schmidt(rho, sigma)


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


def test_sub_fidelity_default():
    """Test sub_fidelity default arguments."""
    rho = bell(0) * bell(0).conj().T
    sigma = rho

    res = sub_fidelity(rho, sigma)
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_sub_fidelity_lower_bound_1():
    """Test sub_fidelity is lower bound on fidelity for rho and sigma."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    sigma = 2 / 3 * e_0 * e_0.conj().T + 1 / 3 * e_1 * e_1.conj().T

    res = sub_fidelity(rho, sigma)
    np.testing.assert_array_less(res, fidelity(rho, sigma))


def test_sub_fidelity_lower_bound_2():
    """Test sub_fidelity is lower bound on fidelity for rho and pi."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    sigma = 1 / 8 * e_0 * e_0.conj().T + 7 / 8 * e_1 * e_1.conj().T

    res = sub_fidelity(rho, sigma)
    np.testing.assert_array_less(res, fidelity(rho, sigma))


def test_non_square_sub_fidelity():
    """Tests for invalid dim for sub_fidelity."""
    rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [1, 0, 0, 1]])
    sigma = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [2, 0, 0, 2]])
    with np.testing.assert_raises(ValueError):
        sub_fidelity(rho, sigma)


def test_sub_fidelity_invalid_dim_sub_fidelity():
    """Tests for invalid dim for sub_fidelity."""
    rho = np.array(
        [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    )
    sigma = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with np.testing.assert_raises(ValueError):
        sub_fidelity(rho, sigma)


def test_trace_distance_same_state():
    r"""Test that: :math:`T(\rho, \sigma) = 0` iff `\rho = \sigma`."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00 = np.kron(e_0, e_0)
    e_11 = np.kron(e_1, e_1)

    u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
    rho = u_vec * u_vec.conj().T
    sigma = rho

    res = trace_distance(rho, sigma)

    np.testing.assert_equal(np.isclose(res, 0), True)


def test_trace_distance_non_density_matrix():
    r"""Test trace distance on non-density matrix."""
    rho = np.array([[1, 2], [3, 4]])
    sigma = np.array([[5, 6], [7, 8]])

    with np.testing.assert_raises(ValueError):
        trace_distance(rho, sigma)


def test_trace_norm():
    """Test trace norm."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00 = np.kron(e_0, e_0)
    e_11 = np.kron(e_1, e_1)

    u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
    rho = u_vec * u_vec.conj().T

    res = trace_norm(rho)
    _, singular_vals, _ = np.linalg.svd(rho)
    expected_res = float(np.sum(singular_vals))

    np.testing.assert_equal(np.isclose(res, expected_res), True)


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
