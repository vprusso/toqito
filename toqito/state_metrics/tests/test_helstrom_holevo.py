"""Tests for helstrom_holevo."""
import numpy as np

from toqito.state_metrics import helstrom_holevo
from toqito.states import basis


def test_helstrom_holevo_same_state():
    """Test Helstrom-Holevo distance on same state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00 = np.kron(e_0, e_0)
    e_11 = np.kron(e_1, e_1)

    u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
    sigma_0 = u_vec * u_vec.conj().T
    sigma_1 = sigma_0

    res = helstrom_holevo(sigma_0, sigma_1)

    np.testing.assert_equal(np.isclose(res, 1 / 2), True)


def test_helstrom_holevo_non_density_matrix():
    """Test Helstrom-Holevo distance on non-density matrix."""
    rho = np.array([[1, 2], [3, 4]])
    sigma = np.array([[5, 6], [7, 8]])

    with np.testing.assert_raises(ValueError):
        helstrom_holevo(rho, sigma)


if __name__ == "__main__":
    np.testing.run_module_suite()
