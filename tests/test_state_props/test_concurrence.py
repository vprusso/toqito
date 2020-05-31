"""Test concurrence."""
import numpy as np

from toqito.state_props import concurrence
from toqito.states import basis


def test_concurrence_entangled():
    """The concurrence on maximally entangled Bell state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)

    u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
    rho = u_vec * u_vec.conj().T

    res = concurrence(rho)
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_concurrence_separable():
    """The concurrence of a product state is zero."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    v_vec = np.kron(e_0, e_1)
    sigma = v_vec * v_vec.conj().T

    res = concurrence(sigma)
    np.testing.assert_equal(np.isclose(res, 0), True)


def test_concurrence_invalid_dim():
    """Tests for invalid dimension inputs."""
    with np.testing.assert_raises(ValueError):
        rho = np.identity(5)
        concurrence(rho)


if __name__ == "__main__":
    np.testing.run_module_suite()
