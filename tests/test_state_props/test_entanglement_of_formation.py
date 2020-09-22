"""Test entanglement_of_formation."""
import numpy as np

from toqito.state_props import entanglement_of_formation
from toqito.states import basis, bell, max_mixed


def test_entanglement_of_formation_bell_state():
    """The entanglement-of-formation on a Bell state."""
    u_vec = bell(0)
    rho = u_vec * u_vec.conj().T

    res = entanglement_of_formation(rho)
    np.testing.assert_equal(np.isclose(res, 1), True)


def test_entanglement_of_formation_maximally_mixed_state():
    """The entanglement-of-formation on a maximally mixed."""
    u_vec = max_mixed(4, False)
    rho = u_vec * u_vec.conj().T

    res = entanglement_of_formation(rho)
    np.testing.assert_equal(np.isclose(res, 0), True)


def test_entanglement_of_formation_invalid_local_dim():
    """Invalid local dimension for entanglement_of_formation."""
    with np.testing.assert_raises(ValueError):
        rho = np.identity(4)
        entanglement_of_formation(rho, 3)


def test_entanglement_of_formation_mixed_state():
    """Not presently known how to calculate for mixed states."""
    with np.testing.assert_raises(ValueError):
        e_0, e_1 = basis(2, 0), basis(2, 1)
        rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
        entanglement_of_formation(rho)


def test_entanglement_of_formation_invalid_non_square():
    """Invalid non-square matrix for entanglement_of_formation."""
    with np.testing.assert_raises(ValueError):
        rho = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        entanglement_of_formation(rho)


if __name__ == "__main__":
    np.testing.run_module_suite()
