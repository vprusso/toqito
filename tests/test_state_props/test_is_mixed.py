"""Test is_mixed."""
import numpy as np

from toqito.state_props import is_mixed
from toqito.states import basis


def test_is_mixed():
    """Return True for mixed quantum state."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    np.testing.assert_equal(is_mixed(rho), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
