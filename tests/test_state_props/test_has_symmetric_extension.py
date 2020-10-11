"""Test has_symmetric_extension."""
import numpy as np

from toqito.state_props import has_symmetric_extension
from toqito.states import bell


def test_has_symmetric_extension_two_qubit():
    """Check whether 2-qubit state has a symmetric extension."""
    rho = np.array([[1, 0, 0, -1], [0, 1, 1 / 2, 0], [0, 1 / 2, 1, 0], [-1, 0, 0, 1]])
    np.testing.assert_equal(has_symmetric_extension(rho), True)


def test_has_symmetric_extension_not_ppt_level_1():
    """Check whether state has level-1 symmetric extension."""
    rho = np.identity(4)
    np.testing.assert_equal(has_symmetric_extension(rho, level=1, dim=None, ppt=False), True)


def test_has_symmetric_extension_entangled_false():
    """Entangled state should not have symmetric extension for some level."""
    rho = bell(0) * bell(0).conj().T
    np.testing.assert_equal(has_symmetric_extension(rho, level=1), False)


def test_has_symmetric_extension_level_2_entangled_false():
    """Entangled state should not have symmetric extension for some level (level-2)."""
    rho = bell(0) * bell(0).conj().T
    np.testing.assert_equal(has_symmetric_extension(rho, level=2), False)


def test_has_symmetric_extension_dim_list():
    """Provide dimension of system as list."""
    rho = bell(0) * bell(0).conj().T
    np.testing.assert_equal(has_symmetric_extension(rho, level=2, dim=[2, 2]), False)


def test_has_symmetric_extension_level_2_entangled_false_non_ppt():
    """Entangled state should not have non-PPT-symmetric extension for some level (level-2)."""
    rho = bell(0) * bell(0).conj().T
    np.testing.assert_equal(has_symmetric_extension(rho, level=2, dim=2, ppt=False), False)


def test_has_symmetric_extension_level_2_entangled_false_ppt():
    """Entangled state should not have PPT-symmetric extension for some level (level-2)."""
    rho = bell(0) * bell(0).conj().T
    rho = np.kron(rho, rho)
    np.testing.assert_equal(has_symmetric_extension(rho, level=2, dim=2), False)


def test_has_symmetric_extension_invalid_dim():
    """Tests for invalid dimension inputs."""
    with np.testing.assert_raises(ValueError):
        rho = np.identity(6)
        has_symmetric_extension(rho, level=1, dim=4)


if __name__ == "__main__":
    np.testing.run_module_suite()
