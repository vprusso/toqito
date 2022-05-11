"""Test is_npt."""
import numpy as np

from toqito.state_props import is_npt
from toqito.states import bell, horodecki


def test_is_npt():
    """Check that NPT matrix returns True."""
    mat = np.identity(9)
    np.testing.assert_equal(is_npt(mat), False)


def test_is_npt_sys():
    """Check that NPT matrix returns True with sys specified."""
    mat = np.identity(9)
    np.testing.assert_equal(is_npt(mat, 2), False)


def test_is_npt_dim_sys():
    """Check that NPT matrix returns True with dim and sys specified."""
    mat = np.identity(9)
    np.testing.assert_equal(is_npt(mat, 2, np.round(np.sqrt(mat.size))), False)


def test_is_npt_tol():
    """Check that NPT matrix returns True."""
    mat = np.identity(9)
    np.testing.assert_equal(is_npt(mat, 2, np.round(np.sqrt(mat.size)), 1e-10), False)


def test_entangled_state():
    """Entangled state of dimension 2 will violate NPT criterion."""
    rho = bell(2) * bell(2).conj().T
    np.testing.assert_equal(is_npt(rho), True)


def test_is_horodecki_npt():
    """Horodecki state is an example of an entangled NPT state."""
    np.testing.assert_equal(is_npt(horodecki(0.5, [3, 3])), False)


if __name__ == "__main__":
    np.testing.run_module_suite()
