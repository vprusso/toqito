"""Test is_ppt."""
import numpy as np

from toqito.state_props import is_ppt
from toqito.states import bell, horodecki


def test_is_ppt():
    """Check that PPT matrix returns True."""
    mat = np.identity(9)
    np.testing.assert_equal(is_ppt(mat), True)


def test_is_ppt_sys():
    """Check that PPT matrix returns True with sys specified."""
    mat = np.identity(9)
    np.testing.assert_equal(is_ppt(mat, 2), True)


def test_is_ppt_dim_sys():
    """Check that PPT matrix returns True with dim and sys specified."""
    mat = np.identity(9)
    np.testing.assert_equal(is_ppt(mat, 2, np.round(np.sqrt(mat.size))), True)


def test_is_ppt_tol():
    """Check that PPT matrix returns True."""
    mat = np.identity(9)
    np.testing.assert_equal(is_ppt(mat, 2, np.round(np.sqrt(mat.size)), 1e-10), True)


def test_entangled_state():
    """Entangled state of dimension 2 will violate PPT criterion."""
    rho = bell(2) * bell(2).conj().T
    np.testing.assert_equal(is_ppt(rho), False)


def test_is_horodecki_ppt():
    """Horodecki state is an example of an entangled PPT state."""
    np.testing.assert_equal(is_ppt(horodecki(0.5, [3, 3])), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
