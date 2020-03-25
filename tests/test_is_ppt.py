"""Tests for is_ppt function."""
import unittest
import numpy as np

from toqito.state.states.bell import bell
from toqito.state.properties.is_ppt import is_ppt


class TestIsPPT(unittest.TestCase):
    """Unit test for is_ppt."""

    def test_is_ppt(self):
        """Check that PPT matrix returns True."""
        mat = np.identity(9)
        self.assertEqual(is_ppt(mat), True)

    def test_is_ppt_sys(self):
        """Check that PPT matrix returns True with sys specified."""
        mat = np.identity(9)
        self.assertEqual(is_ppt(mat, 2), True)

    def test_is_ppt_dim_sys(self):
        """Check that PPT matrix returns True with dim and sys specified."""
        mat = np.identity(9)
        self.assertEqual(is_ppt(mat, 2, np.round(np.sqrt(mat.size))), True)

    def test_is_ppt_tol(self):
        """Check that PPT matrix returns True."""
        mat = np.identity(9)
        self.assertEqual(is_ppt(mat, 2, np.round(np.sqrt(mat.size)), 1e-10), True)

    def test_entangled_state(self):
        """Entangled state of dimension 2 will violate PPT criterion."""
        rho = bell(2) * bell(2).conj().T
        self.assertEqual(is_ppt(rho), False)


if __name__ == "__main__":
    unittest.main()
