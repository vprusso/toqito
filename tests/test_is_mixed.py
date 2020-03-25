"""Tests for is_mixed function."""
import unittest

from toqito.base.ket import ket
from toqito.state.properties.is_mixed import is_mixed


class TestIsMixed(unittest.TestCase):
    """Unit test for is_mixed."""

    def test_is_mixed(self):
        """Return True for mixed quantum state."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
        self.assertEqual(is_mixed(rho), True)


if __name__ == "__main__":
    unittest.main()
