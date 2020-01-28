"""Tests for is_pure function."""
import unittest

from toqito.states.bell import bell
from toqito.states.is_pure import is_pure


class TestIsPure(unittest.TestCase):
    """Unit test for is_pure."""

    def test_is_pure(self):
        """Ensure that pure Bell state returns True."""
        rho = bell(0) * bell(0).conj().T
        self.assertEqual(is_pure(rho), True)


if __name__ == '__main__':
    unittest.main()
