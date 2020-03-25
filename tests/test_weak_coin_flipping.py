"""Tests for weak_coin_flipping function."""
import unittest
import numpy as np

from toqito.base.ket import ket
from toqito.nonlocal_games.coin_flipping.weak_coin_flipping import weak_coin_flipping


class TestWeakCoinFlipping(unittest.TestCase):
    """Unit test for weak_coin_flipping."""

    def test_weak_coin_flipping(self):
        """
        Test for maximally entangled state.

        Refer to the appendix of https://arxiv.org/abs/1703.03887
        """
        e_0, e_1 = ket(2, 0), ket(2, 1)
        e_m = (e_0 - e_1) / np.sqrt(2)

        rho = np.kron(e_1 * e_1.conj().T, e_0 * e_0.conj().T) + np.kron(
            e_m * e_m.conj().T, e_1 * e_1.conj().T
        )

        self.assertEqual(
            np.isclose(weak_coin_flipping(rho), np.cos(np.pi / 8) ** 2), True
        )


if __name__ == "__main__":
    unittest.main()
