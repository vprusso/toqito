print("Starting tests in test_BCS.py...")

import unittest
import numpy as np
from BCS import NonlocalGame

class TestBCS(unittest.TestCase):
    def test_has_perfect_commuting_strategy_satisfiable_bcs(self):
        print("Running satisfiable BCS test...")
        M = np.array([[1, 0],
                      [0, 1]], dtype=int)
        b = np.array([0, 0])
        game = NonlocalGame.from_bcs_game(M, b)
        self.assertTrue(game.has_perfect_commuting_measurement_strategy(), "Satisfiable BCS should be perfect.")

    def test_has_perfect_commuting_strategy_chsh_bcs(self):
        print("Running CHSH BCS test...")
        M = np.array([[1, 1],
                      [1, 1]], dtype=int)
        b = np.array([0, 1])
        game = NonlocalGame.from_bcs_game(M, b)
        self.assertFalse(game.has_perfect_commuting_measurement_strategy(), "CHSH-like BCS should not be perfect.")

    def test_has_perfect_commuting_strategy_magic_square_bcs(self):
        print("Running Magic Square BCS test...")
        def parity_constraint(indices, parity):
            n = 9
            result = np.zeros(tuple([2] * n), dtype=int)
            for assignment in range(2**n):
                bits = [(assignment >> bit) & 1 for bit in range(n)]
                total = sum(bits[i] for i in indices) % 2
                if total == parity:
                    result[tuple(bits)] = 1
            return result

        constraints = []
        constraints.append(parity_constraint([0, 1, 2], parity=0))
        constraints.append(parity_constraint([3, 4, 5], parity=0))
        constraints.append(parity_constraint([6, 7, 8], parity=0))
        constraints.append(parity_constraint([0, 3, 6], parity=1))
        constraints.append(parity_constraint([1, 4, 7], parity=1))
        constraints.append(parity_constraint([2, 5, 8], parity=1))
        
        game = NonlocalGame.from_bcs_game(constraints)
        self.assertTrue(game.has_perfect_commuting_measurement_strategy(), "Magic Square game should be perfect.")

if __name__ == "__main__":
    print("Executing tests...")
    unittest.main()



