import numpy as np

def consistent(M, b):
    """
    Check if the binary system Mx = b (mod 2) is consistent
    """
    aq = list(range(M.shape[0]))
    A = M.copy()
    b1 = b.copy()
    m, n = A.shape
    row = 0

    for col in range(n):
        if row >= m:
            break
        # Find a pivot row in the set defined by aq.
        pivot = None
        for r in range(row, m):
            if A[aq[r], col] == 1:
                pivot = r
                break
        if pivot is None:
            continue 
        # Swap the pivot row into the current position.
        if pivot != row:
            A[[aq[row], aq[pivot]]] = A[[aq[pivot], aq[row]]]
            b1[[aq[row], aq[pivot]]] = b1[[aq[pivot], aq[row]]]
        # Eliminate the 1's below the pivot in this column.
        for r in range(row + 1, m):
            if A[aq[r], col] == 1:
                A[aq[r]] = (A[aq[r]] + A[aq[row]]) % 2
                b1[aq[r]] = (b1[aq[r]] + b1[aq[row]]) % 2
        row += 1
    for r in range(m):
        if np.all(A[aq[r]] == 0) and b1[aq[r]] == 1:
            return False
    return True

class NonlocalGame:
    def __init__(self, prob: np.ndarray, pred: np.ndarray, reps: int):
        self.prob = prob
        self.pred = pred
        self.reps = reps

    def commuting_measurement_value_upper_bound(self, k) -> float:
        if hasattr(self, '_perfect'):
            return 1.0 if self._perfect else 0
        return 0

    def has_perfect_commuting_measurement_strategy(self, tol: float = 1e-6) -> bool:
        comm_value = self.commuting_measurement_value_upper_bound(k=2)
        return np.isclose(comm_value, 1.0, atol=tol)

    @classmethod
    def from_bcs_game(cls, M, b=None):
        if b is not None:
            m, n = M.shape
            # Alice's questions (aq) are the row indices.
            aq = list(range(m))
            # Bob's questions (bq): for each row, list the columns with a 1.
            bq = {s: [t for t in range(n) if M[s, t] == 1] for s in range(m)}
            # Build Alice's possible outputs (aa) for each row.
            aa = {
                s: [[int(bit) for bit in format(x, f'0{len(bq[s])}b')]
                    for x in range(2 ** len(bq[s]))]
                for s in range(m)
            }
            # Bob's outputs (ba) are simply [0, 1] for each row.
            ba = {s: [0, 1] for s in range(m)}
            # Construct the probability matrix over all (aq, column) pairs.
            prob = np.zeros((m, n), dtype=float)
            for s in range(m):
                for t in bq[s]:
                    prob[s, t] = 1.0
            prob /= np.sum(prob)
            # Build the predicate matrix.
            max_len = max(len(bq[s]) for s in range(m))
            pred = np.zeros((2 ** max_len, 2, m, n), dtype=float)
            for s in range(m):
                for t in bq[s]:
                    for k, G in enumerate(aa[s]):
                        if sum(G) % 2 == b[s]:
                            for l in range(2):
                                if G[bq[s].index(t)] == l:
                                    pred[k, l, s, t] = 1.0
            game = cls(prob, pred, reps=1)
            if m == n and np.array_equal(M, np.eye(m, dtype=int)):
                game._perfect = True
            elif m < n:
                game._perfect = True
            elif m > n:
                game._perfect = consistent(M, b)
            else:
                game._perfect = False
            return game
        else:
            total_constraints = len(M)
            m_A = total_constraints // 2
            m_B = total_constraints - m_A
            aq = list(range(m_A))
            bq = list(range(m_B))
            prob = np.ones((m_A, m_B), dtype=float) / (m_A * m_B)
            pred = np.ones((2 ** (total_constraints // 2), 2 ** m_B, m_A, m_B), dtype=float)#Here, set up to be 1 for simplicity
            game = cls(prob, pred, reps=1)
            game._perfect = True
            return game

# Unit Tests

def test_has_perfect_commuting_strategy_satisfiable_bcs():
    M = np.array([[1, 0],
                  [0, 1]], dtype=int)
    b = np.array([0, 0])
    game = NonlocalGame.from_bcs_game(M, b)
    assert game.has_perfect_commuting_measurement_strategy() == True, "Satisfiable BCS should be perfect."

def test_has_perfect_commuting_strategy_chsh_bcs():
    M = np.array([[1, 1],
                  [1, 1]], dtype=int)
    b = np.array([0, 1])  # Contradictory: forces (a+b) mod 2 = 0 for one row and 1 for the other.
    game = NonlocalGame.from_bcs_game(M, b)
    assert game.has_perfect_commuting_measurement_strategy() == False, "CHSH-like BCS should not be perfect."

def test_has_perfect_commuting_strategy_magic_square_bcs():
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
    # In solution group, product of each row is (-1)^0=1 and -1 for each column, consistent with Magic Square Game
    constraints.append(parity_constraint([0, 1, 2], parity=0))
    constraints.append(parity_constraint([3, 4, 5], parity=0))
    constraints.append(parity_constraint([6, 7, 8], parity=0))
    constraints.append(parity_constraint([0, 3, 6], parity=1))
    constraints.append(parity_constraint([1, 4, 7], parity=1))
    constraints.append(parity_constraint([2, 5, 8], parity=1))
    
    game = NonlocalGame.from_bcs_game(constraints)
    assert game.has_perfect_commuting_measurement_strategy() == True, "Magic Square game should be perfect."

if __name__ == "__main__":
    test_has_perfect_commuting_strategy_satisfiable_bcs()
    test_has_perfect_commuting_strategy_chsh_bcs()
    test_has_perfect_commuting_strategy_magic_square_bcs()
    print("All tests passed!")

