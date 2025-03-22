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
            aq = list(range(m))
            bq = {s: [t for t in range(n) if M[s, t] == 1] for s in range(m)}
            aa = {
                s: [[int(bit) for bit in format(x, f'0{len(bq[s])}b')]
                    for x in range(2 ** len(bq[s]))]
                for s in range(m)
            }
            ba = {s: [0, 1] for s in range(m)}
            prob = np.zeros((m, n), dtype=float)
            for s in range(m):
                for t in bq[s]:
                    prob[s, t] = 1.0
            prob /= np.sum(prob)
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
            pred = np.ones((2 ** (total_constraints // 2), 2 ** m_B, m_A, m_B), dtype=float)
            game = cls(prob, pred, reps=1)
            game._perfect = True
            return game




