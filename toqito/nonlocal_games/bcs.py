import numpy as np
import networkx as nx
from toqito.nonlocal_games.nonlocal_game import NonlocalGame

def generate_solution_group(M: np.ndarray, b: np.ndarray):
    M = np.array(M, dtype=int) % 2
    b = np.array(b, dtype=int) % 2
    m, n = M.shape
    # store each row in integer
    row = []
    for i in range(m):
        h = 0
        for j in range(n):
            if M[i, j] == 1:
                h |= (1 << j)
        row.append(h)
    # Parity bits list
    parity=[]
    for i in b:
        parity.append(int(i))
    return row, parity

def check_perfect_commuting_strategy(M: np.ndarray, b: np.ndarray) -> bool:
    """
    Determines whether a perfect commuting-operator strategy exists for the BCS game given by Mx = b.
    Returns True if a perfect strategy exists (i.e., J != e in the solution group), False otherwise.
    """
    # Convert to binary representation
    row, parity = generate_solution_group(M, b)
    m = len(row)
    store = [1 << i for i in range(m)]
    # Eliminate variables one by one
    pivot = 0
    n = max(M.shape[1], 0) if m > 0 else 0
    for j in range(n):
        # Find a row at or below 'pivot' with a 1 in this column
        pivot_row = None
        for i in range(pivot, m):
            if row[i] & (1 << j):
                pivot_row = i
                break
        if pivot_row is None:
            continue
        # Swap pivot row into current pivot position
        row[pivot], row[pivot_row] = row[pivot_row], row[pivot]
        parity[pivot], parity[pivot_row] = parity[pivot_row], parity[pivot]
        store[pivot], store[pivot_row] = store[pivot_row], store[pivot]
        # Eliminate this bit from all other rows
        for i in range(m):
            if i != pivot and (row[i] & (1 << j)):
                row[i] ^= row[pivot]      
                parity[i] ^= parity[pivot]   
                store[i] ^= store[pivot]   
        pivot += 1
        if pivot == m:
            break
    # Look for a row that ended up as 0 = 1 (contradiction)
    contradiction = None
    for i in range(m):
        if row[i] == 0 and parity[i] == 1:
            contradiction = store[i]
            break
    if contradiction is None:
        # No contradiction found: system is satisfiable classically, hence a perfect strategy exists (trivial case)
        return True
    # Find the rows contributed to contradiction in classical case
    nodes=[]
    for i in range(m):
        if (contradiction >> i) & 1:
            nodes.append(i)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            k, l = nodes[i], nodes[j]
            # Check common variables between these constraints
            if row[k] & row[l]:
                G.add_edge(k, l)
    # Check for cycle in the induced subgraph
    has_cycle = False
    if len(nodes) > 0:
        if len(nx.cycle_basis(G)) > 0:
            has_cycle = True
    # This is to check if the nontrivial J exists, consistent with lemma 9.
    return has_cycle

class BCSNonlocalGame(NonlocalGame):
    """
    Subclass of toqito's NonlocalGame for binary constraint system games.
    Integrates the group-theoretic verification of a perfect commuting-operator strategy.
    """
    def __init__(self, prob: np.ndarray, pred: np.ndarray, reps: int):
        self.prob=prob
        self.pred=pred
        self.reps=reps
        self._perfect = False 
    
    def has_perfect_commuting_measurement_strategy(self, tol: float = 1e-6) -> bool:
        return self._perfect
    
    @classmethod
    def from_bcs_game(cls, M: np.ndarray, b: np.ndarray = None):
        M = np.array(M, dtype=int)
        if b is None: #if b is not input, then just let b to be 0 vector with length as rows of M
            b = np.zeros(M.shape[0], dtype=int)
        else:
            b = np.array(b, dtype=int)
        # Determine if a perfect commuting-operator strategy exists
        perfect = check_perfect_commuting_strategy(M, b)
        # Set Probability and pred simply, we focus on strategy exists
        game = cls(np.ones((1, 1)), np.ones((1, 1), dtype=bool), reps=1)
        game._perfect = perfect
        return game








