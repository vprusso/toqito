import numpy as np
import networkx as nx
from toqito.nonlocal_games.nonlocal_game import NonlocalGame

def bcs(M, b):
    """
    Constructs the BCS constraints from matrix M and vector b.
    
    """
    m, n = M.shape
    constraints = []
    for i in range(m):
        rhs = (-1) ** b[i]
        constraint = np.concatenate((M[i], np.array([rhs])))
        constraints.append(constraint)
    return constraints


def generate_solution_group(M: np.ndarray, b: np.ndarray):
    """
    Constructs the solution group structure for the BCS game in a more vectorized manner.
    
    Each row in M is converted into a bitmask by summing powers of 2 for columns
    where M[i, j] == 1. The parity bits are taken directly from b.
    
    Returns:
        row_masks (List[int]): Each integer is a bitmask representing one row of M.
        parity (List[int]): Each element is 0 or 1, taken from b.
    """
    # Ensure M and b are binary (0/1)
    M = np.array(M, dtype=int) % 2
    b = np.array(b, dtype=int) % 2
    
    # Create an array of powers of 2 for each column: [1, 2, 4, ..., 2^(n-1)]
    powers = 1 << np.arange(M.shape[1])  # e.g. if n=3 => [1, 2, 4]
    
    return (M * powers).sum(axis=1).astype(int).tolist(), b.astype(int).tolist()

def check_perfect_commuting_strategy(M: np.ndarray, b: np.ndarray) -> bool:
    """
    Determines whether a perfect commuting-operator strategy exists for the BCS game given by Mx = b.
    Returns True if a perfect strategy exists (i.e., J != e in the solution group), False otherwise.

    This function:
      1. Converts M, b into bitmasks (row, parity).
      2. Performs Gaussian elimination over GF(2).
      3. Checks for a row of form 0 = 1 (contradiction).
      4. Builds a graph of contradictory constraints and checks for a cycle.

    A cycle indicates a nontrivial J, so a perfect strategy exists.
    """
    row, parity = generate_solution_group(M, b)
    m = len(row)
    combo = [1 << i for i in range(m)]

    pivot = 0
    n = M.shape[1] if m > 0 else 0

    # Perform Gaussian elimination in GF(2)
    for j in range(n):
        pivot_row = next((r for r in range(pivot, m) if row[r] & (1 << j)), None)
        if pivot_row is None:
            continue
        row[pivot], row[pivot_row] = row[pivot_row], row[pivot]
        parity[pivot], parity[pivot_row] = parity[pivot_row], parity[pivot]
        combo[pivot], combo[pivot_row] = combo[pivot_row], combo[pivot]

        for i in range(m):
            if i != pivot and (row[i] & (1 << j)):
                row[i] ^= row[pivot]
                parity[i] ^= parity[pivot]
                combo[i] ^= combo[pivot]

        pivot += 1
        if pivot == m:
            break

    # Check for contradiction: a row with 0 = 1
    contradiction = next((combo[r] for r in range(m) if row[r] == 0 and parity[r] == 1), None)
    if contradiction is None:
        # No contradiction → classically satisfiable → perfect strategy
        return True

    # Build graph of constraints that contributed to the contradiction
    nodes = [r for r in range(m) if (contradiction >> r) & 1]
    G = nx.Graph()
    G.add_nodes_from(nodes)

    # Add edge if two rows share a variable
    edges = [
        (u, v)
        for i, u in enumerate(nodes)
        for v in nodes[i+1:]
        if row[u] & row[v]
    ]
    G.add_edges_from(edges)

    # If there's a cycle, a perfect strategy exists
    return bool(nx.cycle_basis(G))

def nonlocal_game_from_constraints(constraints) -> NonlocalGame:
    """
    Constructs a NonlocalGame using Toqito's built-in from_bcs_game function, but
    with input given as a list of constraints (each a numpy array that is a row of M with the RHS appended).
    
    This function converts the constraints back to M and b:
      - For each constraint, the last element is the RHS: if it equals 1, then b=0; if it equals -1, then b=1.
      - The remaining elements form the row of M.
    
    It then calls NonlocalGame.from_bcs_game with the constraints and sets the internal _perfect
    flag using check_perfect_commuting_strategy.
    """
    M_list = []
    b_list = []
    for c in constraints:
        # c is a 1D numpy array: first len(c)-1 entries are M's row, last is RHS.
        M_list.append(c[:-1])
        # If the last element equals 1, then (-1)^b = 1 implies b=0; if equals -1, then b=1.
        b_list.append(0 if c[-1] == 1 else 1)
    M_array = np.array(M_list, dtype=int)
    b_array = np.array(b_list, dtype=int)
    
    perfect = check_perfect_commuting_strategy(M_array, b_array)
    # Use the built-in from_bcs_game function from NonlocalGame:
    game = NonlocalGame.from_bcs_game(constraints, reps=1)
    game._perfect = perfect
    return game
