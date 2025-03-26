import numpy as np
import networkx as nx
from toqito.nonlocal_games.nonlocal_game import NonlocalGame
from typing import List

def create_bcs_constraints(M: np.ndarray, b: np.ndarray) -> List[np.ndarray]:
    r"""
    Construct a list of constraints for a binary constraint system (BCS) game.

    Each row of ``M`` is concatenated with the value :math:`(-1)^{b[i]}` to
    form a 1D NumPy array. The resulting list of arrays can then be passed
    to :meth:`toqito.nonlocal_games.nonlocal_game.NonlocalGame.from_bcs_game`.

    This construction is based on binary constraint system games, which are
    closely related to the linear system games studied in :cite:`cleve2017perfect`.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from binary_constraint_system_game import create_bcs_constraints
        from toqito.nonlocal_games.nonlocal_game import NonlocalGame

        M = np.array([[1, 1], [1, 1]], dtype=int)
        b = np.array([0, 1], dtype=int)
        constraints = create_bcs_constraints(M, b)

        game = NonlocalGame.from_bcs_game(constraints)

    References
    ----------
    .. bibliography::
       :filter: docname in docnames

    :param M: A binary matrix of shape ``(m, n)`` specifying which variables appear
              in each of the ``m`` constraints. Each entry is either 0 or 1.
    :param b: A binary vector of length ``m``. For each row ``i``, the exponent
              :math:`b[i]` determines whether the right-hand side is +1 or -1,
              via :math:`(-1)^{b[i]}`.
    :return: A list of 1D NumPy arrays, where each array is of length ``n+1``.
             The first ``n`` entries correspond to a row of ``M``, and the last
             entry is :math:`\pm 1` depending on ``b[i]``.
    :rtype: List[np.ndarray]
    """
    m, n = M.shape
    constraints = []
    for i in range(m):
        rhs = (-1) ** b[i]
        row_plus_rhs = np.concatenate((M[i], np.array([rhs])))
        constraints.append(row_plus_rhs)
    return constraints

def generate_solution_group(M: np.ndarray, b: np.ndarray):
    r"""
    Construct the solution group structure for the BCS game in a vectorized manner.

    This method supports analyzing the structure of linear system games, as discussed in
    :cite:`cleve2017perfect`, by representing constraints as bitmasks.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from binary_constraint_system_game import generate_solution_group

        M = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)
        b = np.array([0, 1], dtype=int)
        row_masks, parity = generate_solution_group(M, b)
        print(row_masks)  # e.g. [3, 6]
        print(parity)     # [0, 1]

    References
    ----------
    .. bibliography::
       :filter: docname in docnames

    :param M: A binary matrix of shape ``(m, n)``.
    :param b: A binary vector of length ``m``.
    :return: A tuple ``(row_masks, parity)``, where:
             * ``row_masks`` is a list of integers (bitmasks).
             * ``parity`` is a list of integers (0 or 1).
    """
    # Ensure M and b are binary (0/1)
    M = np.array(M, dtype=int) % 2
    b = np.array(b, dtype=int) % 2
    
    # Create an array of powers of 2 for each column: [1, 2, 4, ..., 2^(n-1)]
    powers = 1 << np.arange(M.shape[1])  # e.g. if n=3 => [1, 2, 4]
    
    return (M * powers).sum(axis=1).astype(int).tolist(), b.astype(int).tolist()

def check_perfect_commuting_strategy(M: np.ndarray, b: np.ndarray) -> bool:
    r"""
    Determine whether a perfect commuting-operator strategy exists for a BCS game.

    This function checks if there is a perfect commuting-operator strategy
    for the system of binary constraints ``M x = b``. The logic is based on
    the correspondence between such strategies and operator solutions of
    non-commutative equations described in :cite:`cleve2017perfect`.

    The function performs the following steps:

    1. Converts ``M`` and ``b`` into bitmask representation via
       :func:`generate_solution_group`.
    2. Performs Gaussian elimination over :math:`\mathrm{GF}(2)`.
    3. Checks for a contradiction of the form ``0 = 1``.
    4. Constructs a graph of involved constraints and checks for cycles.
       A cycle implies a nontrivial group element, which corresponds
       to the existence of a perfect strategy.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from binary_constraint_system_game import check_perfect_commuting_strategy

        M = np.array([[1, 1], [1, 1]], dtype=int)
        b = np.array([0, 1], dtype=int)
        has_strategy = check_perfect_commuting_strategy(M, b)
        print(has_strategy)  # True or False

    References
    ----------
    .. bibliography::
       :filter: docname in docnames

    :param M: A binary matrix of shape ``(m, n)``.
    :param b: A binary vector of length ``m``.
    :return: ``True`` if a perfect commuting-operator strategy exists; otherwise ``False``.
    :rtype: bool
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


