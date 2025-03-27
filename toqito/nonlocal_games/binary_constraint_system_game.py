import numpy as np
import networkx as nx
from toqito.nonlocal_games.nonlocal_game import NonlocalGame


def create_bcs_constraints(M: np.ndarray, b: np.ndarray):
    r"""Construct a list of constraints for a binary constraint system (BCS) game.

    This function builds a list of constraints by concatenating each row of the
    binary matrix ``M`` with a constant term given by :math:`(-1)^{b[i]}`.
    The resulting 1D arrays (one per constraint) can be used with nonlocal game routines.

    Examples
    ==========
    .. code-block:: python

        >>> import numpy as np
        >>> from binary_constraint_system_game import create_bcs_constraints
        >>> from toqito.nonlocal_games.nonlocal_game import NonlocalGame
        >>> M = np.array([[1, 1], [1, 1]], dtype=int)
        >>> b = np.array([0, 1], dtype=int)
        >>> constraints = create_bcs_constraints(M, b)
        >>> game = NonlocalGame.from_bcs_game(constraints)

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param M: A binary matrix of shape ``(m, n)`` defining which variables appear in each constraint.
    :param b: A binary vector of length ``m`` that determines the constant term :math:`(-1)^{b[i]}`.
    :return: A list of 1D NumPy arrays, each of length ``n + 1``, representing a constraint.
    """
    m, n = M.shape
    constraints = []
    for i in range(m):
        rhs = (-1) ** b[i]
        row_plus_rhs = np.concatenate((M[i], np.array([rhs])))
        constraints.append(row_plus_rhs)
    return constraints


def generate_solution_group(M: np.ndarray, b: np.ndarray):
    r"""Generate a bitmask representation for a binary constraint system (BCS) game.

    This function converts each row of the binary matrix ``M`` into an integer bitmask,
    pairing it with the corresponding parity from ``b``. The bitmask representation
    can be useful for analyzing linear system games.

    Examples
    ==========
    .. code-block:: python

        >>> import numpy as np
        >>> from binary_constraint_system_game import generate_solution_group
        >>> M = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)
        >>> b = np.array([0, 1], dtype=int)
        >>> row_masks, parity = generate_solution_group(M, b)
        >>> print(row_masks)
        [3, 6]
        >>> print(parity)
        [0, 1]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param M: A binary matrix of shape ``(m, n)``.
    :param b: A binary vector of length ``m``.
    :return: A tuple containing:
             - A list of integer bitmasks (one per row of ``M``).
             - A list of parity values derived from ``b``.
    """
    # Ensure M and b are binary (0/1)
    M = np.array(M, dtype=int) % 2
    b = np.array(b, dtype=int) % 2

    # Create an array of powers of 2 for each column: [1, 2, 4, ..., 2^(n-1)]
    powers = 1 << np.arange(M.shape[1])
    return (M * powers).sum(axis=1).astype(int).tolist(), b.astype(int).tolist()


def check_perfect_commuting_strategy(M: np.ndarray, b: np.ndarray):
    r"""Determine whether a perfect commuting-operator strategy exists for a BCS game.

    This function checks if the binary constraint system defined by ``Mx = b``
    admits a perfect commuting-operator strategy. It converts the constraints
    to bitmask form, performs Gaussian elimination over :math:`\mathrm{GF}(2)`,
    and examines the resulting constraint graph for cycles that indicate a nontrivial
    solution.

    Examples
    ==========
    .. code-block:: python

        >>> import numpy as np
        >>> from binary_constraint_system_game import check_perfect_commuting_strategy
        >>> M = np.array([[1, 1], [1, 1]], dtype=int)
        >>> b = np.array([0, 1], dtype=int)
        >>> has_strategy = check_perfect_commuting_strategy(M, b)
        >>> print(has_strategy)
        True  # or False

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param M: A binary matrix of shape ``(m, n)``.
    :param b: A binary vector of length ``m``.
    :return: ``True`` if a perfect commuting-operator strategy exists; otherwise, ``False``.
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
        return True

    # Build a graph of constraints that contributed to the contradiction
    nodes = [r for r in range(m) if (contradiction >> r) & 1]
    G = nx.Graph()
    G.add_nodes_from(nodes)

    # Add edges where two constraints share a variable
    edges = [
        (u, v)
        for i, u in enumerate(nodes)
        for v in nodes[i + 1 :]
        if row[u] & row[v]
    ]
    G.add_edges_from(edges)
    return bool(nx.cycle_basis(G))

