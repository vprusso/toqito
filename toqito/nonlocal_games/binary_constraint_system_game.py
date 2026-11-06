"""Two-player binary constraint system (BCS) game."""

from collections import Counter

import networkx as nx
import numpy as np


def create_bcs_constraints(M: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    r"""Build the truth-table constraints of a parity binary constraint system (BCS).

    Row ``i`` of the binary matrix ``M``, together with the bit ``b[i]``, encodes the parity constraint

    \[
        \sum_{j : M_{ij} = 1} x_j = b_i \pmod 2
    \]

    on the binary variables \(x_1, \ldots, x_n\). This function returns that constraint as a 0/1 array of shape
    ``(2,) * n`` whose entry at index ``(x_1, ..., x_n)`` is 1 exactly when the assignment satisfies the constraint.
    This is the form consumed by `NonlocalGame.from_bcs_game`.

    Args:
        M: A binary matrix of shape ``(m, n)``. Row ``i`` lists the variables appearing in constraint ``i``.
        b: A binary vector of length ``m`` giving the parity of each constraint.

    Returns:
        A list of ``m`` arrays of shape ``(2,) * n``, one truth table per constraint.

    Examples:
        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.nonlocal_games.binary_constraint_system_game import create_bcs_constraints

        M = np.array([[1, 1], [1, 1]], dtype=int)
        b = np.array([0, 1], dtype=int)
        constraints = create_bcs_constraints(M, b)
        print(constraints[0])
        ```

    """
    M = np.asarray(M, dtype=int) % 2
    b = np.asarray(b, dtype=int) % 2
    _, n = M.shape
    # Column `k` of `assignments` is the binary assignment with flat index `k` in the C-order layout of `(2,) * n`.
    assignments = np.indices((2,) * n).reshape(n, -1)
    parities = (M @ assignments) % 2
    return [(parity == bit).astype(int).reshape((2,) * n) for parity, bit in zip(parities, b)]


def generate_solution_group(M: np.ndarray, b: np.ndarray) -> tuple[list[int], list[int]]:
    r"""Generate a bitmask representation for a binary constraint system (BCS) game.

    This function converts each row of the binary matrix ``M`` into an integer bitmask,
    pairing it with the corresponding parity from ``b``. The bitmask representation
    can be useful for analyzing linear system games.

    Notes
    The method used to determine the existence of a perfect commuting strategy was originally introduced
    in [@cleve2016perfect].

    Args:
        M: A binary matrix of shape (m, n).Each row encodes which variables appear in a constraint.
        b: A binary vector of length m.Each entry determines the parity for its corresponding constraint row.

    Returns:
        A list of integer bitmasks.
        A list of parity values.

    Examples:
        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.nonlocal_games.binary_constraint_system_game import generate_solution_group

        M = np.array([[1, 1, 0], [0, 1, 1]])
        b = np.array([0, 1])
        row_masks, parity = generate_solution_group(M, b)
        print("```")
        print(row_masks)  # Output: [3, 6]
        print(parity)     # Output: [0, 1]
        print("```")
        ```

    """
    # Ensure M and b are binary.
    M = np.array(M, dtype=int) % 2
    b = np.array(b, dtype=int) % 2

    # Create an array of powers of 2 for each column: [1, 2, 4, ..., 2^(n-1)].
    powers = 1 << np.arange(M.shape[1])
    return (M * powers).sum(axis=1).astype(int).tolist(), b.astype(int).tolist()


def _is_satisfiable_gf2(M: np.ndarray, b: np.ndarray) -> bool:
    """Decide whether the parity system ``M x = b`` has a solution over GF(2)."""
    row, parity = generate_solution_group(M, b)
    m = len(row)
    n = M.shape[1] if m > 0 else 0

    pivot = 0
    for j in range(n):
        pivot_row = next((r for r in range(pivot, m) if row[r] & (1 << j)), None)
        if pivot_row is None:
            continue
        row[pivot], row[pivot_row] = row[pivot_row], row[pivot]
        parity[pivot], parity[pivot_row] = parity[pivot_row], parity[pivot]
        for i in range(m):
            if i != pivot and (row[i] & (1 << j)):
                row[i] ^= row[pivot]
                parity[i] ^= parity[pivot]
        pivot += 1
        if pivot == m:
            break

    # A zero row with parity one reads 0 = 1.
    return not any(row[r] == 0 and parity[r] == 1 for r in range(m))


def _parity_system_from_truth_tables(constraints: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Recover ``(M, b)`` from truth-table constraints, raising if any constraint is not a parity constraint."""
    n = constraints[0].ndim
    M = np.zeros((len(constraints), n), dtype=int)
    b = np.zeros(len(constraints), dtype=int)
    for i, table in enumerate(constraints):
        M[i] = [np.diff(table, axis=axis).any() for axis in range(n)]
        b[i] = 1 - int(table[(0,) * n])
    expected = create_bcs_constraints(M, b)
    for i, (table, parity_table) in enumerate(zip(constraints, expected)):
        if not np.array_equal(np.asarray(table) == 1, parity_table == 1):
            raise ValueError(
                f"Constraint {i} is not a parity constraint, so the commuting-operator criterion does not apply"
            )
    return M, b


def check_perfect_commuting_strategy(M: np.ndarray, b: np.ndarray) -> bool:
    r"""Determine whether a perfect commuting-operator strategy exists for a linear BCS game.

    A linear binary constraint system ``Mx = b`` over \(\mathrm{GF}(2)\) has a perfect commuting-operator strategy
    exactly when the element \(J\) of its solution group is nontrivial [@cleve2016perfect]. Two cases are decided here.

    1. If the system has a classical solution, that solution is a perfect strategy.
    2. Otherwise, variables appearing in no constraint are dropped and variables appearing in exactly one constraint
       are solved out together with that constraint, which leaves the solution group unchanged. If every remaining
       variable then appears in exactly two constraints, the system is an incidence system of the multigraph whose
       vertices are the constraints and whose edges are the variables. For such systems a perfect strategy exists
       if and only if every connected component with odd total parity is non-planar [@arkhipov2012extending].

    Systems in which some variable still appears in three or more constraints after the reduction are not covered
    by either case, and a `NotImplementedError` is raised for them.

    Args:
        M: A binary matrix of shape ``(m, n)`` representing a system of parity constraints.
        b: A binary vector of length ``m`` giving the right-hand side of each constraint.

    Raises:
        NotImplementedError: If the system is classically unsatisfiable and, after reduction, is not an incidence
            system of a multigraph.

    Returns:
        ``True`` if a perfect commuting-operator strategy exists; otherwise, ``False``.

    Examples:
        The CHSH system ``x + y = 0``, ``x + y = 1`` has no perfect strategy, while the magic square does.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.nonlocal_games.binary_constraint_system_game import check_perfect_commuting_strategy

        print(check_perfect_commuting_strategy(np.array([[1, 1], [1, 1]]), np.array([0, 1])))

        magic_square = np.array(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1],
            ]
        )
        print(check_perfect_commuting_strategy(magic_square, np.array([0, 0, 0, 0, 0, 1])))
        ```

    """
    M = np.asarray(M, dtype=int) % 2
    b = np.asarray(b, dtype=int) % 2
    if _is_satisfiable_gf2(M, b):
        return True

    # Solve out variables that appear in exactly one constraint, together with that constraint.
    supports = {i: set(np.flatnonzero(M[i]).tolist()) for i in range(M.shape[0])}
    while True:
        counts = Counter(variable for support in supports.values() for variable in support)
        free_constraint = next(
            (i for i, support in supports.items() if any(counts[variable] == 1 for variable in support)), None
        )
        if free_constraint is None:
            break
        del supports[free_constraint]

    # A constraint with no variables left reads 0 = b_i; parity one is the relation J = e.
    for i in list(supports):
        if not supports[i]:
            if b[i] == 1:
                return False
            del supports[i]

    counts = Counter(variable for support in supports.values() for variable in support)
    if any(count > 2 for count in counts.values()):
        raise NotImplementedError(
            "The commuting-operator criterion is only decided for systems that reduce to a multigraph incidence "
            "system (each variable in at most two constraints); this system does not."
        )

    graph = nx.Graph()
    graph.add_nodes_from(supports)
    for variable in counts:
        endpoints = [i for i, support in supports.items() if variable in support]
        graph.add_edge(*endpoints)

    for component in nx.connected_components(graph):
        if sum(int(b[i]) for i in component) % 2 == 1 and nx.check_planarity(graph.subgraph(component))[0]:
            return False
    return True
