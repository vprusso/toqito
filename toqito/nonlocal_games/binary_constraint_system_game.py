"""Two-player binary constraint system (BCS) game."""

import networkx as nx
import numpy as np


def create_bcs_constraints(M: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    r"""Construct a list of constraints in tensor form for a binary constraint system (BCS) game.

    This function builds a list of constraints by converting each row of the binary matrix
    ``M`` of shape (m, n) and the corresponding element of the binary vector ``b``
    into an n-dimensional tensor of shape (2, 2, ..., 2) (one axis per variable).

    The conversion works as follows:
      1. For the i-th constraint, compute the constant value as ``rhs = (-1)**(b[i])``.
      2. Create an n-dimensional array (tensor) of shape ``(2,)*n`` filled with ``-rhs``.
      3. Compute the index from the first n entries of the i-th row of ``M`` by taking each value modulo 2.
      4. Set the tensor element at that index to ``rhs``.

    For example:
    If ``M[i] = [1, 1]`` and ``b[i] = 0`` (so ``rhs = 1``):
    The tensor is of shape (2, 2) and is created as:

    np.array([[-1, -1], [-1, -1]]

    The index is computed as ``(1 % 2, 1 % 2) = (1, 1)``.
    At position (1, 1), the value is set to 1, resulting in:

    np.array([[-1, -1], [-1, 1]]

    This tensor now represents the constraint in full detail.

    :param M: A 2D binary NumPy array of shape (m, n). Each row represents a constraint on n variables.
    :param b: A 1D binary array of length m. Each entry defines the sign of the constraint.
    :return: A list of NumPy arrays, each of shape ``(2,)*n``. Each tensor represents
             one constraint in tensor form.

    Examples
    ==========
    .. jupyter-execute::

     import numpy as np
     from toqito.nonlocal_games.binary_constraint_system_game import create_bcs_constraints

     M = np.array([[1, 1], [1, 1]], dtype=int)
     b = np.array([0, 1], dtype=int)
     constraints = create_bcs_constraints(M, b)
     constraints[0].shape

    References
    ==========
    .. footbibliography::

    """
    m, n = M.shape
    constraints = []
    for i in range(m):
        rhs = (-1) ** b[i]
        constraint_tensor = np.full((2,) * n, fill_value=-rhs, dtype=int)
        idx = tuple(M[i] % 2)
        constraint_tensor[idx] = rhs
        constraints.append(constraint_tensor)
    return constraints


def generate_solution_group(M: np.ndarray, b: np.ndarray) -> tuple[list[int], list[int]]:
    r"""Generate a bitmask representation for a binary constraint system (BCS) game.

    This function converts each row of the binary matrix ``M`` into an integer bitmask,
    pairing it with the corresponding parity from ``b``. The bitmask representation
    can be useful for analyzing linear system games.

    Notes
    ========
    The method used to determine the existence of a perfect commuting strategy was originally introduced
    in :footcite:`Cleve_2016_Perfect`.

    Examples
    ========
    .. jupyter-execute::

     import numpy as np
     from toqito.nonlocal_games.binary_constraint_system_game import generate_solution_group

     M = np.array([[1, 1, 0], [0, 1, 1]])
     b = np.array([0, 1])
     row_masks, parity = generate_solution_group(M, b)
     print(row_masks)  # Output: [3, 6]
     print(parity)     # Output: [0, 1]

    References
    ==========
    .. footbibliography::

    :param M: A binary matrix of shape (m, n).Each row encodes which variables appear in a constraint.
    :param b: A binary vector of length m.Each entry determines the parity for its corresponding constraint row.
    :return: A list of integer bitmasks.
    :return: A list of parity values.

    """
    # Ensure M and b are binary.
    M = np.array(M, dtype=int) % 2
    b = np.array(b, dtype=int) % 2

    # Create an array of powers of 2 for each column: [1, 2, 4, ..., 2^(n-1)].
    powers = 1 << np.arange(M.shape[1])
    return (M * powers).sum(axis=1).astype(int).tolist(), b.astype(int).tolist()


def check_perfect_commuting_strategy(M: np.ndarray, b: np.ndarray) -> bool:
    r"""Determine whether a perfect commuting-operator strategy exists for a BCS game.

    This function checks if the binary constraint system defined by ``Mx = b``
    admits a perfect commuting-operator strategy. It converts the constraints
    to bitmask form, performs Gaussian elimination over :math:`\mathrm{GF}(2)`,
    and examines the resulting constraint graph for cycles that indicate a nontrivial
    solution.

    :param M: A binary matrix representing a system of parity constraints.
    :param b: A binary vector representing the right-hand side of the constraint equations.
    :return: ``True`` if a perfect commuting-operator strategy exists; otherwise, ``False``.

    Examples
    ==========
    .. jupyter-execute::

     import numpy as np
     from toqito.nonlocal_games.binary_constraint_system_game import check_perfect_commuting_strategy

     M = np.array([[1, 1], [1, 1]])
     b = np.array([0, 1])
     check_perfect_commuting_strategy(M, b)

    References
    ==========
    .. footbibliography::

    """
    row, parity = generate_solution_group(M, b)
    m = len(row)
    combo = [1 << i for i in range(m)]

    pivot = 0
    n = M.shape[1] if m > 0 else 0

    # Perform Gaussian elimination in GF(2):
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

    # Check for contradiction: a row with 0 = 1.
    contradiction = next((combo[r] for r in range(m) if row[r] == 0 and parity[r] == 1), None)
    if contradiction is None:
        return True  # no contradiction → perfect strategy exists.

    # Build the subgraph of nodes involved in a contradiction.
    nodes = [r for r in range(m) if (contradiction >> r) & 1]
    G = nx.Graph()
    G.add_nodes_from(nodes)

    # Add edges where two constraints share a variable.
    edges = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :] if row[u] & row[v]]
    G.add_edges_from(edges)
    return bool(nx.cycle_basis(G))
