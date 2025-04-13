import numpy as np
import networkx as nx
from toqito.nonlocal_games.nonlocal_game import NonlocalGame


def create_bcs_constraints(M: np.ndarray, b: np.ndarray):
    r"""Construct a list of constraints in tensor form for a binary constraint system (BCS) game.

    This function builds a list of constraints by converting each row of the binary matrix
    M of shape (m, n) and the corresponding element of the binary vector b
    into an n-dimensional tensor of shape (2, 2, ..., 2) (one axis per variable).

    The conversion works as follows:
    
      1. For the i-th constraint, compute the constant value as 
         rhs = (-1)**(b[i]).
      2. Create an n-dimensional array (tensor) of shape (2,)*n filled with -rhs.
      3. Compute the index from the first n entries of the i-th row of M by taking each value modulo 2.
      4. Set the tensor element at that index to rhs.

    For example, if 
       M[i] = [1, 1] and b[i] = 0 (so rhs = 1),
    then the tensor is of shape (2, 2) and is created as follows:
    
      - Start with a (2, 2) array filled with -1 (since -rhs = -1):

            [ [-1, -1],
              [-1, -1] ]

      - The index is computed as (1 % 2, 1 % 2) = (1, 1).
      - At position (1, 1), the value is set to 1, resulting in:

            [ [-1, -1],
              [-1,  1] ]

    This tensor now represents the constraint in full detail.

    Examples
    ==========
        >>> import numpy as np
        >>> from binary_constraint_system_game import create_bcs_constraints
        >>> M = np.array([[1, 1], [1, 1]], dtype=int)
        >>> b = np.array([0, 1], dtype=int)
        >>> constraints = create_bcs_constraints(M, b)
        >>> constraints[0].shape
        (2, 2)

    References
    ============
        (See bibliography in relevant documentation)

    :param M: A binary matrix of shape (m, n) defining which variables appear in each constraint.
    :param b: A binary vector of length m that determines the constant term (-1)**(b[i]).
    :return: A list of n-dimensional NumPy arrays (tensors) of shape ((2,)*n) representing each constraint.
    """
    m, n = M.shape
    constraints = []
    for i in range(m):
        rhs = (-1) ** b[i]
        # Create an n-dimensional array of shape (2, 2, ..., 2) filled with -rhs.
        constraint_tensor = np.full((2,) * n, fill_value=-rhs, dtype=int)
        # Compute the index from the binary row M[i]. Assume M[i] contains binary values (0 or 1).
        idx = tuple(M[i] % 2)
        # Set the tensor element at that index to rhs.
        constraint_tensor[idx] = rhs
        constraints.append(constraint_tensor)
    return constraints


def tensor_to_raw(constraint_tensor: np.ndarray) -> np.ndarray:
    r"""Convert a tensor-form constraint back to its raw 1D representation.

    The tensor-form constraint is expected to have shape (2, 2, ..., 2) (n times)
    and to be constructed as follows:

      - The array is initially filled with -rhs, where rhs = (-1)**(b[i]).
      - A unique element at some index is overwritten with rhs.

    This function finds the unique element (the one that appears exactly once)
    and returns a 1D array of length n+1, where the first n entries are the coordinates
    (taken directly from the unique index) and the last element is the unique constant rhs.

    Examples
    ==========
        >>> import numpy as np
        >>> from binary_constraint_system_game import tensor_to_raw
        >>> tensor_constraint = np.array([[-1, -1], [-1, 1]])
        >>> raw = tensor_to_raw(tensor_constraint)
        >>> raw
        array([1, 1, 1])

    :param constraint_tensor: An n-dimensional NumPy array representing a constraint (shape (2,)*n).
    :return: A 1D NumPy array of length n+1 where the first n elements are the coordinates (indices)
             and the last element is the unique constant (rhs).
    """
    values, counts = np.unique(constraint_tensor, return_counts=True)
    if len(values) != 2:
        raise ValueError("Constraint tensor does not have exactly two distinct values.")
    if counts[0] == 1:
        unique_value = values[0]
    elif counts[1] == 1:
        unique_value = values[1]
    else:
        raise ValueError("Constraint tensor does not have a unique element that appears exactly once.")
    unique_idx = np.argwhere(constraint_tensor == unique_value)
    if unique_idx.shape[0] != 1:
        raise ValueError("Expected exactly one occurrence of the unique value in the constraint tensor.")
    idx_tuple = tuple(unique_idx[0])
    raw_constraint = np.array(list(idx_tuple) + [unique_value])
    return raw_constraint


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

