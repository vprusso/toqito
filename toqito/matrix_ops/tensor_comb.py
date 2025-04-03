"""Compute tensor combination of list of vectors."""

import itertools
import numpy as np
from toqito.matrix_ops import to_density_matrix


def tensor_comb(
    states: list[np.ndarray], 
    k: int, 
    mode: str = "non-injective", 
    density_matrix: bool = True
) -> dict:
    r"""Generate all possible tensor product combinations of quantum states (vectors).

    This function creates a tensor product of quantum state vectors by generating all possible sequences 
    of length `k` from a given list of quantum states, and computing the tensor product for each sequence.

    Supported sequence modes:
        - "non-injective": Allows repetitions in sequences.
        - "injective": Ensures sequences have no repetitions.
        - "diagonal": Generates sequences where all elements are the same.

    Examples
    ========

    Consider the following basis vectors for a 2-dimensional quantum system.

    .. math::
        e_0 = \left[1, 0 \right]^{\text{T}}, \quad e_1 = \left[0, 1 \right]^{\text{T}}.

    **Non-injective mode (default):**
    
    >>> from toqito.matrix_ops import tensor_comb
    >>> import numpy as np
    >>> e_0 = np.array([1, 0])
    >>> e_1 = np.array([0, 1])
    >>> tensor_comb([e_0, e_1], 2, mode="non-injective")
    {(0, 0): array([[1, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]]), 
     (0, 1): array([[0, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]]), 
     (1, 0): array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 0]]), 
     (1, 1): array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])}

    **Injective mode (no repetitions):**
    
    >>> tensor_comb([e_0, e_1], 2, mode="injective")
    {(0, 1): array([[0, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]]), 
     (1, 0): array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 0]])}

    **Diagonal mode (repeated elements only):**

    >>> tensor_comb([e_0, e_1], 2, mode="diagonal")
    {(0, 0): array([[1, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]]), 
     (1, 1): array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])}

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :raises ValueError: If `k` is greater than the number of states in "injective" mode.
    :param states: A list of quantum state vectors represented as numpy arrays.
    :param k: The length of the sequence for generating tensor products.
    :param mode: Type of sequence to generate ("non-injective", "injective", "diagonal").
    :param density_matrix: Whether to return density matrices (True) or pure state vectors (False).
    :return: A dictionary where:

        - Keys represent sequences (as tuples) of quantum state indices.
        - Values are density matrices (or state vectors) corresponding to the tensor product of the state vectors.

    """
    if not states:
        raise ValueError("Input list of states cannot be empty.")
    
    if mode not in {"injective", "non-injective", "diagonal"}:
        raise ValueError("`mode` must be 'injective', 'non-injective', or 'diagonal'.")

    if mode == "injective" and k > len(states):
        raise ValueError("k must be less than or equal to the number of states for injective sequences.")

    if mode == "injective":
        sequences = list(itertools.permutations(range(len(states)), k))
    elif mode == "non-injective":
        sequences = list(itertools.product(range(len(states)), repeat=k))
    elif mode == "diagonal":
        sequences = [(i,) * k for i in range(len(states))]

    sequences_of_states = {}
    for seq in sequences:
        state_sequence = [states[i] for i in seq]
        sequence_tensor_product = np.array(state_sequence[0])
        for state in state_sequence[1:]:
            sequence_tensor_product = np.kron(sequence_tensor_product, state)

        if density_matrix:
            sequences_of_states[seq] = to_density_matrix(sequence_tensor_product)
        else:
            sequences_of_states[seq] = sequence_tensor_product

    return sequences_of_states
