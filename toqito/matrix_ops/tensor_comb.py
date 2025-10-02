"""Compute tensor combination of list of vectors."""

import itertools

import numpy as np

from toqito.matrix_ops import to_density_matrix


def tensor_comb(
    states: list[np.ndarray],
    k: int,
    mode: str = "injective",
    density_matrix: bool = True,
) -> dict[tuple[int, ...], np.ndarray]:
    r"""Generate all possible tensor product combinations of quantum states (vectors).

    This function creates a tensor product of quantum state vectors by generating all possible sequences of length `k`
    from a given list of quantum states, and computing the tensor product for each sequence.

    Given ``n`` quantum states, this function generates :math:`n^k` combinations of sequences of length ``k``, computes
    the tensor product for each sequence, and converts each tensor product to its corresponding density matrix.

    For one definition and usage of a quantum sequence, refer to :footcite:`Gupta_2024_Optimal`.

    Examples
    ========

    Consider the following basis vectors for a 2-dimensional quantum system.

    .. math::
        e_0 = \left[1, 0 \right]^{\text{T}}, e_1 = \left[0, 1 \right]^{\text{T}}.

    We can generate all possible tensor products for sequences of length 2.

    .. jupyter-execute::

     from toqito.matrix_ops import tensor_comb
     import numpy as np

     e_0 = np.array([1, 0])
     e_1 = np.array([0, 1])

     result = tensor_comb([e_0, e_1], 2, mode="injective", density_matrix=True)

     for key, mat in result.items():
         print(f"tensor_comb{key} =\n{mat}\n")

    References
    ==========
    .. footbibliography::


    :raises ValueError: If the input list of states is empty.
    :param states: A list of state vectors.
    :param k: The length of the sequence.
    :param mode: Determines the type of sequences. Default is :code:`"injective"`.
           ``non-injective`` will allow repetitions in sequences,
           ``injective`` will ensures sequences are injective (no repetitions) and
           ``diagonal`` will allow sequences with repeated indices (diagonal elements).
    :param density_matrix: Determines whether the return is a density matrix or a ket.
            Default is ``True``.

    :return: A dictionary where keys are tuples representing sequences of state indices,
        and values are density matrices of the tensor products of the corresponding
        state vectors or tensor products of the corresponding state vectors based on
        input :code:`density_matrix` being either ``True`` or ``False``.

    """
    if not states:
        raise ValueError("Input list of states cannot be empty.")

    if mode not in ("injective", "non-injective", "diagonal"):
        raise ValueError("mode must be injective, non-injective, or diagonal.")

    if mode == "injective" and k > len(states):
        raise ValueError("k must be less than or equal to the number of states for injective sequences.")

    # Generate sequences based on the selected mode.
    if mode == "injective":
        sequences = list(itertools.permutations(range(len(states)), k))
    elif mode == "non-injective":
        sequences = list(itertools.product(range(len(states)), repeat=k))
    else:  # mode == "diagonal"
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
