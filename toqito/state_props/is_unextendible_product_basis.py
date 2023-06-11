"""Check if a collection of states is an Unextendable Product Basis (UBP)."""

from itertools import combinations
import numpy as np
from scipy.linalg import null_space


def item_partitions(items: list, parts: int, sizes: list[int] = None) -> list:
    r"""
    Constructs all partitions of a given list into a specified number of parts with given sizes.

    This function was adapted from QETLAB [Joh16]_.

    References
    ==========
    .. [Joh16] Nathaniel Johnston.
        "QETLAB: A MATLAB toolbox for quantum entanglement"
        http://www.qetlab.com

    :param items: The list of items to partition.
    :param parts: The number of parts in the partition.
    :param sizes: A list of positive integers specifying the minimum sizes of the parts.
    :return: A list of partitions.
    """

    # Set optional argument default for the size of each part of the partition to be at least 1.
    if sizes == None:
        sizes = [1] * parts

    # Trivial case when the number of parts of the partitions is 1.
    if parts == 1:
        return [[items]]

    # Initialize and construct a list of partitions recusively.
    partitions = []
    # Minimum and maximum number of choices for the first part of a partition.
    min_choices = sizes[0]
    max_choices = len(items) - sum(sizes[i] for i in range(1, parts))
    for j in range(min_choices, max_choices + 1):
        # Start by choosing items for the first part of the partition.
        first_part = combinations(items, j)
        # Loop over all possible choices for the first part.
        for part1 in first_part:
            # Get the unchosen items from the first part.
            unchosen_items = list(set(items) - set(part1))
            # Construct the remaining parts of the partition recursively.
            # The other parts are another partition of the unchosen items with just one less parts.
            # Remove the first item of the original sizes list.
            other_parts = item_partitions(unchosen_items, parts - 1, sizes[1:])
            # Loop over all possible partitions for the second part of the partition.
            for part2 in other_parts:
                # Construct full partition by adding the second part to the first part.
                partition = [list(part1)] + part2
                # Append the partition to the partitions list
                partitions.append(partition)
    return partitions


def is_unextendible_product_basis(local_states_list: list[np.ndarray]):
    r"""
    Determine if a collection of states is an Unextendible Product Basis (UBP) [UPB99]_.

    This function was adapted from QETLAB [Joh16]_.

    References
    ==========
    ..  [UPB99] Bennett, Charles H., et al.
        "Unextendible product bases and bound entanglement."
        Physical Review Letters 82.26 (1999): 5385.
        https://arxiv.org/abs/quant-ph/9808030
    .. [Joh16] Nathaniel Johnston.
        "QETLAB: A MATLAB toolbox for quantum entanglement"
        http://www.qetlab.com

    :raises ValueError: If list elements are not of type numpy.ndarray with two dimensions and the same number of columns.
    :param local_states_list: The list of states to check.
    :return: :code:`(True, None)` if states form an a UPB and :code:`(False, wittness)` otherwise.
    """

    # Input error handling
    if len(local_states_list) == 0:
        raise ValueError("Input must be a nonempty list.")
    for party, states in enumerate(local_states_list):
        if not isinstance(states, np.ndarray):
            raise ValueError(
                "Input list elements must be arrays of type numpy.ndarray."
            )
        if not len(states.shape) == 2:
            raise ValueError("Input list arrays must be 2 dimensional.")
        if states.shape[0] == 0 or states.shape[1] == 0:
            raise ValueError(
                "Input list arrays must have nonzero components in each dimension."
            )
        if party == 0:
            num_cols_first_party = states.shape[1]
        if not states.shape[1] == num_cols_first_party:
            raise ValueError("Input list arrays must have the same number of columns.")

    # Number of parties of the UPB
    num_parties = len(local_states_list)
    # Number of states of the UPB.
    num_states = local_states_list[0].shape[1]
    # A list of local dimensions of respective parties.
    local_dimensions = [party.shape[0] for party in local_states_list]
    # A list that indexes the states for constructing the partitions.
    state_index = [i for i in range(num_states)]

    # Get a list of all permissibe partitions subject to local dimension constraints.
    partitions = item_partitions(
        state_index, num_parties, [dim - 1 for dim in local_dimensions]
    )
    # The number of permissible partitions.
    num_partitions = len(partitions)

    # The remaining code consists of two Blocks "A" and "B" coresponding to two different cases.
    # Block "A" runs if there are no partitions to search.
    # Block "B" runs if there are partitions to search.

    # BLOCK "A" begins here.
    # If there are no partitions, then we know it is not a UPB right away.
    # The following block constructs a witness state
    if num_partitions == 0:
        # Set the boolean for not a UPB
        isUPB = False
        # Initialize a list of empty lists for each party which will contain a local state.
        orth_states = [[]] * num_parties
        # Initialize a count for constructing the orthogonal witness state.
        num_orth_states = 0

        # Loop over each party and construct a local state for the witness.
        # Note: The loop range does not have to be reversed. 
        # A different but valid wittness state is constructed either way.
        # We loop in reverse here to match the QETLAB implementation.
        for party in reversed(range(num_parties)):
            # The party's local witness state depends on if the count is less than the number of states.
            if num_orth_states >= num_states:
                # In this case we trivially construct a computational basis vector.
                orth_state = np.zeros((local_dimensions[party], 1))
                orth_state[party][0] = 1
                orth_states[party] = orth_state
            else:
                # In this case we choose some input states for the party and find another orthogonal state as witness.
                # Number of input states to choose
                more_orth_states = min(
                    [local_dimensions[party] - 1, num_states - num_orth_states]
                )
                # A list of the state indices in the corresponding range.
                local_choices = [num_orth_states + i for i in range(more_orth_states)]
                # Get array of local states chosen by the party.
                local_states = local_states_list[party][:, local_choices]
                # Get the orthogonal complement of chosen local states using Scipy's linalg.null_space.
                # Note: We transpose the array since the original arrays states are given by columns.
                # The null_space() function returns a list of vectors that span the null space.
                local_orth_space = null_space(local_states.T)
                # Check if the null space trivial (0 dimensional) or not.
                if local_orth_space.size == 0:
                    # If so, append empty list for that party.
                    orth_states[party] = []
                else:
                    # If the null space is not trivial, pick any state from the null space.
                    # Here we just choose the first state returned in the null space basis.
                    # We cast the state as a column vector of type numpy.ndarray.
                    local_orth_state = np.array([local_orth_space[:, 0]]).T
                    # Add the local state to the list.
                    orth_states[party] = local_orth_state
                # Increment the count.
                num_orth_states += more_orth_states
        # Set the returned witness variable to the constructed list of orthogonal states.
        witness = orth_states
        # Return output.
        return (isUPB, witness)
        # BLOCK "A" ends here.

    # Block "B" starts here.
    # Loop over all partitions.
    for partition in partitions:
        # Initialize an empty list for the orthogonal witness state.
        orth_states = []
        # Initialize a counter.
        num_orth_states = 1
        # Loop over the local states of the respective parties.
        for party, states in enumerate(local_states_list):
            # Choose state indices given by the party's part of the partition.
            local_choices = partition[party]
            # Get array of the chosen states for the party
            local_states = states[:, local_choices]
            # Get the orthogonal complement of chosen local states using Scipy's linalg.null_space.
            # Note: We transpose the array since the original arrays states are given by columns.
            # The null_space() function returns a list of vectors that span the null space.
            local_orth_space = null_space(local_states.T)
            # Check if the null space trivial (0 dimensional) or not.
            if local_orth_space.size == 0:
                # If so, there are no local orthogonal states for the party.
                num_local_orth_states = 0
            else:
                # Get the dimension of local orthogonal space.
                num_local_orth_states = local_orth_space.shape[1]
            # Increase the count.
            # Note: We multiply since space dimensions are multiplicative over tensor products.
            num_orth_states *= num_local_orth_states
            # Break our of the inner loop if there are no orthogonal states.
            if num_orth_states == 0:
                break
            # If the null space is not trivial, pick any state from the null space.
            # Here we just choose the first state returned in the null space basis.
            # We cast the state as a column vector of type numpy.ndarray.
            local_orth_state = np.array([local_orth_space[:, 0]]).T
            orth_states.append(local_orth_state)
        # If there are orthogonal states, we have our counter example.
        if num_orth_states >= 1:
            # Set the returned boolean for not a UPB.
            isUPB = False
            # Set the returned witness variable to the constructed list of orthogonal states.
            witness = orth_states
            # Return output.
            return (isUPB, witness)

    # If we reach this point, we managed to search over all partitions without finding a counter example.
    # Set the returned boolean for UPB.
    isUPB = True
    # There is no witness in this case, so we return None.
    witness = None
    # Return output.
    return (isUPB, witness)
    # Block "B" ends here.
