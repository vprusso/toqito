"""Check if a collection of states is an Unextendable Product Basis (UBP)."""

from itertools import combinations
import numpy as np
from scipy.linalg import null_space


def item_partitions(items: list, num_parts: int, sizes: list[int] = None) -> list[list]:
    r"""
    Constructs all partitions of a given list into a specified number of parts, with each part of a partition having a specified minimum size.

    A partition of a set :math:`S` into :math:`p` parts is defined as a collection of :math:`p` disjoint sets :math:`P_1, P_2, ..., P_p` such that the union covers the full set :math:`P`:

    .. math::
        S = P_1 \cup P_2 \cup ... \cup P_p.

    In this context, we call the subsets :math:`P_i` the "parts" of the partition of :math:`S`, and the number of elements of a part :math:`P_i` is called the "size" of a part :math:`P_i`.

    Here, a partition of :math:`S` into :math:`p` parts is represented as a ordered list of unordered lists. Members of the ordered outer list index the parts :math:`P_1, P_2, ..., P_p` of the partition, and the 
    unordered inner lists correspond to a particular part :math:`P_i`.  For instance, consider the set :math:`S =\{1,2,3\}`. Then the following are some (but not all) examples of valid partitions:

    .. math::
        \begin{align*}
        A &= [ [1], [2,3,4] ], \\
        B &= [ [1], [3,4,2] ], \\
        C &= [ [1,2], [3,4] ], \\
        D &= [ [3,4], [1,2] ], \\
        \end{align*}

    where partitions :math:`A` and :math:`B` represent the same partition, but partitions :math:`C` and :math:`D` are distinct.

    The following demonstrates usage of :code:`item_partitions()` in various cases.

    Construct all partitions of the list :math`[1, 2, 3]` with :math`2` parts:
    
    >>> # List of items to partition.
    >>> my_list = [1,2,3]
    >>> # Number of parts in the partition.
    >>> num_parts = 2
    >>> # Construct all partitions having 2 parts
    >>> partitions = item_partitions(my_list, 2)
    [[[1], [2, 3]], [[2], [1, 3]], [[3], [1, 2]], [[1, 2], [3]], [[1, 3], [2]], [[2, 3], [1]]]

    Construct all partitions of the list :math`[1, 2, 3]` with :math`2` parts, and with the first part having size at least :math`2`:

    >>> # List of items to partition.
    >>> my_list = [1,2,3]
    >>> # Number of parts in the partition.
    >>> num_parts = 2
    >>> # List of minimum sizes for respective parts.
    >>> sizes = [2,2]
    >>> # Construct all partitions having 2 parts with at least 2 items in the first part.
    >>> partitions = item_partitions(my_list, 2, [2,1])
    [[[1, 2], [3]], [[1, 3], [2]], [[2, 3], [1]]]

    Constructing all partitions of the list :math`[1, 2, 3]` with :math`2` parts, and with both part having size at least :math`2` returns an empty list. This is because there are fewer items in the input list than demanded by the part size constraints: 

    >>> # List of items to partition.
    >>> my_list = [1,2,3]
    >>> # Number of parts in the partition.
    >>> num_parts = 2
    >>> # List of minimum sizes for respective parts.
    >>> sizes = [2,2]
    >>> # Construct all partitions having 2 parts with at least 2 items each part.
    >>> partitions = item_partitions(my_list, 2, [2,2])
    []

    This function was adapted from QETLAB [Joh16]_.

    References
    ==========
    .. [Joh16] Nathaniel Johnston.
        "QETLAB: A MATLAB toolbox for quantum entanglement"
        http://www.qetlab.com

    :raises ValueError: If input `num_parts` or list elements of input `sizes` are not positive integers.
    :param items: A list of unique items to partition.
    :param numparts: The number of parts in the partition.
    :param sizes: A optional list of positive integers specifying the minimum sizes of the parts. If no argument is provided, the minimum size of each part is 1.
    :return: A list of partitions.
    """



    # Input error handling
    if not isinstance(num_parts, int) or num_parts < 1:
        raise ValueError("Input `parts` must be a positive integer.")
    
    # Set optional argument default for the size of each part of the partition to be at least 1.
    if sizes == None:
        sizes = [1] * num_parts

    # More input error handling
    if len(sizes) == 0:
        raise ValueError("Input `sizes` can not be an empty list.")
    if len(sizes) < num_parts:
        raise ValueError("Input `sizes` list must have length `num_parts`.")     
    for size in sizes:
        if not isinstance(size, int) or size < 1:
            raise ValueError("Input `sizes` list must contain only positive integers.")


    # Trivial case when the number of parts of the partitions is 1.
    if num_parts == 1:
        return [[items]]

    # Initialize and construct a list of partitions recusively.
    partitions = []
    # Minimum and maximum number of choices for the first part of a partition.
    min_choices = sizes[0]
    max_choices = len(items) - sum(sizes[i] for i in range(1, num_parts))
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
            other_parts = item_partitions(unchosen_items, num_parts - 1, sizes[1:])
            # Loop over all possible partitions for the second part of the partition.
            for part2 in other_parts:
                # Construct full partition by adding the second part to the first part.
                partition = [list(part1)] + part2
                # Append the partition to the partitions list
                partitions.append(partition)
    return partitions


def is_unextendible_product_basis(local_states_list: list[np.ndarray]) -> tuple[bool, list[np.ndarray] | None]:
    r"""
    Determine if a collection of pure states is an Unextendible Product Basis (UBP) [Ben99]_.

    A UPB is a set of mutually orthogonal product states spanning a proper subspace of a multipartite system whose complementary subspace contains no product state.
        
    Let :math:`\mathcal{H} = \bigotimes_{i=1}^{m} \mathcal{H}_{i}` be an :math:`m`-party system, where :math:`\mathcal{H}_{i} = \mathbb{C}^{d_i}` is a complex Euclidean space of dimension :math:`d_i` of the :math:`i`-th party. Consider a collection :math:`\mathcal{C}=\{|\psi_j\rangle\}_j` of product states of :math:`\mathcal{H}1 expressed as

    .. math::
        \begin{align*}
        |\psi_j\rangle = \bigotimes_{i=1}^{m} |\varphi_{i,j}\rangle,
        \end{align*}

    where :math:`|\varphi_{i,j}\rangle \in \mathcal{H}_i` are local states belonging to the :math:`i`-th party. Then then collection :math:`\mathcal{C}` is a UPB if the following hold.

    (i) The states are mutually orthogonal:

    .. math::
        \langle \psi_j\|\psi_k\rangle = 0 \text{ if} j \neq k.

    (ii) The states span a proper subspace:

    .. math::
        \mathcal{H}_\mathcal{C} = span(|\psi_j\rangle))  \subset \mathcal{H}.

    (iii) The complementary subspace contains no product state:

    .. math::
    |\psi\rangle \in \mathcal{H} - \mathcal{H}_\mathcal{C} \implies |\psi\rangle \text{ is entangled.} 


    Examples
    ==========

    For example, the Tiles UPB is a set of :math:`5` states in the bipartite qutrit space :math:`\mathbb{C}^3\otimes \mathbb{C}^3` given by:

    .. math::
        \begin{align*}
        |\psi_0\rangle &= \tfrac{1}{\sqrt{2}}|0\rangle \otimes \left(|0\rangle - |1\rangle \right), \\
        |\psi_1\rangle &= \tfrac{1}{\sqrt{2}}\left(|0\rangle - |1\rangle \right) \otimes |2\rangle, \\
        |\psi_2\rangle &= \tfrac{1}{\sqrt{2}}|2\rangle \otimes \left(|1\rangle - |2\rangle \right), \\
        |\psi_3\rangle &= \tfrac{1}{\sqrt{2}}\left(|1\rangle - |2\rangle \right) \otimes |0\rangle, \\
        |\psi_4\rangle &= \tfrac{1}{3}\left(|0\rangle + |1\rangle + |2\rangle \right) \otimes \left(|0\rangle + |1\rangle + |2\rangle \right).
        \end{align*}

    The product states of the Tiles UPB can be factored into two sets, :math:`\mathcal{C}_0` and :math:`\mathcal{C}_1`, of local qutrit states over the biparition :math:`\mathbb{C}^3\otimes \mathbb{C}^3` as follows.

    The first party has states :math:`\mathcal{C}_0 = \{|\varphi_{0,j}\rangle\}_j` in :math:`\mathbb{C}^3`:

    .. math::
        \begin{align*}
        |\varphi_{0, 0}\rangle &= |0\rangle , \\
        |\varphi_{0, 1}\rangle &= \tfrac{1}{\sqrt{2}}\left(|0\rangle - |1\rangle \right), \\
        |\varphi_{0, 2}\rangle &= |2\rangle , \\
        |\varphi_{0, 3}\rangle &= \tfrac{1}{\sqrt{2}}\left(|1\rangle - |2\rangle \right), \\
        |\varphi_{0, 4}\rangle &= \tfrac{1}{\sqrt{3}}\left(|0\rangle + |1\rangle + |2\rangle \right).
        \end{align*}

    The second party has states :math:`\mathcal{C}_1 = \{|\varphi_{1,j}\rangle\}_j` in :math:`\mathbb{C}^3`:

    .. math::
        \begin{align*}
        |\varphi_{1, 0}\rangle &= \tfrac{1}{\sqrt{2}}\left(|0\rangle - |1\rangle \right), \\
        |\varphi_{1, 1}\rangle &= |2\rangle,  \\
        |\varphi_{1, 1}\rangle &= \tfrac{1}{\sqrt{2}}\left(|1\rangle - |2\rangle \right), \\
        |\varphi_{1, 3}\rangle &= |0\rangle, \\
        |\varphi_{1, 4}\rangle &= \tfrac{1}{\sqrt{3}}\left(|0\rangle + |1\rangle + |2\rangle \right).
        \end{align*}

    When using :code:`toqito` to determine if a collection of :math:`n` product states :math:`\mathcal{C}=\{|\psi_j\rangle\}_j` of a :math:`m`-party system is a UPB, the states in question must be provided as an ordered list of :math:`m` local collections :math:`\mathcal{C}_i=\{|\varphi_{i,j}\rangle\}_j` comprising each of the product state's local factors (as shown above for the Tiles UPB). 

    The particular data structure used here to represent the local collections :math:`\mathcal{C}_i` of states are two dimensional arrays of type `numpy.ndarray`, where the :math:`j`-th column of each array defines the local state :math:`|\varphi_{i,j}\rangle` of the respective party. In this way, each array has the same number of columns corresponding to the number of product states :math:`n`. Moreover, the array representing the local collection :math:`\mathcal{C}_i` of the :math:`i`-th party must have :math:`d_i` many rows corresponding to the local dimension :math:`d_i` of the :math:`i`-th party.

    Below we verify that the Tiles states are indeed a UPB.

    >>> # Array for the first party's local states of Tiles UPB.
    >>> C_0 = np.array( [[1,  1/np.sqrt(2), 0,             0, 1/np.sqrt(3)],
    >>>                 [0, -1/np.sqrt(2), 0,  1/np.sqrt(2), 1/np.sqrt(3)],
    >>>                 [0,             0, 1, -1/np.sqrt(2), 1/np.sqrt(3)]]
    >>>                 )
    >>> # Array for the second party's local states of Tiles UPB.
    >>> C_1 = np.array( [[ 1/np.sqrt(2), 0,             0, 1,  1/np.sqrt(3)],
    >>>                 [-1/np.sqrt(2), 0,  1/np.sqrt(2), 0, 1/np.sqrt(3)],
    >>>                 [            0, 1, -1/np.sqrt(2), 0,  1/np.sqrt(3)]]
    >>>                 )
    >>> # Construct list of local states of all parties.
    >>> tiles = [C_0, C_1]
    >>> # Verify Tiles is a UPB.
    >>> isUPB, witness = is_unextendible_product_basis(tiles)
    (True, None)

    If any number of states from the Tiles UPB are removed, then the remaining collection of states is no longer a UPB. Here we demonstrate this by removing the :math:`5`-th state from the Tiles UPB.

    >>> # Remove the 5th state from the Tiles UPB
    >>> tiles_state_removed = [C_0[:, 0:4], C_1[:, 0:4]]
    >>> # Verify Tiles with a state removed is not a UPB.
    >>> isUPB, witness = is_unextendible_product_basis(tiles_state_removed)
    (False, [array([[0.],
                    [0.],
                    [1.]]),
             array([[1.11022302e-16],
                    [7.07106781e-01],
                    [7.07106781e-01]])] )
         
    When it exists, the witness state returned is given as an ordered list of column vectors defining a local state for the respective party. The product state resulting from tensoring the local states together is orthogonal to all product states of the input collection. 

    In the example above the output witness is (within numerical precision) given by the two local states :math:`|\phi_0\rangle = |2\rangle` and :math:`|\phi_1\rangle = \frac{1}{\sqrt{2}}(|1\rangle + |2\rangle)` that produce the product state

    .. math::
        |\phi\rangle = \tfrac{1}{\sqrt{2}} |2\rangle \otimes \left(|1\rangle + |2\rangle \right),

    which is orthogonal to all input states considered.

    Note that the when a witness state exists it in general not unique, and the implementation used here only returns one particular reproducible state.

    This function was adapted from QETLAB [Joh16]_.

    References
    ==========
    ..  [Ben99] Bennett, Charles H., et al.
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
