"""Test is_unextendible_product_basis."""

import numpy as np
import pytest

from toqito.state_props.is_unextendible_product_basis import is_unextendible_product_basis
from toqito.state_props.is_unextendible_product_basis import item_partitions
from toqito.matrix_ops import tensor
from toqito.matrix_ops import inner_product
from toqito.state_props import is_mutually_orthogonal
from numpy.linalg import matrix_rank


def tiles_local_state_list():
    """
    Returns a list of arrays representing the local states of the Tiles UPB (unextendible product basis) [Ben99]_.

    The Tiles UPB is a set of $5$ pure product states of the bipartite qutrit space :math:`\mathbb{C}^3\otimes\mathbb{C}^3` given by:

    .. math::
        \begin{align*}
        |\psi_0\rangle &= \tfrac{1}{\sqrt{2}}|0\rangle \otimes \left(|0\rangle - |1\rangle \right), \\
        |\psi_1\rangle &= \tfrac{1}{\sqrt{2}}\left(|0\rangle - |1\rangle \right) \otimes |2\rangle, \\
        |\psi_2\rangle &= \tfrac{1}{\sqrt{2}}|2\rangle \otimes \left(|1\rangle - |2\rangle \right), \\
        |\psi_3\rangle &= \tfrac{1}{\sqrt{2}}\left(|1\rangle - |2\rangle \right) \otimes |0\rangle, \\
        |\psi_4\rangle &= \tfrac{1}{3}\left(|0\rangle + |1\rangle + |2\rangle \right) \otimes \left(|0\rangle + |1\rangle + |2\rangle \right).
        \end{align*}

    The product states of the Tiles UPB can be factored into two sets, :math:`\mathcal{C}_0$ and $\mathcal{C}_1`, of local qutrit states over the biparition :math:`\mathbb{C}^3\otimes \mathbb{C}^3` as follows.

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

    Here, these local collections :math:`\mathcal{C}_i1 of states are represented as two dimensional arrays of type `numpy.ndarray`, where the :math:`j1-th column of each array defines the local state :math:`|\varphi_{i,j}\rangle1 of the respective party. In this way, each array has the same number of columns corresponding to the number of product states :math:`n`.
    """
    tiles_A = np.zeros([3, 5])
    tiles_A[:, 0] = [1, 0, 0]
    tiles_A[:, 1] = [1 / np.sqrt(2), -1 / np.sqrt(2), 0]
    tiles_A[:, 2] = [0, 0, 1]
    tiles_A[:, 3] = [0, 1 / np.sqrt(2), -1 / np.sqrt(2)]
    tiles_A[:, 4] = [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]

    tiles_B = np.zeros([3, 5])
    tiles_B[:, 0] = [1 / np.sqrt(2), -1 / np.sqrt(2), 0]
    tiles_B[:, 1] = [0, 0, 1]
    tiles_B[:, 2] = [0, 1 / np.sqrt(2), -1 / np.sqrt(2)]
    tiles_B[:, 3] = [1, 0, 0]
    tiles_B[:, 4] = [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]

    return [tiles_A, tiles_B]


def genshifts_local_state_list(parties: int):
    """
    Returns a list of arrays representing the local states of the GenShifts UPB (unextendible product basis) for an odd number of parties. DiV00.

    The GenShifts UPBs are a family of UPBs defined for an odd number :math:`m\geq 3` of parties each holding a qubit:
    
    .. math::
        \mathcal{H} = \bigotimes_{i=1}^{m} \mathbb{C}^2.

    The GenShifts UPB for :math:`m=2k-1` parties contains $2k$ product states which are comprised of specially chosen local qubit states. Although there is some freedom in how to define these local states, they must satisfy the following property. These local qubit states, say :math:`|\psi_j\rangle1 for :math:`1\leq j\leq k-11, are chosen such that all are neither identical nor orthogonal to each other or the computational basis states :math:`|0\rangle` and :math:`|1\rangle`. Moreover, states :math:`|\psi_j^\perp\rangle` for :math:`1\leq j\leq k-1` are also chosen to be orthogonal to their respective pairs so that :math:`\langle \psi_j^\perp  |\psi_j\rangle = 0`.

    Having made proper choices for states :math:`|\psi_j\rangle` and :math:`|\psi_j^\perp\rangle`, for :math:`1\leq j\leq k-1`, the product states of the GenShifts UPB for :math:`2k-1` parties are given by:

    .. math::
        \begin{align*}
            |\Psi_0\rangle &= |0, 0, \dots, 0\rangle \\
            |\Psi_1\rangle &= |1,\psi_1, \psi_2, \dots, \psi_{k-1}, \psi_{k-1}^\perp, \dots, \psi_{1}^\perp \rangle \\
            |\Psi_2\rangle &= |\psi_{1}^\perp, 1,\psi_1, \psi_2, \dots, \psi_{k-1}, \psi_{k-1}^\perp, \dots, \psi_{2}^\perp \rangle \\
            &\vdots \\
            |\Psi_{2k-1}\rangle &= |\psi_1, \psi_2, \dots, \psi_{k-1}^\perp, \psi_{k-2}^\perp, \dots, \psi_{1}^\perp , 1\rangle, \\
        \end{align*}

    With the exception of the first state :math:`|\Psi_0\rangle = |0, 0, \dots, 0\rangle`, the rest of the states are given by right-shifted cyclic permutations of :math:`|\Psi_1\rangle`.


    In 'toqito', the particular choices made for the local states :math:`|\psi_j\rangle` and :math:`\psi_j^\perp\rangle` matches the implementation given in `QETLAB` (citeQETLAB). To be precise, consider the qubit states parameterized by an angle value

    .. math::
        |\theta(j)\rangle = \cos(j\frac{\pi}{2k})|0\rangle + \sin(j\frac{\pi}{2k})|1\rangle.

    Then the states :math:`|\theta(j)\rangle1 defined for integers values in the range :math:`1\leq j \leq 2k`, can be uniquely paired as orthogonal states :math:`|\psi_j\rangle` and :math:`|\psi_j^\perp\rangle` given as:

    .. math::
        \begin{align*}
        |\psi_j\rangle &= |\theta(2k-j)\rangle, \\
        |\psi_j^{\perp}\rangle &= |\theta(j)\rangle, 
        \end{align*}

    for :math:`1 \leq j \leq k-1`.

    For example, in the case of :math:`3` parties the GenShifts UPB states are given by the following :math:`4` product states:

    .. math::
        \begin{align*}
            |\Psi_0\rangle &= |0, 0, 0\rangle \\
            |\Psi_1\rangle &= |1, -, + \rangle \\
            |\Psi_2\rangle &= |+, 1, - \rangle \\
            |\Psi_3\rangle &= |-, +, 1 \rangle, \\
        \end{align*}

    where :math:`|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)` and :math:`|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)`.
    """
    if (parties % 2 == 0) or parties == 1:
        raise ValueError("Input must be an odd int greater than 1.")

    k = int((parties + 1) / 2)
    num_states = 2 * k
    upb = [np.zeros([2, num_states]) for _ in range(parties)]
    angles = [(i / k) * np.pi / 2 for i in range(2 * k - 1, k, -1)]
    local_states = [[np.cos(angle), np.sin(angle)] for angle in angles]
    orth_angles = [(i / k) * np.pi / 2 for i in range(1, k)]
    orth_local_states = [[np.cos(angle), np.sin(angle)] for angle in orth_angles]

    state_list = []
    state_list.append([0, 1])
    state_list.extend(local_states)
    state_list.extend(orth_local_states)

    for party in upb:
        party[:, 0] = [1, 0]

    for state in range(1, num_states):
        for i, party in enumerate(upb):
            party[:, state] = state_list[i]
        state_list = state_list[-1:] + state_list[:-1]

    return upb


def test_tiles_mutually_orthogonal():
    tiles = tiles_local_state_list()
    tiles_product_state_list = []
    for i in range(5):
        tiles_product_state = tensor(
            np.array([tiles[0][:, i]]).T, np.array([tiles[1][:, i]]).T
        )
        tiles_product_state_list.append(tiles_product_state)
    res = is_mutually_orthogonal(tiles_product_state_list)
    expected_res = True
    np.testing.assert_equal(res, expected_res)


def test_tiles_incomplete_span():
    tiles = tiles_local_state_list()
    tiles_product_state_matrix = np.zeros([9, 5])
    for i in range(5):
        tiles_product_state = tensor(
            np.array([tiles[0][:, i]]).T, np.array([tiles[1][:, i]]).T
        )
        print(tiles_product_state, np.shape(tiles_product_state))
        tiles_product_state_matrix[:, [i]] = tiles_product_state
    rank = matrix_rank(tiles_product_state_matrix)
    res = rank < 9
    expected_res = True
    np.testing.assert_equal(res, expected_res)


# Test parameter ranges over number of parties of the UPB, which must be odd integer greatar than 1.
@pytest.mark.parametrize("num_parties", [3, 5, 7])
def test_genshifts_mutually_orthogonal(num_parties):
    num_states = num_parties + 1
    genshifts = genshifts_local_state_list(num_parties)
    # Initialize a list to populate with global product states.
    genshifts_product_state_list = []
    for i in range(num_states):
        # Initialize the global product state with first tensor factor.
        genshifts_product_state = np.array([genshifts[0][:, i]]).T
        # Tensor remaining parties' states.
        for party in range(1, num_parties):
            genshifts_product_state = tensor(
                genshifts_product_state, np.array([genshifts[party][:, i]]).T
            )
        genshifts_product_state_list.append(genshifts_product_state)
    res = is_mutually_orthogonal(genshifts_product_state_list)
    expected_res = True
    np.testing.assert_equal(res, expected_res)


# Test parameter ranges over number of parties of the UPB, which must be odd integer greatar than 1.
@pytest.mark.parametrize("num_parties", [3, 5, 7])
def test_genshifts_incomplete_span(num_parties):
    num_states = num_parties + 1
    global_dim = 2**num_parties
    genshifts = genshifts_local_state_list(num_parties)
    # Initialize a matrix to populate with global product states.
    genshifts_product_state_matrix = np.zeros([global_dim, num_states])
    for i in range(num_states):
        # Initialize the global product state with first tensor factor.
        genshifts_product_state = np.array([genshifts[0][:, i]]).T
        # Tensor remaining parties' states.
        for party in range(1, num_parties):
            genshifts_product_state = tensor(
                genshifts_product_state, np.array([genshifts[party][:, i]]).T
            )
        # Set i-th colum of matrix to i-th global state.
        genshifts_product_state_matrix[:, [i]] = genshifts_product_state
    rank = matrix_rank(genshifts_product_state_matrix)
    res = rank < global_dim
    expected_res = True
    np.testing.assert_equal(res, expected_res)


def test_is_unextendible_product_basis_input_empty_list():
    """Empty list as input."""
    with np.testing.assert_raises(ValueError):
        empty_list = []
        is_unextendible_product_basis(empty_list)


def test_is_unextendible_product_basis_input_not_numpy_arrays():
    """List elements are not type numpy.ndarray."""
    with np.testing.assert_raises(ValueError):
        list_of_listarrays = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        is_unextendible_product_basis(list_of_listarrays)


def test_is_unextendible_product_basis_input_arrays_not_two_dimensional():
    """Arrays are not two-dimensional."""
    with np.testing.assert_raises(ValueError):
        array_1D = np.array([1, 2, 3, 4])
        input = [array_1D, array_1D]
        is_unextendible_product_basis(input)


def test_is_unextendible_product_basis_input_arrays_not_same_num_columns():
    """Arrays do not have the same number of columns"""
    with np.testing.assert_raises(ValueError):
        array_1 = np.array([[1, 2], [3, 4]])
        array_2 = np.array([[1], [3]])
        input = [array_1, array_2]
        is_unextendible_product_basis(input)


def test_is_unextendable_product_basis_tiles():
    """Verify if Tiles UPB is a UPB"""
    tiles = tiles_local_state_list()
    res = is_unextendible_product_basis(tiles)
    expected_res = [True, None]
    np.testing.assert_array_equal(res, expected_res)


# Test parameter ranges over number of states removed out of 5 original Tiles UPB states.
@pytest.mark.parametrize("num_states", [1, 2, 3, 4])
def test_is_unextendable_product_basis_tiles_remove_states_false(num_states):
    """Check if Tiles UPB fails to be a UPB when 1, 2, 3, and 4 states are removed."""
    tiles = tiles_local_state_list()
    # Remove some number of states from the Tiles UPB.
    tiles_with_states_removed = [tiles[0][0:, 0:num_states], tiles[1][:, 0:num_states]]
    res = is_unextendible_product_basis(tiles_with_states_removed)
    # We expect these to not be a valid UPB.
    expected_res = False
    np.testing.assert_equal(res[0], expected_res)


# Test parameter ranges over number of states removed out of 5 original Tiles UPB states.
@pytest.mark.parametrize("num_states", [1, 2, 3, 4])
def test_is_unextendable_product_basis_tiles_remove_states_orthogonal_witness(
    num_states,
):
    """Check if witness is orthogonal to Tiles UPB when 1, 2, 3, and 4 states are removed."""
    tiles = tiles_local_state_list()
    # Remove some number of states from the Tiles UPB.
    tiles_with_states_removed = [tiles[0][0:, 0:num_states], tiles[1][:, 0:num_states]]
    # Get the returned witness state.
    witness = is_unextendible_product_basis(tiles_with_states_removed)[1]
    # Construct the global tensor product of the witness state.
    witness_product = tensor(witness[0], witness[1])
    # Here we construct a list of the inner products of the witness with each input Tiles state.
    # Perhaps we can use `toqito.state_props.is_mutually_orthogonal` to test orthogonallity.
    inner_product_list = []
    # Loop over the input Tiles states.
    for i in range(num_states):
        # Construct the global tensor product of the Tiles state.
        # Is there a less convoluted way to do this?
        tiles_product_state = tensor(
            np.array([tiles[0][:, i]]).T, np.array([tiles[1][:, i]]).T
        )
        # Calculate the inner product of the witness state with the Tiles state.
        ip = inner_product(witness_product[:, 0], tiles_product_state[:, 0])
        # Add the inner product to the list.
        inner_product_list.append(ip)
    res = inner_product_list
    # We expect all inner products to be zero since the witness state should be orthogonal.
    expected_res = [0] * num_states
    np.testing.assert_array_almost_equal(res, expected_res)


# Test parameter ranges over number of parties of the UPB, which must be odd integer greatar than 1.
@pytest.mark.parametrize("num_parties", [3, 5, 7])
def test_is_unextendable_product_basis_tiles_GenShifts(num_parties):
    """Verify if GenShifts UPB is a UPB for 3, 5, and, 7 parties"""
    genshifts = genshifts_local_state_list(num_parties)
    res = is_unextendible_product_basis(genshifts)
    expected_res = [True, None]
    np.testing.assert_array_equal(res, expected_res)


def test_item_partitions_num_parts_is_not_int():
    """Number of parts is not of type `int`."""
    with np.testing.assert_raises(ValueError):
        items = [1, 2, 3]
        num_parts = 1.1
        if not isinstance(num_parts, int):
            item_partitions(items, num_parts)


def test_item_partitions_num_parts_is_not_positive_int():
    """Number of parts is not positive integer."""
    with np.testing.assert_raises(ValueError):
        items = [1, 2, 3]
        num_parts = 0
        item_partitions(items, num_parts)


def test_item_partitions_sizes_is_empty_list():
    """List of sizes is empty list."""
    with np.testing.assert_raises(ValueError):
        items = [1, 2, 3]
        num_parts = 2
        sizes = []
        item_partitions(items, num_parts, sizes)


def test_item_partitions_sizes_has_non_int_elements():
    """List of sizes has elements not of type `int`."""
    with np.testing.assert_raises(ValueError):
        items = [1, 2, 3]
        num_parts = 2
        sizes = [1, 1.1]
        item_partitions(items, num_parts, sizes)


def test_item_partitions_sizes_has_non_positive_int_elements():
    """List of sizes has non positive integer elements."""
    with np.testing.assert_raises(ValueError):
        items = [1, 2, 3]
        num_parts = 2
        sizes = [1, 0]
        item_partitions(items, num_parts, sizes)


# Test parameter ranges over number of parts specified in the partitions.
@pytest.mark.parametrize("num_parts", [1, 2, 3])
def test_item_partitions_all_have_same_num_parts(num_parts):
    """Checks if each partition has the same number of parts."""
    items = [1, 2, 3]
    partitions = item_partitions(items, num_parts)
    num_partitions = len(partitions)
    partition_length_list = []
    for partition in partitions:
        partition_length = len(partition)
        partition_length_list.append(partition_length)
    res = partition_length_list
    expected_res = [num_parts] * num_partitions
    np.testing.assert_array_equal(res, expected_res)


# Test parameter ranges over various sizes giving the minimum number of elements in a part of the partitions.
@pytest.mark.parametrize("sizes", [[1, 1], [1, 2], [2, 1]])
def test_item_partitions_parts_have_minimum_size(sizes):
    """Checks if each part of each partition has the required minimum number of elements."""
    items = [1, 2, 3]
    num_parts = 2
    partitions = item_partitions(items, num_parts, sizes)
    satisfied = True
    for partition in partitions:
        for i, part in enumerate(partition):
            if len(part) < sizes[i]:
                satisfied = False
                break
        if not satisfied:
            break
    res = satisfied
    expected_res = True
    np.testing.assert_equal(res, expected_res)


# Test parameter ranges over number of parts of the partition.
@pytest.mark.parametrize("num_parts", [1, 2, 3])
def test_item_partitions_has_valid_partitions(num_parts):
    """Checks if each returned list element is a valid permutation."""
    items = [1, 2, 3]
    partitions = item_partitions(items, num_parts)
    # Boolean flag for valid partition.
    satisfied = True
    for partition in partitions:
        partitioned_set = set()
        # Boolean flag for disjointness of parts.
        is_disjoint = True
        # Boolean flag for union of parts.
        is_complete = False
        for part in partition:
            is_disjoint = partitioned_set.isdisjoint(part)
            if not is_disjoint:
                satisfied = False
                break
            partitioned_set.update(part)
        is_complete = partitioned_set == set(items)
        if not is_complete:
            satisfied = False
            break
    res = satisfied
    expected_res = True
    np.testing.assert_equal(res, expected_res)


if __name__ == "__main__":
    np.testing.run_module_suite()
