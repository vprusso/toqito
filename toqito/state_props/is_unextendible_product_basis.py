"""Check if a set of states form an unextendible product basis."""
import numpy as np
from scipy.linalg import null_space

from itertools import permutations
from more_itertools import set_partitions

from toqito.state_props import is_product
from toqito.matrix_ops import tensor


def is_unextendible_product_basis(vecs: list | np.ndarray, dims: list | np.ndarray) -> tuple[bool, np.ndarray | None]:
    r"""
    Check if a set of vectors form an unextendible product basis (UPB) [UPB99]_.

    Consider a multipartite quantum system :math:`\mathfrak{H} = \bigotimes_{i=1}^{m} \mathfrak{H}_{i}
    with :math:`m` parties with respective dimensions :math:`d_i, i=1,2,..,m`. An (incomplete orthogonal)
    product basis (PB) is a set :math:`S` of pure orthogonal product states spanning a proper subspace
    :math:`H_S` of :math:`H`. An unextendible product basis (UPB) is a PB whose complementary subspace
    :math:`H_S-H` contains no product state.

    This function is inspired from :code:`IsUPB` in QETLAB.

    Examples
    ==========

    See :func:`toqito.states.tile`.

    All the states together form a UPB:

    >>> import numpy as np
    >>> from toqito.states import tile
    >>> from toqito.state_props import is_unextendible_product_basis
    >>> upb_tiles = np.array([tile(i) for i in range(5)])
    >>> dims = np.array([3, 3])
    >>> is_unextendible_product_basis(upb_tiles, dims)
    (True, None)

    However, the first 4 do not:

    >>> import numpy as np
    >>> from toqito.states import tile
    >>> from toqito.state_props import is_unextendible_product_basis
    >>> non_upb_tiles = np.array([tile(i) for i in range(4)])
    >>> dims = np.array([3, 3])
    >>> is_unextendible_product_basis(non_upb_tiles, dims)
    (False, array([-0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -0.00000000e+00,
        0.00000000e+00,  0.00000000e+00, -1.11022302e-16,  7.07106781e-01,
        7.07106781e-01]))

    The orthogonal state is given by

    .. math::
        \frac{1}{\sqrt{2}} |2\rangle \left( |1\rangle + |2\rangle \right)

    References
    ==========
    .. [UPB99] Bennett, Charles H., et al.
        "Unextendible product bases and bound entanglement."
        Physical Review Letters 82.26 (1999): 5385.
        https://arxiv.org/abs/quant-ph/9808030

    :raises ValueError: If product of dimensions does not match the size of a vector.
    :raises ValueError: If at least one vector is not a product state.
    :param vecs: The list of states.
    :param dims: The list of dimensions.
    :return: Returns a tuple. The first element is :code:`True` if input is a UPB and
            :code:`False` otherwise. The second element is a witness (a product state
            orthogonal to all the input vectors) if the input is a PB and :code:`None`
            otherwise.
    """
    # Some error handling:
    if isinstance(vecs, list):
        vecs = np.array(vecs)
    if isinstance(dims, list):
        dims = np.array(dims)

    if np.prod(dims) != vecs.shape[1]:
        raise ValueError("Product of dimensions does not equal the size of each vector")

    if not all(is_product(vec, dims)[0] for vec in vecs):
        raise ValueError("At least one vector is not a product state")
    
    # Number of parties (m)
    num_parties = dims.shape[0]

    # Number of vectors (n)
    num_vecs = vecs.shape[0]

    # If n<m vectors are provided, then we cannot generate set partitions,
    # so it is not a UPB. We will extend the set with m-n null vectors and
    # run the same algorithm.
    if num_vecs < num_parties:
        vecs = np.append(vecs, np.zeros(shape=(num_parties - num_vecs, *vecs.shape[1:])), axis=0)
        num_vecs = vecs.shape[0]

    # Split products
    vecs_split = np.array([is_product(vec, dims)[1] for vec in vecs])

    # Acquire generator to m-partitions of [0,n-1]
    parts_unordered = set_partitions(list(range(num_vecs)), num_parties)

    for part_unordered in parts_unordered:
        for part_ordered in permutations(part_unordered):
            # Witness vectors
            wit = []
            witness_found = True
            for i in range(num_parties):
                # For the i-th party, acquire the matrix
                mat = np.stack([vecs_split[col , i, :] for col in part_ordered[i]])
                # Find the basis of the null space
                null_basis = null_space(mat)
                # If null space is empty then break
                if null_basis.shape[1] == 0:
                    witness_found = False
                    break
                # If null space is non-empty, add a basis vector of null space to witness
                wit.append(null_basis[:, 0])
            # If witness was found, then it is not a UPB, return tensor product of witness vectors
            if witness_found:
                # If wit is empty, tensor returns None
                return (False, tensor(wit))
    
    # If no witness was found, it is a UPB
    return (True, None)

