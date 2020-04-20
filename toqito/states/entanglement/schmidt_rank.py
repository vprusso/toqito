"""Computes the Schmidt rank of a bipartite vector."""
from typing import List, Union
import numpy as np

from toqito.maps.partial_trace import partial_trace


def schmidt_rank(
    vec: np.ndarray, dim: Union[int, List[int], np.ndarray] = None
) -> float:
    r"""
    Compute the Schmidt rank [WIKSR]_.

    For complex Euclidean spaces :math:`\mathcal{X}` and :math:`\mathcal{Y}`, a
    pure state :math:`u \in \mathcal{X} \otimes \mathcal{Y}` possesses an
    expansion of the form:

    .. math::
        u = \sum_{i} \lambda_i v_i w_i

    where :math:`v_i \in \mathcal{X}` and :math:`w_i \in \mathcal{Y}` are
    orthonormal states.

    The Schmidt coefficients are calculated from

    .. math::
        A = \text{Tr}_{\mathcal{B}}(u^* u).

    The Schmidt rank is the number of non-zero eigenvalues of A. The Schmidt
    rank allows us to determine if a given state is entangled or separable.
    For instance:

        - If the Schmidt rank is 1: The state is separable
        - If the Schmidt rank > 1: The state is entangled.

    Compute the Schmidt rank of the vector `vec`, assumed to live in bipartite
    space, where both subsystems have dimension equal to `sqrt(len(vec))`.

    The dimension may be specified by the 1-by-2 vector `dim` and the rank in
    that case is determined as the number of Schmidt coefficients larger than
    `tol`.

    Examples
    ==========

    Computing the Schmidt rank of the entangled Bell state should yield a value
    greater than one.

    >>> from toqito.states.states.bell import bell
    >>> from toqito.states.entanglement.schmidt_rank import schmidt_rank
    >>> rho = bell(0).conj().T * bell(0)
    >>> schmidt_rank(rho)
    2

    Computing the Schmidt rank of the entangled singlet state should yield a
    value greater than :math:`1`.

    >>> from toqito.states.states.bell import bell
    >>> from toqito.states.entanglement.schmidt_rank import schmidt_rank
    >>> u = bell(2).conj().T * bell(2)
    >>> schmidt_rank(u)
    2

    Computing the Schmidt rank of a separable state should yield a value equal
    to :math:`1`.

    >>> from toqito.core.ket import ket
    >>> from toqito.states.entanglement.schmidt_rank import schmidt_rank
    >>> import numpy as np
    >>> e_0, e_1 = ket(2, 0), ket(2, 1)
    >>> e_00 = np.kron(e_0, e_0)
    >>> e_01 = np.kron(e_0, e_1)
    >>> e_10 = np.kron(e_1, e_0)
    >>> e_11 = np.kron(e_1, e_1)
    >>>
    >>> rho = 1 / 2 * (e_00 - e_01 - e_10 + e_11)
    >>> rho = rho.conj().T * rho
    >>> schmidt_rank(rho)
    1

    References
    ==========
    .. [WIKSR] Wikipedia: Schmidt rank
        https://en.wikipedia.org/wiki/Schmidt_decomposition#Schmidt_rank_and_entanglement

    :param vec: A bipartite vector to have its Schmidt rank computed.
    :param dim: A 1-by-2 vector.
    :return: The Schmidt rank of vector `vec`.
    """
    eps = np.finfo(float).eps
    slv = int(np.round(np.sqrt(len(vec))))

    if dim is None:
        dim = slv

    if isinstance(dim, int):
        dim = np.array([dim, len(vec) / dim])
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * len(vec) * eps:
            raise ValueError(
                "Invalid: The value of `dim` must evenly divide "
                "`len(vec)`; please provide a `dim` array "
                "containing the dimensions of the subsystems"
            )
        dim[1] = np.round(dim[1])

    rho = vec.conj().T * vec
    rho_a = partial_trace(rho, 2)

    # Return the number of non-zero eigenvalues of the
    # matrix that traced out the second party's portion.
    return len(np.nonzero(np.linalg.eigvalsh(rho_a))[0])
