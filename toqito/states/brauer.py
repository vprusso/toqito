"""Brauer states."""
import numpy as np

from toqito.matrix_ops import tensor
from toqito.perms import permute_systems, perfect_matchings
from toqito.states import max_entangled


def brauer(dim: int, p_val: int) -> np.ndarray:
    r"""
    Produce all Brauer states [WikBrauer]_.

    Produce a matrix whose columns are all of the (unnormalized) "Brauer" states: states that are
    the :code:`p_val`-fold tensor product of the standard maximally-entangled pure state on
    :code:`dim` local dimensions. There are many such states, since there are many different ways to
    group the :code:`2 * p_val` parties into :code:`p_val` pairs (with each pair corresponding to
    one maximally-entangled state).

    The exact number of such states is:

    ```python
    np.factorial(2 * p_val) / (np.factorial(p_val) * 2**p_val)
    ```

    which is the number of columns of the returned matrix.

    This function has been adapted from QETLAB.

    Examples
    ==========

    Generate a matrix whose columns are all Brauer states on 4 qubits.

    >>> from toqito.states import brauer
    >>> brauer(2, 2)
    [[1. 1. 1.]
     [0. 0. 0.]
     [0. 0. 0.]
     [1. 0. 0.]
     [0. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]
     [0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 1.]
     [0. 1. 0.]
     [0. 0. 0.]
     [1. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]
     [1. 1. 1.]]

    References
    ==========
    .. [WikBrauer] Wikipedia: Brauer algebra
        https://en.wikipedia.org/wiki/Brauer_algebra

    :param dim: Dimension of each local subsystem
    :param p_val: Half of the number of parties (i.e., the state that this function computes will
                  live in :math:`(\mathbb{C}^D)^{\otimes 2 P})`
    :return: Matrix whose columns are all of the unnormalized Brauer states.
    """
    # The Brauer states are computed from perfect matchings of the complete graph. So compute all
    # perfect matchings first.
    phi = tensor(max_entangled(dim, False, False), p_val)
    matchings = perfect_matchings(2 * p_val)
    num_matchings = matchings.shape[0]
    state = np.zeros((dim ** (2 * p_val), num_matchings))

    # Turn these perfect matchings into the corresponding states.
    for i in range(num_matchings):
        state[:, i] = permute_systems(
            phi, matchings[i, :], dim * np.ones((1, 2 * p_val), dtype=int)[0]
        )
    return state
