"""Generates a random pure state vector."""
from typing import List, Union
import numpy as np

from toqito.state.states.max_entangled import max_entangled
from toqito.perms.swap import swap


def random_state_vector(dim: Union[List[int], int],
                        is_real: bool = False,
                        k_param: int = 0) -> np.ndarray:
    """
    Generates a random pure state vector.

    :param dim: The number of rows (and columns) of the unitary matrix.
    :param is_real: Boolean denoting whether the returned matrix has real
                    entries or not. Default is `False`.
    :param k_param: Default 0.
    :return: A `dim`-by-`dim` random unitary matrix.
    """

    # Schmidt rank plays a role.
    if 0 < k_param < np.min(dim):
        # Allow the user to enter a single number for dim.
        if isinstance(dim, int):
            dim = [dim, dim]

        # If you start with a separable state on a larger space and multiply
        # the extra `k_param` dimensions by a maximally entangled state, you
        # get a Schmidt rank `<= k_param` state.
        psi = max_entangled(k_param, True, False)

        a_param = np.random.rand(dim[0]*k_param, 1)
        b_param = np.random.rand(dim[1]*k_param, 1)

        if not is_real:
            a_param = a_param + 1j * np.random.rand(dim[0]*k_param, 1)
            b_param = b_param + 1j * np.random.rand(dim[1]*k_param, 1)

        mat_1 = np.kron(psi.conj().T, np.identity(int(np.prod(dim))))
        mat_2 = swap(np.kron(a_param, b_param),
                     sys=[2, 3],
                     dim=[k_param, dim[0], k_param, dim[1]])

        ret_vec = np.matmul(mat_1, mat_2)
        return np.divide(ret_vec, np.linalg.norm(ret_vec))

    # Schmidt rank is full, so ignore it.
    ret_vec = np.random.rand(dim, 1)
    if not is_real:
        ret_vec = ret_vec + 1j * np.random.rand(dim, 1)
    return np.divide(ret_vec, np.linalg.norm(ret_vec))
