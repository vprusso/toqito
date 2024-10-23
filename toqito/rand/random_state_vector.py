"""Generates a random state vector."""

import numpy as np

from toqito.perms import swap
from toqito.states import max_entangled


def random_state_vector(
        dim: list[int] | int,
        is_real: bool = False,
        k_param: int = 0,
        seed: int | None = None
) -> np.ndarray:
    r"""Generate a random pure state vector.

    Examples
    ==========

    We may generate a random state vector. For instance, here is an example where we can generate a
    :math:`2`-dimensional random state vector.

    >>> from toqito.rand import random_state_vector
    >>> vec = random_state_vector(2)
    >>> vec # doctest: +SKIP
    array([[0.78645233+0.16239043j],
           [0.56649582+0.18494478j]])


    We can verify that this is in fact a valid state vector by computing the corresponding density matrix of the vector
    and checking if the density matrix is pure.

    >>> from toqito.state_props import is_pure
    >>> dm = vec.conj().T @ vec
    >>> is_pure(dm)
    True

    It is also possible to pass a seed for reproducibility.

    >>> from toqito.rand import random_state_vector
    >>> vec = random_state_vector(2, seed=42)
    >>> vec
    array([[0.54521054+0.60483621j],
           [0.30916633+0.49125839j]])

    We can once again verify that this is in fact a valid state vector by computing the
    corresponding density matrix of the vector and checking if the density matrix is pure.

    >>> from toqito.state_props import is_pure
    >>> dm = vec.conj().T @ vec
    >>> is_pure(dm)
    True



    :param dim: The number of rows (and columns) of the unitary matrix.
    :param is_real: Boolean denoting whether the returned matrix has real
                    entries or not. Default is :code:`False`.
    :param k_param: Default 0.
    :param seed: A seed used to instantiate numpy's random number generator.
    :return: A :code:`dim`-by-:code:`dim` random unitary matrix.

    """
    gen = np.random.default_rng(seed=seed)
    # Schmidt rank plays a role.
    if 0 < k_param < np.min(dim):
        # Allow the user to enter a single number for dim.
        if isinstance(dim, int):
            dim = [dim, dim]

        # If you start with a separable state on a larger space and multiply
        # the extra `k_param` dimensions by a maximally entangled state, you
        # get a Schmidt rank `<= k_param` state.
        psi = max_entangled(k_param, True, False).toarray()

        a_param = gen.random((dim[0] * k_param, 1))
        b_param = gen.random((dim[1] * k_param, 1))

        if not is_real:
            a_param = a_param + 1j * gen.random((dim[0] * k_param, 1))
            b_param = b_param + 1j * gen.random((dim[1] * k_param, 1))

        mat_1 = np.kron(psi.conj().T, np.identity(int(np.prod(dim))))
        mat_2 = swap(
            np.kron(a_param, b_param),
            sys=[2, 3],
            dim=[k_param, dim[0], k_param, dim[1]],
        )

        ret_vec = mat_1 @ mat_2
        return np.divide(ret_vec, np.linalg.norm(ret_vec))

    # Schmidt rank is full, so ignore it.
    ret_vec = gen.random((dim, 1))
    if not is_real:
        ret_vec = ret_vec + 1j * gen.random((dim, 1))
    return np.divide(ret_vec, np.linalg.norm(ret_vec))
