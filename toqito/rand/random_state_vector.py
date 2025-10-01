"""Generates a random state vector."""

import numpy as np

from toqito.perms import swap
from toqito.states import max_entangled


def random_state_vector(
    dim: list[int] | tuple[int, ...] | int,
    is_real: bool = False,
    k_param: int = 0,
    seed: int | None = None,
) -> np.ndarray:
    r"""Generate a random pure state vector.

    Randomness model
    ----------------

    We sample entries independently from the standard normal distribution using ``numpy``'s
    ``default_rng``.  If ``is_real`` is ``False`` (default), the imaginary part is sampled in the
    same way and added with the factor :math:`i`; otherwise the vector is real.  The sampled vector
    is normalized to have unit Euclidean norm.  When ``k_param`` is strictly positive, the returned
    state describes a bipartite system of dimensions ``dim`` (or ``[dim, dim]`` if ``dim`` is an
    integer) with Schmidt rank at most ``k_param``.  This is achieved by drawing local factors and
    combining them with a maximally entangled resource state.

    Examples
    ==========

    We may generate a random state vector. For instance, here is an example where we can generate a
    :math:`2`-dimensional random state vector.

    .. jupyter-execute::

     from toqito.rand import random_state_vector

     vec = random_state_vector(2)

     vec

    We can verify that this is in fact a valid state vector by computing the corresponding density
    matrix of the vector and checking if the density matrix is pure.

    .. jupyter-execute::

     from toqito.state_props import is_pure

     dm = vec @ vec.conj().T

     is_pure(dm)

    It is also possible to pass a seed for reproducibility.

    .. jupyter-execute::

     from toqito.rand import random_state_vector

     vec = random_state_vector(2, seed=42)

     vec

    We can once again verify that this is in fact a valid state vector by computing the
    corresponding density matrix of the vector and checking if the density matrix is pure.

    .. jupyter-execute::

     from toqito.state_props import is_pure

     dm = vec @ vec.conj().T

     is_pure(dm)

    :param dim: Either a positive integer giving the total Hilbert-space dimension, or a length-2
        sequence specifying the individual subsystem dimensions for bipartite sampling.
    :param is_real: Boolean denoting whether the returned vector has real entries. Default is
        :code:`False`, which produces complex amplitudes.
    :param k_param: Optional upper bound on the Schmidt rank when ``dim`` describes a bipartite
        system.  Set to :code:`0` (default) to ignore the Schmidt rank constraint.  Must be
        non-negative and strictly less than the smaller subsystem dimension when used.
    :param seed: A seed used to instantiate numpy's random number generator.
    :return: A normalized column vector of shape ``(total_dim, 1)`` where ``total_dim`` equals
        :code:`dim` if ``dim`` is an integer and equals the product of entries in ``dim``
        otherwise.

    """
    gen = np.random.default_rng(seed=seed)
    if k_param < 0:
        msg = "k_param must be non-negative."
        raise ValueError(msg)

    if isinstance(dim, int):
        dims_seq: list[int] | None = None
        min_dim = dim
        total_dim = dim
    else:
        dims_seq = list(dim)
        if len(dims_seq) == 0:
            msg = "dim must not be empty when provided as a sequence."
            raise ValueError(msg)
        if not all(isinstance(val, int) and val > 0 for val in dims_seq):
            msg = "dim entries must be positive integers."
            raise ValueError(msg)
        min_dim = min(dims_seq)
        total_dim = int(np.prod(dims_seq))

    if 0 < k_param < min_dim:
        if isinstance(dim, int):
            dims_pair = [dim, dim]
        else:
            if len(dims_seq) != 2:
                msg = "When k_param > 0, dim must be an integer or a length-2 sequence."
                raise ValueError(msg)
            dims_pair = dims_seq

        psi = max_entangled(k_param, True, False).toarray()

        a_param = gen.random((dims_pair[0] * k_param, 1))
        b_param = gen.random((dims_pair[1] * k_param, 1))

        if not is_real:
            a_param = a_param + 1j * gen.random((dims_pair[0] * k_param, 1))
            b_param = b_param + 1j * gen.random((dims_pair[1] * k_param, 1))

        mat_1 = np.kron(psi.conj().T, np.identity(int(np.prod(dims_pair))))
        mat_2 = swap(
            np.kron(a_param, b_param),
            sys=[2, 3],
            dim=[k_param, dims_pair[0], k_param, dims_pair[1]],
        )

        ret_vec = mat_1 @ mat_2
        ret_vec = ret_vec.reshape(-1, 1)
        return np.divide(ret_vec, np.linalg.norm(ret_vec))

    ret_vec = gen.random((total_dim, 1))
    if not is_real:
        ret_vec = ret_vec + 1j * gen.random((total_dim, 1))
    return np.divide(ret_vec, np.linalg.norm(ret_vec))
